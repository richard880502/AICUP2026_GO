# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Evaluates only paired HD1final-value changes in the complete original pipeline.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Both new conditions refit C1 only on fitting players, never held players.
"""

import json
import pathlib

import numpy as np
import torch

import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.reward_alignment import (
    evaluate as fold_evaluate,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")


def splice(data: dict, roots: dict) -> None:
    """Changes only selected actor-perspective win/score columns, preserving facts."""
    lookup = {str(key): row for row, key in enumerate(data["game_key"])}
    keys = roots["game_key"]
    if len(set(keys)) != len(keys) or not set(keys).issubset(lookup):
        raise ValueError("Invalid HD1root game keys")
    rows = np.asarray([lookup[str(key)] for key in keys])
    if not np.array_equal(data["player"][rows], roots["player"]):
        raise ValueError("Root actor IDs differ from original validation")
    if not np.isin(roots["color"], ("B", "W")).all():
        raise ValueError("Invalid actor color in HD1root data")
    data["hd1"] = np.zeros(len(data["y"]), dtype=bool)
    data["hd1"][rows] = True
    sign = np.where(roots["color"] == "B", 1, -1)
    for condition, field in enumerate(("final65", "final05")):
        values = data["final"].copy()
        values[rows, 0] = np.where(
            sign == 1,
            roots["roots"][:, condition, 1],
            1 - roots["roots"][:, condition, 1],
        )
        values[rows, 1] = sign * roots["roots"][:, condition, 2]
        if not np.array_equal(
            values[:, 2:], data["final"][:, 2:], equal_nan=True
        ):
            raise ValueError("Komi splice changed SGF-derived facts")
        if not np.array_equal(
            values[~data["hd1"]], data["final"][~data["hd1"]], equal_nan=True
        ):
            raise ValueError("Komi splice changed an unselected game")
        data[field] = values


def cross_validate(data: dict) -> dict:
    """Uses identical questions/folds with unchanged original neural members."""
    folds = fold_evaluate.game_folds(data)
    output: dict = {
        name: []
        for name in (
            "y",
            "player",
            "count",
            "exposed",
            "original",
            "fresh",
            "komi05",
        )
    }
    for fold in range(5):
        fitting = {name: values[folds != fold] for name, values in data.items()}
        testing = {name: values[folds == fold] for name, values in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Player overlap in HD1calibration folds")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        output["y"].append(labels)
        output["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        output["count"].append(np.asarray([len(rows) for rows in questions]))
        output["exposed"].append(
            np.asarray([testing["hd1"][rows].any() for rows in questions])
        )
        for method, field in (
            ("original", "final"),
            ("fresh", "final65"),
            ("komi05", "final05"),
        ):
            fitting["final"] = data[field][folds != fold]
            testing["final"] = data[field][folds == fold]
            model = calibration.fit(fitting, MEMBERS)
            output[method].append(
                calibration.predict(model, testing, questions)
            )
        print(f"HD1final-value calibration fold {fold+1}/5", flush=True)
    return {name: np.concatenate(values) for name, values in output.items()}


def summarize(arrays: dict) -> dict:
    """Reports both fixed contrasts and exposed/unexposed player-query effects."""
    rewards = {
        method: statistics.player_scores(arrays[method], arrays["y"])
        for method in ("original", "fresh", "komi05")
    }
    intervals = {}
    for reference in ("original", "fresh"):
        delta = rewards["komi05"] - rewards[reference]
        clustered = np.asarray(
            [
                delta[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        )
        intervals[reference] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
    delta = rewards["komi05"] - rewards["original"]
    unexposed = float(delta[~arrays["exposed"]].mean())
    if abs(float(rewards["original"].mean()) - 0.7106057000201695) > 1e-12:
        raise ValueError("Original HD1comparison baseline failed exact replay")
    return {
        "scope": "exploratory original400-player HD1komi calibration CV; not Public",
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "exposed_questions": int(arrays["exposed"].sum()),
        "exposed_delta": float(delta[arrays["exposed"]].mean()),
        "unexposed_delta": unexposed,
        "rank_deltas": [
            float(delta[arrays["y"] == rank].mean()) for rank in range(10)
        ],
        "query_size_deltas": {
            f"{low}-{high}": float(
                delta[
                    (arrays["count"] >= low) & (arrays["count"] <= high)
                ].mean()
            )
            for low, high in ((5, 9), (10, 14), (15, 20))
        },
        "changes": {
            "improved": int((delta > 0).sum()),
            "worsened": int((delta < 0).sum()),
        },
        "screen_pass": intervals["original"]["delta"] >= 0.003
        and intervals["original"]["low"] > 0
        and intervals["fresh"]["low"] > 0
        and unexposed >= -0.005,
        "official_submission_created": False,
    }


def main() -> None:
    """Refuses incomplete extractions and preserves all production artifacts."""
    root = pathlib.Path.cwd()
    output = root / "production/hd1_komi_20261109"
    if (
        not (output / "EXTRACTION_COMPLETE").exists()
        or (output / "report.json").exists()
    ):
        raise ValueError(
            "HD1extraction is incomplete or evaluation already exists"
        )
    with np.load(root / "production/rank_joint_20261108/prepared.npz") as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS)
        }
    with np.load(output / "paired_roots.npz") as cache:
        roots = {name: np.asarray(cache[name]) for name in cache.files}
    if (
        roots["roots"].shape != (1542, 2, 3)
        or not np.isfinite(roots["roots"]).all()
        or (roots["roots"][:, :, 0] < 64).any()
    ):
        raise ValueError("Invalid fixed-cohort64visit roots")
    splice(data, roots)
    torch.set_num_threads(4)
    arrays = cross_validate(data)
    result = summarize(arrays)
    np.savez_compressed(output / "predictions.npz", **arrays)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
