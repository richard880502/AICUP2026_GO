# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Runs the fixed bounded utility-fusion screen on held-player calibration folds.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Frozen original members and C1 are retained; only relative weights adapt.
"""

import hashlib
import json
import pathlib

import numpy as np
import torch

import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.dense_data_scale import (
    fusion_objective,
)
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.reward_alignment import evaluate
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")


def question_inputs(model, data: dict) -> dict:
    """Constructs fixed corrections and verifies exact-decision zero-delta parity."""
    questions, labels = residual_shift.validation_questions(
        data["player"], data["y"]
    )
    logs = np.stack([np.log(data[name].clip(1e-6, 1)) for name in MEMBERS])
    features, context = residual_shift.question_inputs(
        questions,
        blend_eval.fuse(logs, model.weights),
        data["d96"],
        data["opp_scratch"],
        data["final"],
    )
    offset = model.residual.logits(features, context) - features
    rows = np.zeros((len(questions), 20), dtype=int)
    mask = np.zeros(rows.shape, dtype=bool)
    for index, selected in enumerate(questions):
        rows[index, : len(selected)] = selected
        mask[index, : len(selected)] = True
    inputs = {"members": logs.transpose(1, 0, 2), "rows": rows, "mask": mask}
    original = calibration.predict(model, data, questions)
    replay = fusion_objective.predict(inputs, offset, model.weights)
    if not np.allclose(original, replay, atol=1e-6, rtol=0):
        raise ValueError("Zero-delta posterior parity failed")
    if not np.array_equal(rank_lib.decode(original), rank_lib.decode(replay)):
        raise ValueError("Zero-delta decision parity failed")
    return {
        "logs": inputs,
        "offset": offset,
        "y": labels,
        "original": original,
        "player": np.asarray([data["player"][rows[0]] for rows in questions]),
        "count": np.asarray([len(rows) for rows in questions]),
        "zero_delta_error": float(np.max(np.abs(original - replay))),
    }


def cross_validate(data: dict) -> tuple[dict, list]:
    """Fits both objectives only on each outer fold's320fitting players."""
    folds = evaluate.game_folds(data)
    outputs: dict = {
        name: []
        for name in ("original", "nll", "utility", "y", "player", "count")
    }
    records = []
    for fold in range(5):
        fitting = {name: values[folds != fold] for name, values in data.items()}
        testing = {name: values[folds == fold] for name, values in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Outer player overlap")
        model = calibration.fit(fitting, MEMBERS)
        training = question_inputs(model, fitting)
        held = question_inputs(model, testing)
        for name in ("original", "y", "player", "count"):
            outputs[name].append(held[name])
        record = {
            "fold": fold,
            "original_weights": model.weights.tolist(),
            "zero_delta_error": held["zero_delta_error"],
        }
        for method in ("nll", "utility"):
            weights = fusion_objective.fit(
                training["logs"],
                training["offset"],
                training["y"],
                model.weights,
                method,
            )
            outputs[method].append(
                fusion_objective.predict(held["logs"], held["offset"], weights)
            )
            record[f"{method}_weights"] = weights.tolist()
        records.append(record)
        print(f"Bounded fusion fold {fold+1}/5 complete", flush=True)
    return {
        name: np.concatenate(values) for name, values in outputs.items()
    }, records


def summarize(arrays: dict, records: list) -> dict:
    """Clusters repeated questions by player for both planned comparisons."""
    rewards = {
        method: statistics.player_scores(arrays[method], arrays["y"])
        for method in ("original", "nll", "utility")
    }
    intervals = {}
    changes = {}
    for reference in ("original", "nll"):
        delta = rewards["utility"] - rewards[reference]
        clustered = np.asarray(
            [
                delta[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        )
        intervals[reference] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
        changes[reference] = {
            "changed": int(
                np.sum(
                    rank_lib.decode(arrays["utility"])
                    != rank_lib.decode(arrays[reference])
                )
            ),
            "improved": int(np.sum(delta > 0)),
            "worsened": int(np.sum(delta < 0)),
        }
    delta = rewards["utility"] - rewards["original"]
    ranks = [float(delta[arrays["y"] == rank].mean()) for rank in range(10)]
    if abs(float(rewards["original"].mean()) - 0.7106057000201695) > 1e-12:
        raise ValueError("Original C1+C2 baseline did not replay")
    return {
        "scope": "exploratory historical400-player calibration CV, not Public",
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "changes": changes,
        "rank_deltas": ranks,
        "game_count_deltas": {
            f"{low}-{high}": float(
                delta[
                    (arrays["count"] >= low) & (arrays["count"] <= high)
                ].mean()
            )
            for low, high in ((5, 9), (10, 14), (15, 20))
        },
        "folds": records,
        "screen_pass": intervals["original"]["delta"] >= 0.005
        and intervals["original"]["low"] > 0
        and intervals["nll"]["low"] > 0
        and min(ranks) >= -0.05,
        "official_submission_created": False,
    }


def main() -> None:
    """Preserves all original production files and writes isolated predictions."""
    root = pathlib.Path.cwd()
    directory = root / "production/fusion_utility_20261109"
    if directory.exists():
        raise FileExistsError(f"Refusing existing fusion screen: {directory}")
    source = root / "production/rank_joint_20261108/prepared.npz"
    with np.load(source) as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS)
        }
    torch.set_num_threads(4)
    arrays, records = cross_validate(data)
    result = summarize(arrays, records)
    result["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "predictions.npz", **arrays)
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
