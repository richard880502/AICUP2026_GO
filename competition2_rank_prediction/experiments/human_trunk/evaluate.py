# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Tests the frozen trunk probe as an additional original C1+C2 member.

Example: python -m competition2_rank_prediction.experiments.human_trunk.evaluate
All neural/probe training stays frozen; calibration uses player-disjoint folds.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.reward_alignment import evaluate
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch", "human_trunk")
matplotlib.use("Agg")


def load_data(args) -> dict[str, np.ndarray]:
    """Aligns the fixed probe to original read-only validation-member caches."""
    cache = (
        args.research.parents[1] / "production/rank_joint_20261108/prepared.npz"
    )
    with np.load(cache) as payload:
        data = {
            name: np.asarray(payload[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS[:3])
        }
    with np.load(args.output / "probe_validation.npz") as probe:
        lookup = {str(key): row for row, key in enumerate(probe["game_key"])}
        rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
        if not np.array_equal(
            np.asarray(probe["y"])[rows], data["y"]
        ) or not np.array_equal(
            np.asarray(probe["player"])[rows], data["player"]
        ):
            raise ValueError(
                "Probe validation labels/players do not match baseline"
            )
        data["human_trunk"] = np.asarray(probe["probabilities"])[rows]
    return data


def cross_validate(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Refits original or four-member fusion/C1 only on each fitting-player fold."""
    folds = evaluate.game_folds(data)
    arrays: dict[str, list[np.ndarray]] = {"y": [], "player": [], "count": []}
    records = []
    for fold in range(5):
        fitting = {name: values[folds != fold] for name, values in data.items()}
        testing = {name: values[folds == fold] for name, values in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Calibration player overlap")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        arrays["y"].append(labels)
        arrays["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        arrays["count"].append(np.asarray([len(rows) for rows in questions]))
        for method, members in (
            ("baseline", MEMBERS[:3]),
            ("candidate", MEMBERS),
        ):
            model = calibration.fit(fitting, members)
            arrays.setdefault(method, []).append(
                calibration.predict(model, testing, questions)
            )
            records.append(
                {
                    "fold": fold,
                    "method": method,
                    "weights": model.weights.tolist(),
                }
            )
        print(f"Trunk conditional calibration fold {fold + 1}/5", flush=True)
    return {
        name: np.concatenate(values) for name, values in arrays.items()
    }, records


def summarize(arrays: dict[str, np.ndarray], records: list[dict]) -> dict:
    """Reports conditional utility and clustered uncertainty under the fixed gate."""
    rewards = {
        method: statistics.player_scores(arrays[method], arrays["y"])
        for method in ("baseline", "candidate")
    }
    delta = rewards["candidate"] - rewards["baseline"]
    clustered = np.asarray(
        [
            delta[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    interval = statistics.bootstrap_players(clustered, confidence=0.975)
    ranks = [float(delta[arrays["y"] == rank].mean()) for rank in range(10)]
    return {
        "scope": "exploratory original400-player calibration CV, not Public or fresh confirmation",
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": interval,
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
        "screen_pass": interval["delta"] >= 0.005
        and interval["low"] > 0
        and min(ranks) >= -0.05,
        "official_submission_created": False,
    }


def main() -> None:
    """Evaluates exactly one fixed probe, preserving original production artifacts."""
    args = common.parse_arguments(__doc__ or "", "human_trunk_20261109")
    if (args.output / "report.json").exists():
        raise FileExistsError(
            "Refusing to overwrite completed conditional evaluation"
        )
    torch.set_num_threads(4)
    arrays, records = cross_validate(load_data(args))
    report = summarize(arrays, records)
    if abs(report["utility"]["baseline"] - 0.7106057000201695) > 1e-12:
        raise RuntimeError("Original C1+C2 baseline failed to replay")
    figure, axis = plt.subplots(figsize=(7, 4))
    interval = report["paired_player_97_5ci"]
    axis.errorbar(
        interval["delta"],
        0,
        xerr=[
            [interval["delta"] - interval["low"]],
            [interval["high"] - interval["delta"]],
        ],
        fmt="o",
        capsize=5,
    )
    axis.axvline(0, color="gray")
    axis.set_yticks([0], ["C1+C2 + trunk minus original"])
    axis.set_xlabel("Official utility delta; paired-player97.5% interval")
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(
        args.output / "evaluation.npz", allow_pickle=False, **arrays
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
