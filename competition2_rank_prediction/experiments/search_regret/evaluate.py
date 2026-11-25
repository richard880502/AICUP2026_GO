# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Tests searched regret beyond raw controls in full C1+C2.

Example: python -m competition2_rank_prediction.experiments.search_regret.evaluate
Requires complete original-validation extraction and refuses repeat evaluation.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.human_context import (
    evaluate as context_evaluation,
)
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.reward_alignment import (
    evaluate as fold_evaluation,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")
METHODS = ("original", "regularization", "raw", "search")
matplotlib.use("Agg")


def load_data(args) -> dict[str, np.ndarray]:
    """Loads original validation caches and aligned26Ddescriptors."""
    if not (args.output / "EXTRACTION_COMPLETE").exists():
        raise RuntimeError(
            "All search feature shards must finish before evaluation"
        )
    cache = (
        args.research.parents[1] / "production/rank_joint_20261108/prepared.npz"
    )
    with np.load(cache) as payload:
        data = {
            name: np.asarray(payload[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS)
        }
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("values", "game_key", "player_id", "rank", "masks")
    }
    for path in sorted(args.output.glob("feat_*.npz")):
        with np.load(path) as payload:
            for name in parts:
                parts[name].append(np.asarray(payload[name]))
    features = {name: np.concatenate(values) for name, values in parts.items()}
    if (
        len(features["game_key"]) != 8000
        or len(set(features["game_key"])) != 8000
    ):
        raise ValueError(
            "Search extraction must cover8000unique original-validation games"
        )
    lookup = {str(key): row for row, key in enumerate(features["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
    if not np.array_equal(
        features["player_id"][rows], data["player"]
    ) or not np.array_equal(
        rank_lib.rank_ids(features["rank"][rows]), data["y"]
    ):
        raise ValueError(
            "Search feature player/rank metadata differs from baseline"
        )
    data["context"] = features["values"][rows]
    if data["context"].shape != (8000, 26):
        raise ValueError("Expected26fixed raw/search features per game")
    return data


def fold_predictions(
    fitting: dict[str, np.ndarray], testing: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Fits all methods under the same player-disjoint C1+C2protocol."""
    fitting["context"], testing["context"] = context_evaluation.impute(
        fitting["context"], testing["context"]
    )
    base = calibration.fit(fitting, MEMBERS)
    train_logits, train_features, labels, _ = (
        context_evaluation.question_inputs(fitting, base.weights)
    )
    test_logits, test_features, truth, questions = (
        context_evaluation.question_inputs(testing, base.weights)
    )
    predictions = {"original": calibration.predict(base, testing, questions)}
    for name, stop in (
        ("regularization", train_features.shape[1] - 26),
        ("raw", train_features.shape[1] - 12),
        ("search", train_features.shape[1]),
    ):
        model = residual_shift.ShiftResidual().fit(
            train_logits, train_features[:, :stop], labels, lam=0.1
        )
        predictions[name] = model.predict_proba(
            test_logits, test_features[:, :stop]
        )
    players = np.asarray([testing["player"][rows[0]] for rows in questions])
    return predictions, truth, players


def cross_validate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cross-validates imputation, scaling and raw/search corrections."""
    folds = fold_evaluation.game_folds(data)
    arrays: dict[str, list[np.ndarray]] = {"y": [], "player": []}
    for fold in range(5):
        fitting = {name: values[folds != fold] for name, values in data.items()}
        testing = {name: values[folds == fold] for name, values in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Calibration player leakage")
        predictions, labels, players = fold_predictions(fitting, testing)
        arrays["y"].append(labels)
        arrays["player"].append(players)
        for method, values in predictions.items():
            arrays.setdefault(method, []).append(values)
        print(f"Search-regret calibration fold {fold + 1}/5", flush=True)
    return {name: np.concatenate(values) for name, values in arrays.items()}


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    """Separates search value from unusual-move/raw-quality selection."""
    rewards = {
        method: statistics.player_scores(arrays[method], arrays["y"])
        for method in METHODS
    }
    intervals = {}
    for reference in ("original", "raw"):
        delta = rewards["search"] - rewards[reference]
        clustered = np.asarray(
            [
                delta[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        )
        intervals[f"search_minus_{reference}"] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
    ranks = [
        float(
            (rewards["search"] - rewards["original"])[
                arrays["y"] == rank
            ].mean()
        )
        for rank in range(10)
    ]
    return {
        "scope": (
            "exploratory original400-player calibration CV, "
            "not Public or fresh confirmation"
        ),
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "rank_deltas": ranks,
        "screen_pass": intervals["search_minus_original"]["delta"] >= 0.005
        and intervals["search_minus_original"]["low"] > 0
        and intervals["search_minus_raw"]["low"] > 0
        and min(ranks) >= -0.05,
        "official_submission_created": False,
    }


def plot(report: dict, directory) -> None:
    """Displays both preregistered contrasts with player-clustered intervals."""
    figure, axis = plt.subplots(figsize=(8, 4))
    for index, reference in enumerate(("original", "raw")):
        interval = report["paired_player_97_5ci"][f"search_minus_{reference}"]
        axis.errorbar(
            interval["delta"],
            index,
            xerr=[
                [interval["delta"] - interval["low"]],
                [interval["high"] - interval["delta"]],
            ],
            fmt="o",
            capsize=4,
        )
    axis.axvline(0, color="gray")
    axis.set_yticks(
        (0, 1), ("Search minus original", "Search minus raw controls")
    )
    axis.set_xlabel("Official utility delta; paired-player97.5% interval")
    figure.tight_layout()
    figure.savefig(directory / "comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Evaluates once, preserving production and creating no submission."""
    args = common.parse_arguments(__doc__ or "", "search_regret_20261109")
    if (args.output / "report.json").exists():
        raise FileExistsError(
            "Refusing to overwrite completed search-regret evaluation"
        )
    torch.set_num_threads(4)
    data = load_data(args)
    arrays = cross_validate(data)
    report = summarize(arrays)
    report["missing_game_counts_per_feature"] = (
        (~np.isfinite(data["context"])).sum(axis=0).tolist()
    )
    if abs(report["utility"]["original"] - 0.7106057000201695) > 1e-12:
        raise RuntimeError("Original C1+C2 baseline did not replay exactly")
    plot(report, args.output)
    np.savez_compressed(
        args.output / "evaluation.npz", allow_pickle=False, **arrays
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
