# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Tests conditional HumanSL context beyond all frozen C1+C2 members.

Example: python -m competition2_rank_prediction.experiments.human_context.evaluate
Evaluates only complete game-key-aligned feature extraction.
"""

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.reward_alignment import evaluate
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")
METHODS = ("original", "regularization", "normal", "human")
matplotlib.use("Agg")


def game_statistics(tokens: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Returns normal20/human60 mean/std features, keeping missing games NaN."""
    if tokens.shape[:3] != masks.shape or tokens.shape[2:] != (4, 10):
        raise ValueError(
            "Expected context tokens [games, turns, 4, 10] and matching masks"
        )
    count = masks.sum(axis=1)[..., None]
    mean = (tokens * masks[..., None]).sum(axis=1) / np.maximum(count, 1)
    centered = tokens - mean[:, None]
    variance = (centered**2 * masks[..., None]).sum(axis=1) / np.maximum(
        count, 1
    )
    summaries = np.concatenate((mean, np.sqrt(variance)), axis=2)
    summaries = np.where(count > 0, summaries, np.nan)
    return summaries.reshape(len(tokens), 80)


def impute(
    fitting: np.ndarray, testing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fits column means only on fitting-player games, with safe empty columns."""
    valid = np.isfinite(fitting)
    count = valid.sum(axis=0)
    means = np.where(valid, fitting, 0).sum(axis=0) / np.maximum(count, 1)
    return (
        np.where(valid, fitting, means),
        np.where(np.isfinite(testing), testing, means),
    )


def load_data(args) -> dict[str, np.ndarray]:
    """Requires every original validation game and verifies metadata alignment."""
    cache = (
        args.research.parents[1] / "production/rank_joint_20261108/prepared.npz"
    )
    with np.load(cache) as payload:
        data = {
            name: np.asarray(payload[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS)
        }
    features = rank_lib.load_features(args.output)
    if len(features["game_key"]) != len(data["game_key"]) or set(
        features["game_key"]
    ) != set(data["game_key"]):
        raise ValueError(
            "Incomplete extraction: expected all 8000 validation games"
        )
    lookup = {str(key): row for row, key in enumerate(features["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
    if not np.array_equal(rank_lib.rank_ids(features["rank"][rows]), data["y"]):
        raise ValueError("Extracted rank metadata differs from baseline")
    if not np.array_equal(features["player_id"][rows], data["player"]):
        raise ValueError("Extracted player metadata differs from baseline")
    data["context"] = game_statistics(
        features["tokens"][rows], features["masks"][rows]
    )
    return data


def question_inputs(
    data: dict[str, np.ndarray], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Builds original C1 inputs and matching normalized context averages."""
    questions, labels = residual_shift.validation_questions(
        data["player"], data["y"]
    )
    logs = np.stack([np.log(data[name].clip(1e-6, 1)) for name in MEMBERS])
    logits, features = residual_shift.question_inputs(
        questions,
        blend_eval.fuse(logs, weights),
        data["d96"],
        data["opp_scratch"],
        data["final"],
    )
    context = np.stack(
        [data["context"][rows].mean(axis=0) for rows in questions]
    )
    return (
        logits,
        np.concatenate((features, context), axis=1),
        labels,
        questions,
    )


def fold_predictions(
    fitting: dict[str, np.ndarray], testing: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compares frozen production with matched shrinkage/normal/human controls."""
    fitting["context"], testing["context"] = impute(
        fitting["context"], testing["context"]
    )
    base = calibration.fit(fitting, MEMBERS)
    train_logits, train_features, train_labels, _ = question_inputs(
        fitting, base.weights
    )
    test_logits, test_features, labels, questions = question_inputs(
        testing, base.weights
    )
    predictions = {"original": calibration.predict(base, testing, questions)}
    for name, stop in (
        ("regularization", train_features.shape[1] - 80),
        ("normal", train_features.shape[1] - 60),
        ("human", train_features.shape[1]),
    ):
        model = residual_shift.ShiftResidual().fit(
            train_logits,
            train_features[:, :stop],
            train_labels,
            lam=0.1,
        )
        predictions[name] = model.predict_proba(
            test_logits, test_features[:, :stop]
        )
    players = np.asarray([testing["player"][rows[0]] for rows in questions])
    return predictions, labels, players


def run_folds(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keeps every calibration and imputation fit player-disjoint from evaluation."""
    folds = evaluate.game_folds(data)
    arrays: dict[str, list[np.ndarray]] = {"y": [], "player": []}
    for fold in range(5):
        fitting = {key: value[folds != fold] for key, value in data.items()}
        testing = {key: value[folds == fold] for key, value in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Calibration player overlap")
        predictions, labels, players = fold_predictions(fitting, testing)
        arrays["y"].append(labels)
        arrays["player"].append(players)
        for name, values in predictions.items():
            arrays.setdefault(name, []).append(values)
        print(f"Human context calibration fold {fold + 1}/5", flush=True)
    return {name: np.concatenate(values) for name, values in arrays.items()}


def report_results(arrays: dict[str, np.ndarray]) -> dict:
    """Reports both production benefit and HumanSL-specific incremental benefit."""
    rewards = {
        name: statistics.player_scores(arrays[name], arrays["y"])
        for name in METHODS
    }
    intervals = {}
    for reference in ("original", "normal"):
        delta = rewards["human"] - rewards[reference]
        clustered = np.asarray(
            [
                delta[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        )
        intervals[f"human_minus_{reference}"] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
    ranks = [
        float(
            (rewards["human"] - rewards["original"])[arrays["y"] == rank].mean()
        )
        for rank in range(10)
    ]
    return {
        "scope": "exploratory original 400-player validation cohort, not Public",
        "utility": {
            name: float(values.mean()) for name, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "rank_deltas": ranks,
        "screen_pass": intervals["human_minus_original"]["delta"] >= 0.005
        and intervals["human_minus_original"]["low"] > 0
        and intervals["human_minus_normal"]["low"] > 0
        and min(ranks) >= -0.05,
    }


def plot_results(report: dict, directory: pathlib.Path) -> None:
    """Renders baseline comparisons and player-clustered uncertainty."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    values = [report["utility"][name] for name in METHODS]
    axes[0].bar(METHODS, values)
    axes[0].set_ylim(min(values) - 0.005, max(values) + 0.005)
    axes[0].set_title("Full C1+C2 calibration CV")
    axes[0].set_ylabel("Official utility (truncated axis)")
    for index, reference in enumerate(("original", "normal")):
        interval = report["paired_player_97_5ci"][f"human_minus_{reference}"]
        center = interval["delta"] * 100
        axes[1].errorbar(
            center,
            index,
            xerr=[
                [center - interval["low"] * 100],
                [interval["high"] * 100 - center],
            ],
            fmt="o",
            capsize=4,
        )
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set_yticks((0, 1), ("Human minus original", "Human minus normal"))
    axes[1].set_xlabel("Utility difference ×100; player97.5% interval")
    figure.tight_layout()
    figure.savefig(directory / "comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Evaluates the fixed pilot once and records explicit missing-game coverage."""
    args = common.parse_arguments(__doc__ or "", "human_context_20261109")
    if (args.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite completed evaluation")
    torch.set_num_threads(4)
    data = load_data(args)
    arrays = run_folds(data)
    report = report_results(arrays)
    report["missing_feature_game_counts"] = (
        (~np.isfinite(data["context"])).sum(axis=0).tolist()
    )
    report["original_replay_matches"] = (
        abs(report["utility"]["original"] - 0.7106057000201695) < 1e-12
    )
    if not report["original_replay_matches"]:
        raise RuntimeError("Original C1+C2 baseline failed to replay")
    plot_results(report, args.output)
    np.savez_compressed(
        args.output / "evaluation.npz", allow_pickle=False, **arrays
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
