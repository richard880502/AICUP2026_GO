"""Compares the locked two-model and historical four-model pipelines.

Both variants fit weights and class offsets on validation players only. The
script reports paired bootstrap uncertainty on the 400-player holdout.

Example:
    c1/.venv/bin/python -m experiments.model_selection.evaluate
"""

import json
import pathlib
from typing import Any

import numpy as np

import blend_eval
import rank_lib
from experiments.bag_distribution import run as bag_run

ROOT = pathlib.Path(__file__).resolve().parents[2]
TWO_MODELS = ("d96_h", "resnet8c_h")
FOUR_MODELS = ("d96_h", "resnet8c_h", "resnet5_h", "move_h")


def validation_data(
    models: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads aligned validation log-probabilities for selected models."""
    logs = []
    reference: dict[str, np.ndarray] | None = None
    for model in models:
        path = ROOT / "results" / f"val_probs_{model}.npz"
        with np.load(path, allow_pickle=True) as data:
            key = "transformer" if model.startswith("d96") else model
            logs.append(np.log(np.clip(data[key], 1e-6, 1.0)))
            if reference is None:
                reference = {
                    name: data[name]
                    for name in ("y", "player", "game_key")
                }
            elif not np.array_equal(data["game_key"], reference["game_key"]):
                raise ValueError(f"Validation ordering differs for {model!r}")
    if reference is None:
        raise ValueError("At least one model is required")
    return (
        np.stack(logs),
        reference["y"].astype(np.int64),
        reference["player"].astype(str),
    )


def fit_offsets(
    fused: np.ndarray,
    players: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fits the production class offsets from simulated validation bags."""
    rng = np.random.default_rng(seed)
    rows_by_player = bag_run.rows_by_group(players)
    probabilities = []
    for _ in range(10):
        for rows in rows_by_player.values():
            size = min(int(rng.integers(5, 21)), len(rows))
            selected = rng.choice(rows, size=size, replace=False)
            probabilities.append(rank_lib.pool(fused[selected], "logmean"))
    return rank_lib.fit_class_offsets(np.stack(probabilities))


def evaluate_variant(
    models: tuple[str, ...],
    holdout: dict[str, np.ndarray],
    holdout_bags: list[tuple[str, np.ndarray]],
    truth: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fits validation-only parameters and scores one holdout variant."""
    logs, labels, players = validation_data(models)
    weights = blend_eval.fit_weights(logs, labels)
    offsets = fit_offsets(blend_eval.fuse(logs, weights), players, 0)
    question_probabilities = holdout_question_probabilities(
        models,
        holdout,
        holdout_bags,
        weights,
    )
    calibrated = rank_lib.apply_class_offsets(
        question_probabilities,
        offsets,
    )
    prediction = rank_lib.decode(calibrated)
    utility = bag_run.utility_scores(prediction, truth)
    sizes = np.asarray([len(rows) for _, rows in holdout_bags])
    report = {
        "models": list(models),
        "score": float(utility.mean()),
        "weights": {
            model: float(weight)
            for model, weight in zip(models, weights)
        },
        "offsets": offsets.tolist(),
        "by_question_size": {
            "5_to_9": float(utility[sizes < 10].mean()),
            "10_to_14": float(
                utility[(sizes >= 10) & (sizes < 15)].mean(),
            ),
            "15_to_20": float(utility[sizes >= 15].mean()),
        },
    }
    return utility, report


def holdout_question_probabilities(
    models: tuple[str, ...],
    holdout: dict[str, np.ndarray],
    holdout_bags: list[tuple[str, np.ndarray]],
    weights: np.ndarray,
) -> np.ndarray:
    """Fuses holdout games and pools each multi-game question."""
    holdout_logs = np.stack([
        np.log(np.clip(holdout[model], 1e-6, 1.0))
        for model in models
    ])
    holdout_fused = blend_eval.fuse(holdout_logs, weights)
    return np.stack([
        rank_lib.pool(holdout_fused[rows], "logmean")
        for _, rows in holdout_bags
    ])


def bootstrap_interval(
    difference: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Returns the mean and paired 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    estimates: np.ndarray = np.empty(10000, dtype=np.float64)
    for index, _ in enumerate(estimates):
        rows = rng.integers(0, len(difference), len(difference))
        estimates[index] = difference[rows].mean()
    return (
        float(difference.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def main() -> None:
    """Evaluates and records the two locked production candidates."""
    holdout, holdout_bags, truth = load_holdout_questions()
    two_utility, two_report = evaluate_variant(
        TWO_MODELS,
        holdout,
        holdout_bags,
        truth,
    )
    four_utility, four_report = evaluate_variant(
        FOUR_MODELS,
        holdout,
        holdout_bags,
        truth,
    )
    interval = bootstrap_interval(two_utility - four_utility, 20261005)
    report = {
        "selection_basis": (
            "five-fold player-disjoint question fusion assigned near-zero "
            "weight to move_h and resnet5_h"
        ),
        "two_model": two_report,
        "four_model": four_report,
        "two_minus_four": {
            "mean": interval[0],
            "95_percent_ci": [interval[1], interval[2]],
        },
    }
    output = pathlib.Path(__file__).resolve().parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {path}")


def load_holdout_questions(
) -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray]], np.ndarray]:
    """Loads holdout model outputs and aligned question labels."""
    holdout_path = ROOT / "features" / "holdout" / "probs_final.npz"
    with np.load(holdout_path, allow_pickle=True) as data:
        holdout = {key: data[key] for key in data.files}
    groups = holdout["player_id"].astype(str)
    rows_by_question = bag_run.rows_by_group(groups)
    holdout_bags = list(sorted(rows_by_question.items()))
    _, _, game_labels = bag_run.load_holdout()
    truth = np.asarray([
        game_labels[rows[0]]
        for _, rows in holdout_bags
    ])
    return holdout, holdout_bags, truth


if __name__ == "__main__":
    main()
