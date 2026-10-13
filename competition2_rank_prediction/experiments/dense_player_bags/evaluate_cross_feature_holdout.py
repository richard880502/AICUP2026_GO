"""Evaluates the validation-locked cross-feature bag fusion on holdout.

The original 414D bag model and the 606D move-augmented bag model use
different embedding sources. Their predictions are fused with weights that
were selected on validation cohorts only. This module performs no holdout
search or training.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.evaluate_cross_feature_holdout \
      --device cpu
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import torch

from experiments.bag_distribution import run as bag_run
from experiments.dense_player_bags import evaluate_holdout
from experiments.dense_player_bags import evaluate_validation
from experiments.dense_player_bags import train_bags
from experiments.hierarchical_move import evaluate_fusion
from experiments.model_selection import evaluate as model_selection

import rank_lib


@dataclasses.dataclass(frozen=True)
class EvaluationPaths:
    """Contains both embedding sources and model checkpoints."""

    original_embeddings: pathlib.Path
    move_embeddings: pathlib.Path
    original_checkpoint: pathlib.Path
    move_checkpoint: pathlib.Path


@dataclasses.dataclass(frozen=True)
class LockedWeights:
    """Contains all validation-selected fusion coefficients."""

    move_bag_weight: float
    production_weight: float
    main_reference_weight: float


@dataclasses.dataclass(frozen=True)
class EvaluationConfiguration:
    """Contains immutable inputs for one locked evaluation."""

    paths: EvaluationPaths
    weights: LockedWeights
    device: torch.device


@dataclasses.dataclass(frozen=True)
class HoldoutPredictions:
    """Contains aligned predictions and question metadata."""

    production: np.ndarray
    original_bag: np.ndarray
    move_bag: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray


def _utility(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Returns per-question competition utility."""
    return bag_run.utility_scores(rank_lib.decode(probabilities), labels)


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Reports a paired candidate-minus-reference bootstrap comparison."""
    difference = _utility(candidate, labels) - _utility(reference, labels)
    delta = model_selection.bootstrap_interval(difference, seed)
    return {
        "delta": delta[0],
        "delta_95_percent_ci": [delta[1], delta[2]],
    }


def _variant_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, Any]:
    """Summarizes one fixed prediction variant."""
    return {
        "score": float(_utility(probabilities, labels).mean()),
        "by_games": evaluate_fusion.score_by_size(
            probabilities,
            labels,
            sizes,
        ),
    }


def _load_model_predictions(
    split: dict[str, np.ndarray],
    checkpoint: pathlib.Path,
    questions: list[tuple[str, np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads one dense bag model and predicts all holdout questions."""
    embeddings = split["embeddings"].astype(np.float32)
    model = evaluate_validation.load_model(
        checkpoint,
        embeddings.shape[1],
        device,
    )
    return evaluate_holdout.bag_questions(
        [model],
        embeddings,
        questions,
        device,
    )


def _load_predictions(
    configuration: EvaluationConfiguration,
) -> HoldoutPredictions:
    """Loads aligned production and two dense-bag predictions."""
    original = train_bags.load_split(
        configuration.paths.original_embeddings / "holdout.npz",
    )
    move = train_bags.load_split(
        configuration.paths.move_embeddings / "holdout.npz",
    )
    holdout, questions = evaluate_holdout.load_questions()
    production_keys = holdout["game_key"].astype(str)
    if not np.array_equal(original["game_key"].astype(str), production_keys):
        raise ValueError("Original and production holdout games are misaligned")
    if not np.array_equal(move["game_key"].astype(str), production_keys):
        raise ValueError("Move and production holdout games are misaligned")

    original_bag, sizes = _load_model_predictions(
        original,
        configuration.paths.original_checkpoint,
        questions,
        configuration.device,
    )
    move_bag, move_sizes = _load_model_predictions(
        move,
        configuration.paths.move_checkpoint,
        questions,
        configuration.device,
    )
    if not np.array_equal(sizes, move_sizes):
        raise ValueError("Original and move question sizes are misaligned")

    weights, offsets = evaluate_holdout.production_parameters()
    production = evaluate_holdout.production_questions(
        holdout,
        questions,
        weights,
        offsets,
    )
    labels = evaluate_holdout.label_questions(questions)
    return HoldoutPredictions(
        production,
        original_bag,
        move_bag,
        labels,
        sizes,
    )


def run_evaluation(
    configuration: EvaluationConfiguration,
) -> dict[str, Any]:
    """Runs the locked cross-feature evaluation and comparisons."""
    predictions = _load_predictions(configuration)
    cross_bag = bag_run.normalize_log_blend(
        predictions.original_bag,
        predictions.move_bag,
        configuration.weights.move_bag_weight,
    )
    main_reference = bag_run.normalize_log_blend(
        predictions.production,
        predictions.original_bag,
        configuration.weights.main_reference_weight,
    )
    cross_feature = bag_run.normalize_log_blend(
        predictions.production,
        cross_bag,
        configuration.weights.production_weight,
    )
    return {
        "production": _variant_report(
            predictions.production,
            predictions.labels,
            predictions.sizes,
        ),
        "original_bag": _variant_report(
            predictions.original_bag,
            predictions.labels,
            predictions.sizes,
        ),
        "move_bag": _variant_report(
            predictions.move_bag,
            predictions.labels,
            predictions.sizes,
        ),
        "cross_bag": _variant_report(
            cross_bag,
            predictions.labels,
            predictions.sizes,
        ),
        "main_reference": _variant_report(
            main_reference,
            predictions.labels,
            predictions.sizes,
        ),
        "cross_feature": _variant_report(
            cross_feature,
            predictions.labels,
            predictions.sizes,
        ),
        "cross_vs_production": _comparison(
            cross_feature,
            predictions.production,
            predictions.labels,
            20261020,
        ),
        "cross_vs_main_reference": _comparison(
            cross_feature,
            main_reference,
            predictions.labels,
            20261021,
        ),
    }


def main() -> None:
    """Evaluates exactly one pre-locked cross-feature configuration."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--original-embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings_logits",
    )
    parser.add_argument(
        "--move-embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings_logits_moves",
    )
    parser.add_argument(
        "--original-checkpoint",
        type=pathlib.Path,
        default=experiment / "results" / "transformer_logits" / "model.pt",
    )
    parser.add_argument(
        "--move-checkpoint",
        type=pathlib.Path,
        default=(
            experiment / "results" / "transformer_logits_moves" / "model.pt"
        ),
    )
    parser.add_argument("--move-bag-weight", type=float, default=0.30)
    parser.add_argument("--production-weight", type=float, default=0.28)
    parser.add_argument("--main-reference-weight", type=float, default=0.34)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            experiment
            / "results"
            / "cross_feature_locked"
            / "holdout_report.json"
        ),
    )
    args = parser.parse_args()
    for name in (
        "move_bag_weight",
        "production_weight",
        "main_reference_weight",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")

    report = {
        "protocol": {
            "holdout_weight_search": False,
            "move_bag_weight": args.move_bag_weight,
            "production_weight": args.production_weight,
            "main_reference_weight": args.main_reference_weight,
            "selection_validation_seed": 20261009,
            "locked_confirmation_seed": 20261019,
            "holdout_evaluations": 1,
        },
        "holdout": run_evaluation(
            EvaluationConfiguration(
                paths=EvaluationPaths(
                    original_embeddings=args.original_embeddings,
                    move_embeddings=args.move_embeddings,
                    original_checkpoint=args.original_checkpoint,
                    move_checkpoint=args.move_checkpoint,
                ),
                weights=LockedWeights(
                    move_bag_weight=args.move_bag_weight,
                    production_weight=args.production_weight,
                    main_reference_weight=args.main_reference_weight,
                ),
                device=torch.device(args.device),
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
