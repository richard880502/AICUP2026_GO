"""Selects a candidate-only ensemble across two ArcFace embedding sizes.

The candidate library determines both the per-dimension model count and the
blend coefficient.  The held-out query profiles are evaluated only after those
choices are frozen.

Example:
    .venv/bin/python evaluate_arcface_cross_dimension_ensemble.py \\
      --profiles features/independent_s64_rank5k_profiles.npz \\
      --output results/arcface_cross_dimension_ensemble.json --device cuda:0
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import torch

import evaluate_arcface_ensemble
import evaluate_decision_features
import train_profile_metric


@dataclasses.dataclass(frozen=True)
class DimensionEnsemble:
    """Stores selected models and validation scores for one embedding size."""

    embedding_dimension: int
    selections: list[evaluate_arcface_ensemble.ModelSelection]
    validation_scores: dict[int, np.ndarray]


def parse_arguments() -> argparse.Namespace:
    """Parses profile-cache, ArcFace, and reproducibility settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        nargs=2,
        default=(512, 1024),
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=(20260927, 20260928, 20260929),
    )
    parser.add_argument("--split-seed", type=int, default=20260927)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument(
        "--blend-by-query-count",
        action="store_true",
        help="Select a candidate-only blend weight separately for each query size.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_metric_config(
    arguments: argparse.Namespace,
    embedding_dimension: int,
) -> train_profile_metric.MetricConfig:
    """Builds a fixed ArcFace configuration for an embedding size."""
    return train_profile_metric.MetricConfig(
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.weight_decay,
        embedding_dimension,
        arguments.margin,
        arguments.scale,
    )


def select_dimension_ensemble(
    validation_tensors: evaluate_arcface_ensemble.ValidationTensors,
    arguments: argparse.Namespace,
    embedding_dimension: int,
) -> DimensionEnsemble:
    """Fits one embedding-size ensemble using candidate validation scores."""
    metric_config = make_metric_config(arguments, embedding_dimension)
    selections = []
    score_sets = []
    for seed in arguments.model_seeds:
        selection, scores = evaluate_arcface_ensemble.fit_validation_model(
            validation_tensors,
            metric_config,
            seed,
        )
        selections.append(selection)
        score_sets.append(scores)
    model_count, _ = evaluate_arcface_ensemble.select_model_count(score_sets)
    return DimensionEnsemble(
        embedding_dimension,
        selections[:model_count],
        evaluate_arcface_ensemble.average_score_matrices(score_sets[:model_count]),
    )


def blend_scores(
    first_scores: dict[int, np.ndarray],
    second_scores: dict[int, np.ndarray],
    second_weight: float,
) -> dict[int, np.ndarray]:
    """Blends row-standardized score matrices with a fixed coefficient."""
    blended = {}
    for game_count in train_profile_metric.QUERY_GAME_COUNTS:
        first_normalized = evaluate_decision_features.standardize_score_rows(
            first_scores[game_count],
        )
        second_normalized = evaluate_decision_features.standardize_score_rows(
            second_scores[game_count],
        )
        blended[game_count] = (
            (1.0 - second_weight) * first_normalized
            + second_weight * second_normalized
        )
    return blended


def blend_score_matrix(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    second_weight: float,
) -> np.ndarray:
    """Blends two standardized candidate-score matrices."""
    first_normalized = evaluate_decision_features.standardize_score_rows(
        first_scores,
    )
    second_normalized = evaluate_decision_features.standardize_score_rows(
        second_scores,
    )
    return (
        (1.0 - second_weight) * first_normalized
        + second_weight * second_normalized
    )


def select_blend_weight(
    first_scores: dict[int, np.ndarray],
    second_scores: dict[int, np.ndarray],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Selects a cross-dimension blend only from candidate validation games."""
    best_weight = 0.0
    best_metrics = evaluate_arcface_ensemble.metrics_by_count(first_scores)
    best_score = train_profile_metric.average_score(best_metrics)
    for second_weight in np.linspace(0.05, 1.0, 20):
        metrics = evaluate_arcface_ensemble.metrics_by_count(blend_scores(
            first_scores,
            second_scores,
            float(second_weight),
        ))
        score = train_profile_metric.average_score(metrics)
        if score > best_score:
            best_weight = float(second_weight)
            best_metrics = metrics
            best_score = score
    return best_weight, best_metrics


def select_per_count_blend_weights(
    first_scores: dict[int, np.ndarray],
    second_scores: dict[int, np.ndarray],
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    """Selects one candidate-only fusion weight for every published size."""
    weights = {}
    metrics_by_count = {}
    for game_count in train_profile_metric.QUERY_GAME_COUNTS:
        best_weight = 0.0
        best_metrics = evaluate_arcface_ensemble.make_metrics(
            first_scores[game_count],
        )
        best_score = best_metrics["top5_exponential_score"]
        for second_weight in np.linspace(0.05, 1.0, 20):
            metrics = evaluate_arcface_ensemble.make_metrics(blend_score_matrix(
                first_scores[game_count],
                second_scores[game_count],
                float(second_weight),
            ))
            score = metrics["top5_exponential_score"]
            if score > best_score:
                best_weight = float(second_weight)
                best_metrics = metrics
                best_score = score
        weights[game_count] = best_weight
        metrics_by_count[str(game_count)] = best_metrics
    return weights, metrics_by_count


def blend_scores_by_count(
    first_scores: dict[int, np.ndarray],
    second_scores: dict[int, np.ndarray],
    second_weights: dict[int, float],
) -> dict[int, np.ndarray]:
    """Blends score matrices using a separately selected weight per size."""
    return {
        game_count: blend_score_matrix(
            first_scores[game_count],
            second_scores[game_count],
            second_weights[game_count],
        )
        for game_count in train_profile_metric.QUERY_GAME_COUNTS
    }


def refit_dimension_ensemble(
    data: train_profile_metric.ProfileData,
    dimension_ensemble: DimensionEnsemble,
    arguments: argparse.Namespace,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Refits a frozen dimension ensemble on all candidates and scores queries."""
    metric_config = make_metric_config(
        arguments,
        dimension_ensemble.embedding_dimension,
    )
    score_sets = []
    for selection in dimension_ensemble.selections:
        score_sets.append(evaluate_arcface_ensemble.refit_query_model(
            data,
            selection,
            metric_config,
            device,
        ))
    return evaluate_arcface_ensemble.average_score_matrices(score_sets)


def serialize_dimension(
    dimension_ensemble: DimensionEnsemble,
) -> dict[str, object]:
    """Produces JSON-safe model selection metadata for one embedding size."""
    return {
        "embedding_dimension": dimension_ensemble.embedding_dimension,
        "selected_models": [
            {
                "seed": selection.seed,
                "selected_epoch": selection.checkpoint.epoch,
                "validation_proxy": selection.checkpoint.score,
            }
            for selection in dimension_ensemble.selections
        ],
    }


def main() -> None:
    """Runs candidate-only cross-dimension ArcFace ensemble evaluation."""
    arguments = parse_arguments()
    if arguments.validation_games < max(train_profile_metric.QUERY_GAME_COUNTS):
        raise ValueError("validation-games must be at least 20")
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if len(set(arguments.embedding_dimensions)) != len(arguments.embedding_dimensions):
        raise ValueError("embedding-dimensions must be distinct")
    if len(set(arguments.model_seeds)) != len(arguments.model_seeds):
        raise ValueError("model-seeds must be distinct")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data = train_profile_metric.load_profiles(arguments.profiles)
    split = train_profile_metric.make_candidate_split(
        data,
        arguments.validation_games,
        arguments.split_seed,
    )
    validation_tensors = evaluate_arcface_ensemble.make_validation_tensors(
        data,
        split,
        device,
    )
    first_ensemble = select_dimension_ensemble(
        validation_tensors,
        arguments,
        arguments.embedding_dimensions[0],
    )
    second_ensemble = select_dimension_ensemble(
        validation_tensors,
        arguments,
        arguments.embedding_dimensions[1],
    )
    if arguments.blend_by_query_count:
        blend_weights, validation_metrics = select_per_count_blend_weights(
            first_ensemble.validation_scores,
            second_ensemble.validation_scores,
        )
        blend_weight = None
    else:
        blend_weight, validation_metrics = select_blend_weight(
            first_ensemble.validation_scores,
            second_ensemble.validation_scores,
        )
        blend_weights = None
    first_query_scores = refit_dimension_ensemble(
        data,
        first_ensemble,
        arguments,
        device,
    )
    second_query_scores = refit_dimension_ensemble(
        data,
        second_ensemble,
        arguments,
        device,
    )
    if blend_weights is None:
        if blend_weight is None:
            raise RuntimeError("A global blend weight was not selected")
        query_scores = blend_scores(
            first_query_scores,
            second_query_scores,
            blend_weight,
        )
    else:
        query_scores = blend_scores_by_count(
            first_query_scores,
            second_query_scores,
            blend_weights,
        )
    query_metrics = evaluate_arcface_ensemble.metrics_by_count(query_scores)
    results = {
        "method": "candidate_only_cross_dimension_arcface_ensemble",
        "uses_query_labels_for_training": False,
        "split_seed": arguments.split_seed,
        "selected_second_dimension_weight": blend_weight,
        "selected_second_dimension_weights": blend_weights,
        "dimensions": [
            serialize_dimension(first_ensemble),
            serialize_dimension(second_ensemble),
        ],
        "validation_proxy": train_profile_metric.average_score(validation_metrics),
        "validation_metrics": validation_metrics,
        "query_proxy": train_profile_metric.average_score(query_metrics),
        "query_metrics": query_metrics,
    }
    evaluate_decision_features.result_io.write_json(arguments.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
