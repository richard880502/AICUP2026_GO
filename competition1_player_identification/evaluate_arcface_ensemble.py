"""Selects a candidate-only ensemble of ArcFace player-centroid retrievers.

Each model uses a different initialization seed but the same candidate-game
validation split.  The number of models is selected from that validation split;
the independent query cache is used only after all ensemble settings are fixed.

Example:
    .venv/bin/python evaluate_arcface_ensemble.py \\
      --profiles features/independent_s64_rank5k_profiles.npz \\
      --output results/arcface_ensemble.json --device cuda:0
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import torch

import evaluate_decision_features
import train_profile_metric


QUERY_GAME_COUNTS = (5, 10, 15, 20)


@dataclasses.dataclass(frozen=True)
class ModelSelection:
    """Stores one independently initialized candidate-only ArcFace model."""

    seed: int
    checkpoint: train_profile_metric.Checkpoint


@dataclasses.dataclass(frozen=True)
class ValidationTensors:
    """Stores standardized candidate references and candidate validation data."""

    train_profiles: torch.Tensor
    train_classes: torch.Tensor
    validation_profiles: torch.Tensor
    class_count: int


def parse_arguments() -> argparse.Namespace:
    """Parses profile-cache, ensemble, and training parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
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
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_metrics(scores: np.ndarray) -> dict[str, float]:
    """Computes competition metrics for player-aligned score matrices."""
    ranks = (scores > np.diag(scores)[:, None]).sum(axis=1) + 1
    official = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top5_exponential_score": float(np.mean(official)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def metrics_by_count(
    scores_by_count: dict[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Computes retrieval metrics at every official query-game count."""
    return {
        str(game_count): make_metrics(scores_by_count[game_count])
        for game_count in QUERY_GAME_COUNTS
    }


def embedding_scores(
    model: train_profile_metric.ArcFaceModel,
    candidate_profiles: torch.Tensor,
    candidate_classes: torch.Tensor,
    query_profiles: torch.Tensor,
    class_count: int,
) -> dict[int, np.ndarray]:
    """Computes cosine centroid scores for grouped query profiles."""
    model.eval()
    with torch.no_grad():
        candidate_embeddings = model.embeddings(candidate_profiles)
        candidate_centroids = train_profile_metric.make_centroids(
            candidate_embeddings,
            candidate_classes,
            class_count,
        )
        flat_queries = query_profiles.reshape(-1, query_profiles.shape[-1])
        query_embeddings = model.embeddings(flat_queries).reshape(
            query_profiles.shape[0],
            query_profiles.shape[1],
            -1,
        )
    scores = {}
    for game_count in QUERY_GAME_COUNTS:
        query_centroids = torch.nn.functional.normalize(
            query_embeddings[:, :game_count].mean(dim=1),
            dim=1,
        )
        scores[game_count] = query_centroids.matmul(
            candidate_centroids.T,
        ).cpu().numpy()
    return scores


def make_validation_tensors(
    data: train_profile_metric.ProfileData,
    split: train_profile_metric.CandidateSplit,
    device: torch.device,
) -> ValidationTensors:
    """Standardizes candidate references and validation games from references."""
    train_profiles, means, scales = train_profile_metric.standardize(
        data.candidates[split.train_indices],
        data.candidates[split.train_indices],
    )
    validation_profiles = np.nan_to_num(
        (data.candidates[split.validation_indices] - means) / scales,
    ).astype(np.float32)
    train_classes = train_profile_metric.make_class_ids(
        data.candidate_labels[split.train_indices],
        data.players,
    )
    return ValidationTensors(
        torch.from_numpy(train_profiles).to(device),
        torch.from_numpy(train_classes).to(device),
        torch.from_numpy(validation_profiles).to(device),
        len(data.players),
    )


def fit_validation_model(
    validation_tensors: ValidationTensors,
    metric_config: train_profile_metric.MetricConfig,
    seed: int,
) -> tuple[ModelSelection, dict[int, np.ndarray]]:
    """Fits one seeded ArcFace model and returns validation score matrices."""
    torch.manual_seed(seed)
    checkpoint = train_profile_metric.select_checkpoint(
        validation_tensors.train_profiles,
        validation_tensors.train_classes,
        validation_tensors.validation_profiles,
        metric_config,
        validation_tensors.class_count,
    )
    model = train_profile_metric.ArcFaceModel(
        validation_tensors.train_profiles.shape[1],
        metric_config.embedding_dimension,
        validation_tensors.class_count,
    ).to(validation_tensors.train_profiles.device)
    model.load_state_dict(checkpoint.state)
    return (
        ModelSelection(seed, checkpoint),
        embedding_scores(
            model,
            validation_tensors.train_profiles,
            validation_tensors.train_classes,
            validation_tensors.validation_profiles,
            validation_tensors.class_count,
        ),
    )


def average_score_matrices(
    score_sets: list[dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Averages row-normalized independent model scores."""
    if not score_sets:
        raise ValueError("score_sets must not be empty")
    scores = {}
    for game_count in QUERY_GAME_COUNTS:
        normalized = [
            evaluate_decision_features.standardize_score_rows(
                score_set[game_count],
            )
            for score_set in score_sets
        ]
        scores[game_count] = np.mean(np.stack(normalized), axis=0)
    return scores


def select_model_count(
    validation_sets: list[dict[int, np.ndarray]],
) -> tuple[int, dict[str, dict[str, float]]]:
    """Selects a prefix ensemble size from candidate-only validation scores."""
    best_count = 1
    best_metrics = metrics_by_count(average_score_matrices(validation_sets[:1]))
    best_score = train_profile_metric.average_score(best_metrics)
    for model_count in range(2, len(validation_sets) + 1):
        metrics = metrics_by_count(
            average_score_matrices(validation_sets[:model_count]),
        )
        score = train_profile_metric.average_score(metrics)
        if score > best_score:
            best_count = model_count
            best_metrics = metrics
            best_score = score
    return best_count, best_metrics


def group_query_profiles(
    data: train_profile_metric.ProfileData,
) -> np.ndarray:
    """Groups the first 20 query profiles in stable candidate-player order."""
    groups = []
    for player in data.players:
        player_profiles = data.queries[data.query_labels == player]
        if len(player_profiles) < max(QUERY_GAME_COUNTS):
            raise ValueError(f"Player {player!r} has fewer than 20 query games")
        groups.append(player_profiles[:max(QUERY_GAME_COUNTS)])
    return np.stack(groups)


def refit_query_model(
    data: train_profile_metric.ProfileData,
    selection: ModelSelection,
    metric_config: train_profile_metric.MetricConfig,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Refits one selected model on every candidate game and scores queries."""
    torch.manual_seed(selection.seed)
    candidates, means, scales = train_profile_metric.standardize(
        data.candidates,
        data.candidates,
    )
    candidate_tensor = torch.from_numpy(candidates).to(device)
    candidate_classes = torch.from_numpy(train_profile_metric.make_class_ids(
        data.candidate_labels,
        data.players,
    )).to(device)
    model = train_profile_metric.ArcFaceModel(
        candidates.shape[1],
        metric_config.embedding_dimension,
        len(data.players),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=metric_config.learning_rate,
        weight_decay=metric_config.weight_decay,
    )
    for _ in range(selection.checkpoint.epoch):
        train_profile_metric.run_epoch(
            model,
            optimizer,
            candidate_tensor,
            candidate_classes,
            metric_config,
        )
    query_profiles = np.nan_to_num(
        (group_query_profiles(data) - means) / scales,
    ).astype(np.float32)
    return embedding_scores(
        model,
        candidate_tensor,
        candidate_classes,
        torch.from_numpy(query_profiles).to(device),
        len(data.players),
    )


def select_ensemble(
    data: train_profile_metric.ProfileData,
    arguments: argparse.Namespace,
    metric_config: train_profile_metric.MetricConfig,
    device: torch.device,
) -> tuple[list[ModelSelection], int, dict[str, dict[str, float]]]:
    """Fits seeded validation models and selects a candidate-only prefix."""
    split = train_profile_metric.make_candidate_split(
        data,
        arguments.validation_games,
        arguments.split_seed,
    )
    validation_tensors = make_validation_tensors(data, split, device)
    selections = []
    validation_sets = []
    for seed in arguments.model_seeds:
        selection, validation_scores = fit_validation_model(
            validation_tensors,
            metric_config,
            seed,
        )
        selections.append(selection)
        validation_sets.append(validation_scores)
    model_count, metrics = select_model_count(validation_sets)
    return selections, model_count, metrics


def main() -> None:
    """Selects and evaluates a seeded ArcFace centroid ensemble."""
    arguments = parse_arguments()
    if arguments.validation_games < max(QUERY_GAME_COUNTS):
        raise ValueError("validation-games must be at least 20")
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if len(set(arguments.model_seeds)) != len(arguments.model_seeds):
        raise ValueError("model-seeds must be distinct")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    metric_config = train_profile_metric.MetricConfig(
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.embedding_dimension,
        arguments.margin,
        arguments.scale,
    )
    data = train_profile_metric.load_profiles(arguments.profiles)
    selections, model_count, validation_metrics = select_ensemble(
        data,
        arguments,
        metric_config,
        device,
    )
    query_sets = [
        refit_query_model(data, selection, metric_config, device)
        for selection in selections[:model_count]
    ]
    query_metrics = metrics_by_count(average_score_matrices(query_sets))
    results = {
        "method": "candidate_only_arcface_centroid_ensemble",
        "uses_query_labels_for_training": False,
        "split_seed": arguments.split_seed,
        "selected_model_count": model_count,
        "selected_models": [
            {
                "seed": selection.seed,
                "selected_epoch": selection.checkpoint.epoch,
                "validation_proxy": selection.checkpoint.score,
            }
            for selection in selections[:model_count]
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
