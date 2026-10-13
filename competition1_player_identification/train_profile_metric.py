"""Trains a candidate-only ArcFace metric model on cached rich profiles.

The input profiles are the same multi-scale statistics used by the strongest
Gaussian retriever.  The model is trained only on the candidate library, while
the candidate-game validation split selects the epoch by the competition score.

Example:
    .venv/bin/python train_profile_metric.py \
      --profiles features/independent_s64_rank5k_profiles.npz \
      --output results/rich_profile_arcface.json --device cuda:0
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import torch

from go_retrieval import result_io


QUERY_GAME_COUNTS = (5, 10, 15, 20)


@dataclasses.dataclass(frozen=True)
class MetricConfig:
    """Controls the candidate-only ArcFace metric learner."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    embedding_dimension: int
    margin: float
    scale: float


@dataclasses.dataclass(frozen=True)
class ProfileData:
    """Stores rich profile arrays and stable anonymized player labels."""

    candidates: np.ndarray
    candidate_labels: np.ndarray
    queries: np.ndarray
    query_labels: np.ndarray
    players: np.ndarray


@dataclasses.dataclass(frozen=True)
class CandidateSplit:
    """Stores candidate-only reference and validation indices by player."""

    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """Stores the best candidate-only retrieval checkpoint."""

    state: dict[str, torch.Tensor]
    epoch: int
    score: float
    metrics: dict[str, dict[str, float]]


class ArcFaceModel(torch.nn.Module):
    """Maps game profiles to normalized embeddings and ArcFace class logits."""

    def __init__(
        self,
        input_dimension: int,
        embedding_dimension: int,
        class_count: int,
    ) -> None:
        super().__init__()
        hidden_dimension = min(1024, max(256, input_dimension // 2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dimension, embedding_dimension),
        )
        self.class_weights = torch.nn.Parameter(
            torch.empty(class_count, embedding_dimension),
        )
        torch.nn.init.xavier_uniform_(self.class_weights)

    def embeddings(self, profiles: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized player-style embeddings."""
        return torch.nn.functional.normalize(self.encoder(profiles), dim=1)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        """Returns normalized game embeddings for standard module use."""
        return self.embeddings(profiles)

    def logits(
        self,
        profiles: torch.Tensor,
        labels: torch.Tensor | None,
        margin: float,
        scale: float,
    ) -> torch.Tensor:
        """Returns cosine logits with ArcFace margin only on target classes."""
        cosine = self.embeddings(profiles).matmul(
            torch.nn.functional.normalize(self.class_weights, dim=1).T,
        )
        if labels is None or margin == 0.0:
            return cosine * scale
        target_cosine = cosine.gather(1, labels[:, None]).clamp(-1.0, 1.0)
        target_angle = torch.acos(target_cosine)
        adjusted_target = torch.cos(target_angle + margin)
        adjusted = cosine.scatter(1, labels[:, None], adjusted_target)
        return adjusted * scale


def parse_arguments() -> argparse.Namespace:
    """Parses cached-profile and metric-learning settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_profiles(path: pathlib.Path) -> ProfileData:
    """Loads profiles materialized by cache_rich_profiles.py."""
    with np.load(path) as payload:
        candidates = np.asarray(payload["candidate_profiles"], dtype=np.float32)
        candidate_labels = np.asarray(payload["candidate_labels"])
        queries = np.asarray(payload["query_profiles"], dtype=np.float32)
        query_labels = np.asarray(payload["query_labels"])
    return ProfileData(
        candidates,
        candidate_labels,
        queries,
        query_labels,
        np.unique(candidate_labels),
    )


def make_class_ids(labels: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Maps stable player identifiers onto contiguous training classes."""
    class_ids = np.searchsorted(players, labels)
    if not np.array_equal(players[class_ids], labels):
        raise ValueError("Profile label absent from candidate player pool")
    return class_ids.astype(np.int64, copy=False)


def make_candidate_split(
    data: ProfileData,
    validation_games: int,
    seed: int,
) -> CandidateSplit:
    """Reserves equal candidate-game validation groups for every player."""
    generator = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for player in data.players:
        player_indices = np.flatnonzero(data.candidate_labels == player)
        if len(player_indices) <= validation_games:
            raise ValueError(f"Player {player!r} has insufficient candidate games")
        shuffled = generator.permutation(player_indices)
        train_indices.append(shuffled[validation_games:])
        validation_indices.append(shuffled[:validation_games])
    return CandidateSplit(np.concatenate(train_indices), np.stack(validation_indices))


def standardize(
    reference_profiles: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits candidate-only normalization and applies it to one profile matrix."""
    means = reference_profiles.mean(axis=0)
    scales = reference_profiles.std(axis=0).clip(min=1e-6)
    normalized = np.nan_to_num((profiles - means) / scales).astype(np.float32)
    return normalized, means, scales


def make_metrics(scores: np.ndarray) -> dict[str, float]:
    """Computes competition metrics from aligned player score rows."""
    ranks = (scores > np.diag(scores)[:, None]).sum(axis=1) + 1
    official = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top5_exponential_score": float(np.mean(official)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def average_score(metrics: dict[str, dict[str, float]]) -> float:
    """Averages the official score across the published query lengths."""
    scores = []
    for game_count in QUERY_GAME_COUNTS:
        scores.append(metrics[str(game_count)]["top5_exponential_score"])
    return float(np.mean(scores))


def make_centroids(
    embeddings: torch.Tensor,
    classes: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    """Aggregates normalized game embeddings into normalized player centroids."""
    sums = torch.zeros(
        class_count,
        embeddings.shape[1],
        device=embeddings.device,
    )
    sums.index_add_(0, classes, embeddings)
    counts = torch.bincount(classes, minlength=class_count).clamp_min(1)
    return torch.nn.functional.normalize(sums / counts[:, None], dim=1)


def evaluate_embeddings(
    model: ArcFaceModel,
    candidate_games: torch.Tensor,
    candidate_classes: torch.Tensor,
    query_games: torch.Tensor,
    class_count: int,
) -> dict[str, dict[str, float]]:
    """Scores candidate centroids against multiple query-game aggregates."""
    model.eval()
    with torch.no_grad():
        candidate_embeddings = model.embeddings(candidate_games)
        candidate_centroids = make_centroids(
            candidate_embeddings,
            candidate_classes,
            class_count,
        )
        query_embeddings = model.embeddings(query_games.reshape(
            -1,
            query_games.shape[-1],
        )).reshape(query_games.shape[0], query_games.shape[1], -1)
    metrics = {}
    for game_count in QUERY_GAME_COUNTS:
        query_centroids = torch.nn.functional.normalize(
            query_embeddings[:, :game_count].mean(dim=1),
            dim=1,
        )
        metrics[str(game_count)] = make_metrics(
            query_centroids.matmul(candidate_centroids.T).cpu().numpy(),
        )
    return metrics


def run_epoch(
    model: ArcFaceModel,
    optimizer: torch.optim.Optimizer,
    profiles: torch.Tensor,
    labels: torch.Tensor,
    config: MetricConfig,
) -> None:
    """Runs one shuffled ArcFace cross-entropy training epoch."""
    model.train()
    permutation = torch.randperm(len(profiles), device=profiles.device)
    for indices in permutation.split(config.batch_size):
        logits = model.logits(
            profiles[indices],
            labels[indices],
            config.margin,
            config.scale,
        )
        loss = torch.nn.functional.cross_entropy(logits, labels[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def select_checkpoint(
    train_profiles: torch.Tensor,
    train_classes: torch.Tensor,
    validation_games: torch.Tensor,
    config: MetricConfig,
    class_count: int,
) -> Checkpoint:
    """Fits ArcFace and selects the best candidate-only retrieval epoch."""
    model = ArcFaceModel(train_profiles.shape[1], config.embedding_dimension, class_count)
    model = model.to(train_profiles.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint = None
    for epoch in range(1, config.epochs + 1):
        run_epoch(model, optimizer, train_profiles, train_classes, config)
        metrics = evaluate_embeddings(
            model,
            train_profiles,
            train_classes,
            validation_games,
            class_count,
        )
        score = average_score(metrics)
        if checkpoint is None or score > checkpoint.score:
            checkpoint = Checkpoint(
                {name: value.detach().clone() for name, value in model.state_dict().items()},
                epoch,
                score,
                metrics,
            )
    if checkpoint is None:
        raise RuntimeError("No metric checkpoint was selected")
    return checkpoint


def group_query_games(
    profiles: np.ndarray,
    labels: np.ndarray,
    players: np.ndarray,
) -> np.ndarray:
    """Groups the first 20 offline query profiles in player order."""
    grouped = []
    for player in players:
        player_profiles = profiles[labels == player]
        if len(player_profiles) < max(QUERY_GAME_COUNTS):
            raise ValueError(f"Player {player!r} has fewer than 20 query games")
        grouped.append(player_profiles[:max(QUERY_GAME_COUNTS)])
    return np.stack(grouped)


def refit_and_evaluate(
    data: ProfileData,
    checkpoint: Checkpoint,
    config: MetricConfig,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Refits selected epochs on all candidates and scores offline queries."""
    torch.manual_seed(seed)
    candidates, means, scales = standardize(data.candidates, data.candidates)
    candidate_tensor = torch.from_numpy(candidates).to(device)
    candidate_classes = torch.from_numpy(
        make_class_ids(data.candidate_labels, data.players),
    ).to(device)
    model = ArcFaceModel(candidates.shape[1], config.embedding_dimension, len(data.players))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(checkpoint.epoch):
        run_epoch(model, optimizer, candidate_tensor, candidate_classes, config)
    query_profiles = np.nan_to_num(
        (group_query_games(data.queries, data.query_labels, data.players) - means)
        / scales,
    ).astype(np.float32)
    return evaluate_embeddings(
        model,
        candidate_tensor,
        candidate_classes,
        torch.from_numpy(query_profiles).to(device),
        len(data.players),
    )


def main() -> None:
    """Runs candidate-only checkpoint selection and frozen query evaluation."""
    arguments = parse_arguments()
    if arguments.validation_games < max(QUERY_GAME_COUNTS):
        raise ValueError("validation-games must be at least 20")
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    config = MetricConfig(
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.embedding_dimension,
        arguments.margin,
        arguments.scale,
    )
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data = load_profiles(arguments.profiles)
    split = make_candidate_split(data, arguments.validation_games, arguments.seed)
    train_profiles, means, scales = standardize(
        data.candidates[split.train_indices],
        data.candidates[split.train_indices],
    )
    validation_profiles = np.nan_to_num(
        (data.candidates[split.validation_indices] - means) / scales,
    ).astype(np.float32)
    checkpoint = select_checkpoint(
        torch.from_numpy(train_profiles).to(device),
        torch.from_numpy(
            make_class_ids(data.candidate_labels[split.train_indices], data.players),
        ).to(device),
        torch.from_numpy(validation_profiles).to(device),
        config,
        len(data.players),
    )
    query_metrics = refit_and_evaluate(data, checkpoint, config, arguments.seed, device)
    results = {
        "method": "candidate_only_rich_profile_arcface",
        "uses_query_labels_for_training": False,
        "embedding_dimension": config.embedding_dimension,
        "margin": config.margin,
        "scale": config.scale,
        "selected_epoch": checkpoint.epoch,
        "validation_metrics": checkpoint.metrics,
        "validation_proxy": checkpoint.score,
        "query_metrics": query_metrics,
        "query_proxy": average_score(query_metrics),
    }
    result_io.write_json(arguments.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
