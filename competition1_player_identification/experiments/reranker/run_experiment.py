"""Trains a candidate-only hard-negative reranker and evaluates it once.

Synthetic queries are sampled only from the candidate library.  Query-owner
players are split before task generation, so no player's synthetic queries
appear in both development partitions.  Hyperparameters and Top-K are chosen
on this candidate-only validation set.  The independent query cohort is read
only after all choices are frozen.

Example:
    .venv/bin/python -m experiments.reranker.run_experiment
"""

import argparse
import dataclasses
import json
import pathlib
import typing

import numpy as np
# scikit-learn 1.9.1 does not publish a py.typed marker.
import sklearn.ensemble  # type: ignore[import-untyped]
import torch


QUERY_COUNTS = (5, 10, 15, 20)
RECALL_COUNTS = (5, 10, 20, 30, 50)
DEFAULT_EMBEDDINGS = pathlib.Path(
    "experiments/reranker/artifacts/frozen_game_embeddings.npz",
)
DEFAULT_SELECTION = pathlib.Path(
    "results/best/arcface_cross_dimension_ensemble.json",
)
DEFAULT_OUTPUT = pathlib.Path(
    "experiments/reranker/results/first_reranker.json",
)
FEATURE_NAMES = (
    "arc512",
    "arc1024",
    "arc_ensemble",
    "stage1_rank",
    "candidate_minus_top1",
    "candidate_minus_top2",
    "top1_minus_candidate",
    "query_count",
    "g2g_max_mean",
    "g2g_max_std",
    "g2g_max_min",
    "g2g_max_median",
    "g2g_top3_mean",
    "g2g_top3_min",
    "g2g_top5_mean",
    "g2g_top5_min",
    "g2g_top10_mean",
    "g2g_all_mean",
    "g2g_all_std",
    "g2g_q10",
    "g2g_q25",
    "g2g_q50",
    "g2g_q75",
    "g2g_q90",
)


@dataclasses.dataclass(frozen=True)
class EmbeddingBranch:
    """Stores one development or final embedding branch."""

    candidates_512: np.ndarray
    candidates_1024: np.ndarray
    queries_512: np.ndarray
    queries_1024: np.ndarray


@dataclasses.dataclass(frozen=True)
class EmbeddingData:
    """Stores player-grouped development and final embeddings."""

    players: np.ndarray
    development: EmbeddingBranch
    final: EmbeddingBranch
    development_reference_positions: np.ndarray
    development_holdout_positions: np.ndarray


@dataclasses.dataclass(frozen=True)
class Task:
    """Defines one synthetic or independent multi-game query."""

    owner: int
    game_count: int
    query_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class PairRows:
    """Stores the tabular query-candidate rows."""

    features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    owners: np.ndarray
    counts: np.ndarray
    candidate_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class PairDataset:
    """Stores grouped pair rows and their Stage-1 context."""

    rows: PairRows
    stage1_orders: list[np.ndarray]
    stage1_scores: list[np.ndarray]

    @property
    def features(self) -> np.ndarray:
        """Returns model feature rows."""
        return self.rows.features

    @property
    def labels(self) -> np.ndarray:
        """Returns binary pair labels."""
        return self.rows.labels

    @property
    def group_ids(self) -> np.ndarray:
        """Returns query group identifiers."""
        return self.rows.group_ids

    @property
    def owners(self) -> np.ndarray:
        """Returns the owner index for each pair row."""
        return self.rows.owners

    @property
    def counts(self) -> np.ndarray:
        """Returns query game counts for each pair row."""
        return self.rows.counts

    @property
    def candidate_ids(self) -> np.ndarray:
        """Returns candidate indices for each pair row."""
        return self.rows.candidate_ids


def parse_arguments() -> argparse.Namespace:
    """Parses cache, sampling, model, and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=pathlib.Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--selection", type=pathlib.Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--samples-per-count", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--weight-mode",
        choices=("global", "per-count"),
        default="global",
    )
    return parser.parse_args()


def _standardize_scores(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std().clip(min=1e-6)


def load_embeddings(path: pathlib.Path) -> EmbeddingData:
    """Loads embeddings and groups candidate games by stable player order."""
    with np.load(path) as payload:
        players = np.asarray(payload["players"])
        labels = np.asarray(payload["candidate_labels"])
        development_arrays = (
            np.asarray(payload["development_candidate_512"], dtype=np.float32),
            np.asarray(payload["development_candidate_1024"], dtype=np.float32),
        )
        final_arrays = (
            np.asarray(payload["final_candidate_512"], dtype=np.float32),
            np.asarray(payload["final_candidate_1024"], dtype=np.float32),
        )
        query_arrays = (
            np.asarray(payload["final_query_512"], dtype=np.float32),
            np.asarray(payload["final_query_1024"], dtype=np.float32),
        )
        reference_positions = np.asarray(
            payload["development_reference_positions"],
            dtype=np.int64,
        )
        holdout_positions = np.asarray(
            payload["development_holdout_positions"],
            dtype=np.int64,
        )
    player_indices = [np.flatnonzero(labels == player) for player in players]
    game_counts = {len(indices) for indices in player_indices}
    if len(game_counts) != 1:
        raise ValueError("Candidate players do not have equal game counts")
    indices = np.stack(player_indices)
    development = EmbeddingBranch(
        development_arrays[0][:, indices],
        development_arrays[1][:, indices],
        development_arrays[0][:, indices],
        development_arrays[1][:, indices],
    )
    final = EmbeddingBranch(
        final_arrays[0][:, indices],
        final_arrays[1][:, indices],
        query_arrays[0],
        query_arrays[1],
    )
    return EmbeddingData(
        players,
        development,
        final,
        reference_positions,
        holdout_positions,
    )


def make_tasks(
    owners: np.ndarray,
    samples_per_count: int,
    holdout_positions: np.ndarray,
    generator: np.random.Generator,
) -> list[Task]:
    """Creates repeated candidate-only query/reference partitions."""
    tasks = []
    for owner in owners:
        for game_count in QUERY_COUNTS:
            for _ in range(samples_per_count):
                indices = generator.choice(
                    holdout_positions[int(owner)],
                    size=game_count,
                    replace=False,
                )
                tasks.append(Task(int(owner), game_count, np.sort(indices)))
    return tasks


@dataclasses.dataclass(frozen=True)
class PairBuildTensors:
    """Stores GPU tensors shared by every query task."""

    references_1024: torch.Tensor
    queries_512: torch.Tensor
    queries_1024: torch.Tensor
    centroids_512: torch.Tensor
    centroids_1024: torch.Tensor


@dataclasses.dataclass(frozen=True)
class PairBuildConfig:
    """Stores shared pair-feature construction settings."""

    top_k: int
    blend_weights: dict[int, float]
    independent: bool
    device: torch.device


@dataclasses.dataclass(frozen=True)
class TaskRows:
    """Stores all rows and ranking context for one query task."""

    features: np.ndarray
    labels: np.ndarray
    owners: np.ndarray
    counts: np.ndarray
    candidate_ids: np.ndarray
    order: np.ndarray
    stage1_scores: np.ndarray


def _reference_games(
    candidates: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    games = []
    for player, player_positions in enumerate(positions):
        games.append(candidates[:, player, player_positions])
    return np.stack(games, axis=1)


def _prepare_pair_tensors(
    data: EmbeddingData,
    independent: bool,
    device: torch.device,
) -> PairBuildTensors:
    branch = data.final if independent else data.development
    if independent:
        references_512 = branch.candidates_512
        references_1024 = branch.candidates_1024
    else:
        references_512 = _reference_games(
            branch.candidates_512,
            data.development_reference_positions,
        )
        references_1024 = _reference_games(
            branch.candidates_1024,
            data.development_reference_positions,
        )
    reference_512_tensor = torch.from_numpy(references_512).to(device)
    reference_1024_tensor = torch.from_numpy(references_1024).to(device)
    return PairBuildTensors(
        reference_1024_tensor,
        torch.from_numpy(branch.queries_512).to(device),
        torch.from_numpy(branch.queries_1024).to(device),
        torch.nn.functional.normalize(
            reference_512_tensor.mean(dim=2),
            dim=2,
        ),
        torch.nn.functional.normalize(
            reference_1024_tensor.mean(dim=2),
            dim=2,
        ),
    )


def _standardized_model_scores(
    query_games: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    query_centroids = torch.nn.functional.normalize(
        query_games.mean(dim=1),
        dim=1,
    )
    model_scores = torch.einsum("md,mpd->mp", query_centroids, centroids)
    model_scores = (
        model_scores - model_scores.mean(dim=1, keepdim=True)
    ) / model_scores.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-8)
    scores = model_scores.mean(dim=0)
    return (scores - scores.mean()) / scores.std(
        unbiased=False,
    ).clamp_min(1e-8)


def _g2g_features(
    query_games: torch.Tensor,
    references: torch.Tensor,
) -> np.ndarray:
    similarities = torch.einsum(
        "mnd,mkrd->knr",
        query_games,
        references,
    ) / query_games.shape[0]
    sorted_rows = similarities.sort(dim=2).values
    max_values = sorted_rows[:, :, -1]
    top3 = sorted_rows[:, :, -min(3, sorted_rows.shape[2]):].mean(dim=2)
    top5 = sorted_rows[:, :, -min(5, sorted_rows.shape[2]):].mean(dim=2)
    top10 = sorted_rows[:, :, -min(10, sorted_rows.shape[2]):].mean(dim=2)
    quantiles = torch.quantile(
        similarities.flatten(start_dim=1),
        torch.tensor(
            (0.1, 0.25, 0.5, 0.75, 0.9),
            device=similarities.device,
        ),
        dim=1,
    ).T
    return torch.column_stack((
        max_values.mean(dim=1),
        max_values.std(dim=1, unbiased=False),
        max_values.min(dim=1).values,
        max_values.median(dim=1).values,
        top3.mean(dim=1),
        top3.min(dim=1).values,
        top5.mean(dim=1),
        top5.min(dim=1).values,
        top10.mean(dim=1),
        similarities.mean(dim=(1, 2)),
        similarities.std(dim=(1, 2), unbiased=False),
        quantiles,
    )).cpu().numpy()


def _query_games(tensor: torch.Tensor, task: Task) -> torch.Tensor:
    indices = torch.from_numpy(task.query_indices).to(tensor.device)
    return tensor[:, task.owner].index_select(1, indices)


def _task_rows(
    tensors: PairBuildTensors,
    task: Task,
    top_k: int,
    second_weight: float,
) -> TaskRows:
    query_512 = _query_games(tensors.queries_512, task)
    query_1024 = _query_games(tensors.queries_1024, task)
    scores_512 = _standardized_model_scores(query_512, tensors.centroids_512)
    scores_1024 = _standardized_model_scores(query_1024, tensors.centroids_1024)
    ensemble_tensor = (
        (1.0 - second_weight) * scores_512
        + second_weight * scores_1024
    )
    ensemble = ensemble_tensor.cpu().numpy()
    order = np.argsort(-ensemble, kind="stable")
    candidates = order[:top_k]
    local_features = _g2g_features(
        query_1024,
        tensors.references_1024.index_select(
            1,
            torch.from_numpy(candidates.copy()).to(query_1024.device),
        ),
    )
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    stage_features = np.column_stack((
        scores_512.cpu().numpy()[candidates],
        scores_1024.cpu().numpy()[candidates],
        ensemble[candidates],
        ranks[candidates],
        ensemble[candidates] - ensemble[order[0]],
        ensemble[candidates] - ensemble[order[1]],
        ensemble[order[0]] - ensemble[candidates],
        np.full(top_k, task.game_count),
    )).astype(np.float32)
    return TaskRows(
        np.column_stack((stage_features, local_features)),
        (candidates == task.owner).astype(np.int64),
        np.full(top_k, task.owner, dtype=np.int64),
        np.full(top_k, task.game_count, dtype=np.int64),
        candidates,
        order,
        ensemble,
    )


def retrieval_rows(
    tensors: PairBuildTensors,
    task: Task,
    second_weight: float,
) -> TaskRows:
    """Builds unlabeled Top-20 features; task.owner selects a query row only."""
    rows = _task_rows(tensors, task, 20, second_weight)
    return dataclasses.replace(
        rows,
        labels=np.full(20, -1, dtype=np.int64),
        owners=np.full(20, -1, dtype=np.int64),
    )


def build_pair_dataset(
    data: EmbeddingData,
    tasks: list[Task],
    config: PairBuildConfig,
) -> PairDataset:
    """Builds Stage-1 and local-match rows for hard candidate groups."""
    tensors = _prepare_pair_tensors(data, config.independent, config.device)
    task_rows = []
    for group_id, task in enumerate(tasks):
        task_rows.append(_task_rows(
            tensors,
            task,
            config.top_k,
            config.blend_weights[task.game_count],
        ))
        if (group_id + 1) % 200 == 0:
            print(f"features {group_id + 1}/{len(tasks)}", flush=True)
    groups = [
        np.full(config.top_k, group_id, dtype=np.int64)
        for group_id in range(len(task_rows))
    ]
    rows = PairRows(
        np.concatenate([item.features for item in task_rows]),
        np.concatenate([item.labels for item in task_rows]),
        np.concatenate(groups),
        np.concatenate([item.owners for item in task_rows]),
        np.concatenate([item.counts for item in task_rows]),
        np.concatenate([item.candidate_ids for item in task_rows]),
    )
    return PairDataset(
        rows,
        [item.order for item in task_rows],
        [item.stage1_scores for item in task_rows],
    )


def _competition_metrics(ranks: np.ndarray) -> dict[str, float]:
    score = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top5_exponential_score": float(np.mean(score)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def _stage1_ranks(dataset: PairDataset) -> np.ndarray:
    return np.asarray([
        int(np.flatnonzero(order == owner)[0]) + 1
        for order, owner in zip(dataset.stage1_orders, _task_owners(dataset))
    ])


def _task_owners(dataset: PairDataset) -> np.ndarray:
    _, first_indices = np.unique(dataset.group_ids, return_index=True)
    return dataset.owners[first_indices]


def _reranked_ranks(
    model: sklearn.ensemble.HistGradientBoostingClassifier,
    dataset: PairDataset,
    reranker_weights: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(dataset.features)[:, 1]
    stage1_ranks = _stage1_ranks(dataset)
    reranked = stage1_ranks.copy()
    for group_id, owner in enumerate(_task_owners(dataset)):
        row_indices = np.flatnonzero(dataset.group_ids == group_id)
        candidates = dataset.candidate_ids[row_indices]
        if owner not in candidates:
            continue
        game_count = int(dataset.counts[row_indices[0]])
        reranker_scores = _standardize_scores(probabilities[row_indices])
        stage1_scores = _standardize_scores(
            dataset.stage1_scores[group_id][candidates],
        )
        blended = (
            reranker_weights[game_count] * reranker_scores
            + (1.0 - reranker_weights[game_count]) * stage1_scores
        )
        ranking = np.argsort(-blended, kind="stable")
        reranked[group_id] = int(np.flatnonzero(
            candidates[ranking] == owner,
        )[0]) + 1
    return stage1_ranks, reranked


def _metrics_by_count(
    ranks: np.ndarray,
    dataset: PairDataset,
) -> dict[str, dict[str, float]]:
    _, first_indices = np.unique(dataset.group_ids, return_index=True)
    task_counts = dataset.counts[first_indices]
    return {
        str(game_count): _competition_metrics(ranks[task_counts == game_count])
        for game_count in QUERY_COUNTS
    }


def evaluate_model(
    model: sklearn.ensemble.HistGradientBoostingClassifier,
    dataset: PairDataset,
    reranker_weights: dict[int, float],
) -> dict[str, object]:
    """Evaluates reranked Top-K and Stage-1 recall on grouped rows."""
    stage1_ranks, reranked = _reranked_ranks(
        model,
        dataset,
        reranker_weights,
    )
    metrics_by_count = _metrics_by_count(reranked, dataset)
    stage1_by_count = _metrics_by_count(stage1_ranks, dataset)
    recall = {
        str(count): float(np.mean(stage1_ranks <= count))
        for count in RECALL_COUNTS
    }
    return {
        "reranker": metrics_by_count,
        "stage1": stage1_by_count,
        "stage1_recall": recall,
        "reranker_proxy": float(np.mean([
            metrics_by_count[str(count)]["top5_exponential_score"]
            for count in QUERY_COUNTS
        ])),
        "stage1_proxy": float(np.mean([
            stage1_by_count[str(count)]["top5_exponential_score"]
            for count in QUERY_COUNTS
        ])),
        "reranker_weights": {
            str(count): reranker_weights[count]
            for count in QUERY_COUNTS
        },
    }


def select_reranker_weights_by_count(
    model: sklearn.ensemble.HistGradientBoostingClassifier,
    dataset: PairDataset,
) -> tuple[dict[int, float], dict[str, object], dict[str, object]]:
    """Selects one candidate-only blend weight for each query size."""
    trial_results = {}
    trial_scores: dict[str, object] = {}
    for weight in np.linspace(0.0, 1.0, 11):
        weights = {count: float(weight) for count in QUERY_COUNTS}
        results = evaluate_model(model, dataset, weights)
        trial_results[float(weight)] = results
        metrics = typing.cast(
            dict[str, dict[str, float]],
            results["reranker"],
        )
        trial_scores[f"{weight:.1f}"] = {
            str(count): metrics[str(count)]["top5_exponential_score"]
            for count in QUERY_COUNTS
        }
    best_weights = {}
    for count in QUERY_COUNTS:
        best_weight = 0.0
        best_score = -np.inf
        for weight, results in trial_results.items():
            metrics = typing.cast(
                dict[str, dict[str, float]],
                results["reranker"],
            )
            score = metrics[str(count)]["top5_exponential_score"]
            if score > best_score:
                best_weight = weight
                best_score = score
        best_weights[count] = best_weight
    return (
        best_weights,
        evaluate_model(model, dataset, best_weights),
        trial_scores,
    )


def select_global_reranker_weight(
    model: sklearn.ensemble.HistGradientBoostingClassifier,
    dataset: PairDataset,
) -> tuple[dict[int, float], dict[str, object], dict[str, object]]:
    """Selects one shared blend weight using candidate-only validation."""
    best_weights = {count: 0.0 for count in QUERY_COUNTS}
    best_results = evaluate_model(model, dataset, best_weights)
    best_score = typing.cast(float, best_results["reranker_proxy"])
    scores: dict[str, object] = {"0.0": best_score}
    for weight in np.linspace(0.1, 1.0, 10):
        weights = {count: float(weight) for count in QUERY_COUNTS}
        results = evaluate_model(model, dataset, weights)
        score = typing.cast(float, results["reranker_proxy"])
        scores[f"{weight:.1f}"] = score
        if score > best_score:
            best_weights = weights
            best_results = results
            best_score = score
    return best_weights, best_results, scores


def _load_blend_weights(path: pathlib.Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    raw_weights = payload["selected_second_dimension_weights"]
    return {int(key): float(value) for key, value in raw_weights.items()}


def _independent_tasks(player_count: int) -> list[Task]:
    tasks = []
    for owner in range(player_count):
        for game_count in QUERY_COUNTS:
            tasks.append(Task(owner, game_count, np.arange(game_count)))
    return tasks


@dataclasses.dataclass(frozen=True)
class DevelopmentData:
    """Stores player-disjoint development partitions and pair rows."""

    train_owners: np.ndarray
    validation_owners: np.ndarray
    train_pairs: PairDataset
    validation_pairs: PairDataset


@dataclasses.dataclass(frozen=True)
class ExperimentOutcome:
    """Stores selected settings and final evaluation outputs."""

    development: DevelopmentData
    selected_weights: dict[int, float]
    validation_weight_scores: dict[str, object]
    validation_results: dict[str, object]
    independent_results: dict[str, object]


def _make_pair_config(
    arguments: argparse.Namespace,
    blend_weights: dict[int, float],
    independent: bool,
) -> PairBuildConfig:
    return PairBuildConfig(
        arguments.top_k,
        blend_weights,
        independent,
        torch.device(arguments.device),
    )


def _development_data(
    arguments: argparse.Namespace,
    data: EmbeddingData,
    blend_weights: dict[int, float],
    generator: np.random.Generator,
) -> DevelopmentData:
    player_order = generator.permutation(len(data.players))
    validation_count = round(len(player_order) * arguments.validation_fraction)
    validation_owners = np.sort(player_order[:validation_count])
    train_owners = np.sort(player_order[validation_count:])
    train_tasks = make_tasks(
        train_owners,
        arguments.samples_per_count,
        data.development_holdout_positions,
        generator,
    )
    validation_tasks = make_tasks(
        validation_owners,
        arguments.samples_per_count,
        data.development_holdout_positions,
        generator,
    )
    train_pairs = build_pair_dataset(
        data,
        train_tasks,
        _make_pair_config(arguments, blend_weights, False),
    )
    validation_pairs = build_pair_dataset(
        data,
        validation_tasks,
        _make_pair_config(arguments, blend_weights, False),
    )
    return DevelopmentData(
        train_owners,
        validation_owners,
        train_pairs,
        validation_pairs,
    )


def _fit_model(
    train_pairs: PairDataset,
    seed: int,
) -> sklearn.ensemble.HistGradientBoostingClassifier:
    model = sklearn.ensemble.HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=50,
        max_leaf_nodes=7,
        max_bins=63,
        l2_regularization=1.0,
        random_state=seed,
        class_weight="balanced",
        early_stopping=True,
    )
    trainable_groups = np.unique(
        train_pairs.group_ids[train_pairs.labels == 1],
    )
    train_mask = np.isin(train_pairs.group_ids, trainable_groups)
    model.fit(
        train_pairs.features[train_mask],
        train_pairs.labels[train_mask],
    )
    return model


def _run_experiment(
    arguments: argparse.Namespace,
    data: EmbeddingData,
    blend_weights: dict[int, float],
) -> ExperimentOutcome:
    development = _development_data(
        arguments,
        data,
        blend_weights,
        np.random.default_rng(arguments.seed),
    )
    model = _fit_model(development.train_pairs, arguments.seed)
    if arguments.weight_mode == "global":
        selection = select_global_reranker_weight(
            model,
            development.validation_pairs,
        )
    else:
        selection = select_reranker_weights_by_count(
            model,
            development.validation_pairs,
        )
    selected_weights, validation_results, validation_weight_scores = selection
    independent_pairs = build_pair_dataset(
        data,
        _independent_tasks(len(data.players)),
        _make_pair_config(arguments, blend_weights, True),
    )
    independent_results = evaluate_model(
        model,
        independent_pairs,
        selected_weights,
    )
    return ExperimentOutcome(
        development,
        selected_weights,
        validation_weight_scores,
        validation_results,
        independent_results,
    )


def _result_payload(
    arguments: argparse.Namespace,
    outcome: ExperimentOutcome,
) -> dict[str, object]:
    return {
        "method": "candidate_only_hard_negative_hist_gradient_boosting",
        "uses_independent_query_labels_for_training_or_selection": False,
        "seed": arguments.seed,
        "top_k": arguments.top_k,
        "samples_per_count": arguments.samples_per_count,
        "weight_mode": arguments.weight_mode,
        "train_player_count": int(len(outcome.development.train_owners)),
        "validation_player_count": int(
            len(outcome.development.validation_owners),
        ),
        "train_query_owner_indices": outcome.development.train_owners.tolist(),
        "validation_query_owner_indices": (
            outcome.development.validation_owners.tolist()
        ),
        "validation_players_may_appear_as_unlabeled_hard_negative_candidates": True,
        "candidate_identity_is_not_a_feature": True,
        "feature_names": FEATURE_NAMES,
        "selected_reranker_weights": {
            str(count): outcome.selected_weights[count]
            for count in QUERY_COUNTS
        },
        "validation_weight_scores": outcome.validation_weight_scores,
        "validation": outcome.validation_results,
        "independent": outcome.independent_results,
        "feature_importances": {},
    }


def main() -> None:
    """Runs player-disjoint development and one frozen independent evaluation."""
    arguments = parse_arguments()
    if arguments.top_k < 5 or arguments.top_k > 50:
        raise ValueError("top-k must be between 5 and 50")
    data = load_embeddings(arguments.embeddings)
    outcome = _run_experiment(
        arguments,
        data,
        _load_blend_weights(arguments.selection),
    )
    results = _result_payload(arguments, outcome)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
