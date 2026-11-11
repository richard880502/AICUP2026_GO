"""Selects candidate-only blends and reranks unlabeled official queries.

Uses the existing 24-feature Top-20 group-softmax implementation.
"""

import dataclasses
import json

import numpy as np
import torch

from competition1_player_identification.experiments.reranker import run_experiment
from competition1_player_identification.experiments.reranker import run_group_mlp
from competition1_player_identification.production_pipeline import metric


@dataclasses.dataclass
class Context:
    """Stores the pool's frozen branches and candidate-only blend weights."""

    metric_context: metric.Context
    first: metric.Branch
    second: metric.Branch
    weights: dict[int, float]


def normalize(values: np.ndarray) -> np.ndarray:
    """L2 normalizes along the final dimension."""
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True),
                               1e-8)


def row_standardize(values: np.ndarray) -> np.ndarray:
    """Matches the existing ensemble's per-row score normalization."""
    return (values - values.mean(axis=-1, keepdims=True)) / np.maximum(
        values.std(axis=-1, keepdims=True), 1e-8,
    )


def take_positions(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Selects a different game-index sequence for each candidate player."""
    return np.stack([
        values[:, owner, selected]
        for owner, selected in enumerate(positions)
    ], axis=1)


def score_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Computes the established per-seed standardized centroid ensemble."""
    similarities = np.einsum(
        "mqd,mpd->mqp",
        normalize(queries.mean(axis=2)),
        normalize(references.mean(axis=2)),
    )
    return row_standardize(row_standardize(similarities).mean(axis=0))


def utility(scores: np.ndarray) -> float:
    """Scores candidate-held-out queries whose owner is their row index."""
    orders = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(orders == np.arange(len(orders))[:, None], axis=1) + 1
    return float(np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0).mean())


def select_weights(context: Context) -> dict[int, float]:
    """Selects dimension weights using only candidate-held-out games."""
    references = [
        take_positions(branch.development,
                       context.metric_context.reference_positions)
        for branch in (context.first, context.second)
    ]
    held = [take_positions(branch.development,
                            context.metric_context.holdout_positions)
            for branch in (context.first, context.second)]
    weights = {}
    for count in context.metric_context.counts:
        scores = [score_matrix(reference, query[:, :, :count])
                  for reference, query in zip(references, held)]
        trials = [(utility((1 - weight) * scores[0] + weight * scores[1]),
                   float(weight)) for weight in np.linspace(0, 1, 21)]
        weights[count] = max(trials, key=lambda item: item[0])[1]
    return weights


def packed_queries(branch: metric.Branch, context: Context) -> np.ndarray:
    """Groups flattened official query embeddings without assuming an owner."""
    queries = context.metric_context.pool.queries
    packed = np.zeros((len(branch.queries), len(queries.counts),
                       int(queries.counts.max()), branch.queries.shape[-1]),
                      dtype=np.float32)
    for index, count in enumerate(queries.counts):
        packed[:, index, :count] = branch.queries[:, queries.groups == index]
    return packed


def make_tensors(context: Context, development: bool) -> run_experiment.PairBuildTensors:
    """Builds frozen reference and query tensors for local matching."""
    if development:
        references = [take_positions(branch.development,
                                      context.metric_context.reference_positions)
                      for branch in (context.first, context.second)]
        queries = [context.first.development, context.second.development]
    else:
        references = [context.first.final, context.second.final]
        queries = [packed_queries(context.first, context),
                   packed_queries(context.second, context)]
    reference_tensors = [
        torch.from_numpy(values).to(context.metric_context.device)
        for values in references
    ]
    query_tensors = [
        torch.from_numpy(values).to(context.metric_context.device)
        for values in queries
    ]
    return run_experiment.PairBuildTensors(
        reference_tensors[1], query_tensors[0], query_tensors[1],
        torch.nn.functional.normalize(reference_tensors[0].mean(dim=2), dim=2),
        torch.nn.functional.normalize(reference_tensors[1].mean(dim=2), dim=2),
    )


def concatenate_rows(rows: list[run_experiment.TaskRows]) -> run_experiment.PairDataset:
    """Collects independently constructed Top-20 query groups."""
    return run_experiment.PairDataset(
        run_experiment.PairRows(
            np.concatenate([item.features for item in rows]),
            np.concatenate([item.labels for item in rows]),
            np.repeat(np.arange(len(rows)), 20),
            np.concatenate([item.owners for item in rows]),
            np.concatenate([item.counts for item in rows]),
            np.concatenate([item.candidate_ids for item in rows]),
        ),
        [item.order for item in rows],
        [item.stage1_scores for item in rows],
    )


def make_dataset(context: Context, owners: np.ndarray,
                 development: bool) -> run_experiment.PairDataset:
    """Builds supervised synthetic tasks or strictly unlabeled test tasks."""
    tensors = make_tensors(context, development)
    generator = np.random.default_rng(20261004)
    rows = []
    for owner in owners:
        counts = context.metric_context.counts if development else (
            int(context.metric_context.pool.queries.counts[owner]),
        )
        for count in counts:
            for _ in range(4 if development else 1):
                indices = generator.choice(
                    context.metric_context.holdout_positions[owner],
                    size=count, replace=False,
                ) if development else np.arange(count)
                row = run_experiment.retrieval_rows(
                    tensors, run_experiment.Task(int(owner), count, indices),
                    context.weights[count],
                )
                if development:
                    row = dataclasses.replace(
                        row, labels=(row.candidate_ids == owner).astype(np.int64),
                        owners=np.full(20, owner, dtype=np.int64),
                    )
                rows.append(row)
        if len(rows) % 200 == 0:
            print(f"pair groups={len(rows)} development={development}", flush=True)
    return concatenate_rows(rows)


def fit_reranker(context: Context) -> run_group_mlp.ProductionFit:
    """Fits the frozen-encoder reranker on disjoint candidate owner players."""
    owners = np.random.default_rng(20261004).permutation(
        len(context.metric_context.pool.candidates.players),
    )
    boundary = int(0.8 * len(owners))
    fitted = run_group_mlp.fit_production_pairs(
        make_dataset(context, owners[:boundary], True),
        make_dataset(context, owners[boundary:], True),
        context.metric_context.device,
    )
    torch.save({
        "state": fitted.selection.state, "mean": fitted.mean,
        "standard_deviation": fitted.standard_deviation,
        "weight": fitted.selection.weight, "epoch": fitted.selection.epoch,
        "train_owners": owners[:boundary], "validation_owners": owners[boundary:],
    }, context.metric_context.output / "reranker.pt")
    (context.metric_context.output / "reranker_selection.json").write_text(
        json.dumps({"epoch": fitted.selection.epoch,
                    "weight": fitted.selection.weight,
                    "candidate_validation_score": fitted.selection.score,
                    "dimension_weights": context.weights,
                    "history": fitted.selection.history}, indent=2),
        encoding="utf-8",
    )
    return fitted


def predict(context: Context, fitted: run_group_mlp.ProductionFit) -> np.ndarray:
    """Returns Top-5 candidate indices for every unlabeled official question."""
    pairs = make_dataset(context, np.arange(len(
        context.metric_context.pool.queries.counts,
    )), False)
    logits = run_group_mlp.predict_production_pairs(
        fitted, pairs, context.metric_context.device,
    ).reshape(-1, 20)
    orders = []
    for index, order in enumerate(pairs.stage1_orders):
        candidates = order[:20]
        scores = fitted.selection.weight * row_standardize(logits[index])
        scores += (1 - fitted.selection.weight) * row_standardize(
            pairs.stage1_scores[index][candidates],
        )
        orders.append(candidates[np.argsort(-scores, kind="stable")][:5])
    return np.stack(orders)
