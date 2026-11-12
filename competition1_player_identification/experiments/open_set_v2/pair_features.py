"""Builds fixed-size game-pair features for frozen Top-20 reranking.

CandidateBank uses ten references and five available query games per player.
The feature builder never fits encoders, preprocessing, or identity tables.
"""

import dataclasses

import numpy as np

TOP_K = 20
STAGE_NAMES = (
    "cosine_a",
    "cosine_b",
    "ensemble_cosine",
    "stage_rank_scaled",
    "gap_to_top1",
    "gap_to_top2",
    "query_count_scaled",
)
MATCH_NAMES = (
    "reference_mean_norm",
    "query_mean_norm",
    "all_mean",
    "all_std",
    "max_mean",
    "max_std",
    "max_min",
    "max_median",
    "top3_mean",
    "top3_min",
    "top5_mean",
    "all_q10",
    "all_q90",
)


@dataclasses.dataclass
class CandidateBank:
    """Stores frozen games and normalized centroids for both branches.

    Attributes:
        references_a: First-branch ten-game reference bags.
        references_b: Second-branch ten-game reference bags.
        centroids_a: Unit first-branch reference centroids.
        centroids_b: Unit second-branch reference centroids.
        queries_a: First-branch available query-game embeddings.
        queries_b: Second-branch available query-game embeddings.
    """

    references_a: np.ndarray
    references_b: np.ndarray
    centroids_a: np.ndarray
    centroids_b: np.ndarray
    queries_a: np.ndarray
    queries_b: np.ndarray


@dataclasses.dataclass
class Task:
    """Separates query row selection from any optional answer label.

    Attributes:
        query_index: Query-bank row, independent of its unknown identity.
        count: Number of query games used.
        candidates: Reference-player positions in the retrieval pool.
        truth_owner: Optional answer for training or metric reporting only.
    """

    query_index: int
    count: int
    candidates: np.ndarray
    truth_owner: int | None = None


@dataclasses.dataclass
class PairRows:
    """Holds pair features and untouched stage-one scores for one question.

    Attributes:
        features: Twenty candidate rows with fixed feature dimensions.
        top_ids: Candidate slots in the original reference pool.
        scores: Scores against the entire candidate pool.
        label: True Top-20 slot or -1 when missed or unknown.
        truth_position: Full-pool answer slot or -1 when unknown.
    """

    features: np.ndarray
    top_ids: np.ndarray
    scores: np.ndarray
    label: int
    truth_position: int


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalizes rows for cosine matching without fitting any parameters."""
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(
        min=1e-8
    )


def build_bank(
    first: np.ndarray,
    second: np.ndarray,
    positions: np.ndarray,
) -> CandidateBank:
    """Uses fixed, distinct ten-reference / five-query row positions."""
    if positions.ndim != 2 or positions.shape[1] != 15:
        raise ValueError("Candidate positions must have shape [players, 15]")
    if len(first) != len(second):
        raise ValueError("Frozen branches contain different game counts")
    first_games = normalize(np.asarray(first[positions], dtype=np.float32))
    second_games = normalize(np.asarray(second[positions], dtype=np.float32))
    return CandidateBank(
        first_games[:, :10],
        second_games[:, :10],
        normalize(first_games[:, :10].mean(1)),
        normalize(second_games[:, :10].mean(1)),
        first_games[:, 10:],
        second_games[:, 10:],
    )


def match_statistics(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Returns thirteen distribution summaries per candidate, never a flatten."""
    if query.ndim != 2 or query.shape[0] == 0:
        raise ValueError("Query must contain at least one game embedding")
    if references.ndim != 3 or references.shape[1] != 10:
        raise ValueError("References must contain ten games per candidate")
    similarities = np.reshape(
        query @ references.reshape(-1, references.shape[-1]).T,
        (len(query), len(references), 10),
    ).transpose(1, 0, 2)
    maxima = similarities.max(2)
    top3 = np.partition(similarities, -3, axis=2)[:, :, -3:].mean(2)
    top5 = np.partition(similarities, -5, axis=2)[:, :, -5:].mean(2)
    flattened = similarities.reshape(len(references), -1)
    quantiles = np.quantile(flattened, (0.1, 0.9), axis=1)
    return np.column_stack(
        (
            np.linalg.norm(references.mean(1), axis=1),
            np.full(len(references), np.linalg.norm(query.mean(0))),
            flattened.mean(1),
            flattened.std(1),
            maxima.mean(1),
            maxima.std(1),
            maxima.min(1),
            np.median(maxima, axis=1),
            top3.mean(1),
            top3.min(1),
            top5.mean(1),
            quantiles[0],
            quantiles[1],
        )
    ).astype(np.float32)


def stage_scores(
    bank: CandidateBank,
    task: Task,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores a few-game query against the specified frozen candidate pool."""
    first = bank.centroids_a[task.candidates] @ normalize(
        bank.queries_a[task.query_index, : task.count].mean(0),
    )
    second = bank.centroids_b[task.candidates] @ normalize(
        bank.queries_b[task.query_index, : task.count].mean(0),
    )
    return first, second, (1 - weight) * first + weight * second


def make_rows(context: tuple) -> PairRows:
    """Mines only natural stage-one hard candidates; never forces the answer."""
    bank, task, weight, branch = context
    if branch not in ("a", "b", "both"):
        raise ValueError(f"Unknown matching branch: {branch!r}")
    if not 1 <= task.count <= 5 or len(task.candidates) < TOP_K:
        raise ValueError("Need 1-5 query games and at least twenty candidates")
    first, second, scores = stage_scores(bank, task, weight)
    top = np.argsort(-scores, kind="stable")[:TOP_K]
    owners = task.candidates[top]
    stage = np.column_stack(
        (
            first[top],
            second[top],
            scores[top],
            np.arange(1, TOP_K + 1) / TOP_K,
            scores[top[0]] - scores[top],
            scores[top[1]] - scores[top],
            np.full(TOP_K, task.count / 5),
        )
    )
    families = [stage]
    if branch in ("a", "both"):
        families.append(
            match_statistics(
                bank.queries_a[task.query_index, : task.count],
                bank.references_a[owners],
            )
        )
    if branch in ("b", "both"):
        families.append(
            match_statistics(
                bank.queries_b[task.query_index, : task.count],
                bank.references_b[owners],
            )
        )
    labels = np.flatnonzero(owners == task.truth_owner)
    truth = np.flatnonzero(task.candidates == task.truth_owner)
    return PairRows(
        np.concatenate(families, axis=1).astype(np.float32),
        top,
        scores,
        int(labels[0]) if len(labels) else -1,
        int(truth[0]) if len(truth) else -1,
    )


def reorder_scores(
    stage: np.ndarray,
    candidate_ids: np.ndarray,
    logits: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Reorders only Top-20, preserving the base score profile and pool tail."""
    flattened = stage.reshape(-1, stage.shape[-1])
    baseline = np.take_along_axis(flattened, candidate_ids, axis=1)
    normalized = baseline - baseline.mean(1, keepdims=True)
    normalized /= baseline.std(1, keepdims=True).clip(min=1e-6)
    predictions = logits - logits.mean(1, keepdims=True)
    predictions /= logits.std(1, keepdims=True).clip(min=1e-6)
    order = np.argsort(
        -((1 - weight) * normalized + weight * predictions),
        axis=1,
        kind="stable",
    )
    reordered = np.take_along_axis(candidate_ids, order, axis=1)
    result = flattened.copy()
    # Positive, decreasing offsets disambiguate ties within the selected pool.
    confidence = baseline + (TOP_K - np.arange(TOP_K))[None] * 1e-6
    np.put_along_axis(result, reordered, confidence, axis=1)
    return result.reshape(stage.shape)
