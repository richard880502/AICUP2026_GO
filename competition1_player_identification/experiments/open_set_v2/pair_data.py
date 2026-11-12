"""Creates training-only synthetic hard-candidate tasks for frozen reranking.

Only clean training players supply training rows. The fixed style-validation
cohort is used for model selection and never for fitting pair normalization.
"""

import dataclasses
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)


@dataclasses.dataclass
class PairDataset:
    """Stores grouped Top-20 pairs, retaining missed positives explicitly.

    Attributes:
        features: Query groups by twenty candidates by pair features.
        targets: Correct candidate's slot, or -1 for a stage-one miss.
        top_ids: Candidate positions inside each 600-player reference pool.
        scores: Unmodified stage-one scores against all 600 candidates.
        counts: Number of query games per group.
        truth_positions: Full reference-pool answer slots or -1 when unknown.
    """

    features: np.ndarray
    targets: np.ndarray
    top_ids: np.ndarray
    scores: np.ndarray
    counts: np.ndarray
    truth_positions: np.ndarray


def row_positions(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Creates ten-reference / five-query training bags without reused games."""
    selected = np.load(root / "engine_inputs/global_indices.npy")
    raw = dataset.load_arrays(root / "raw_cache", ("players",))
    with np.load(root / "clean_splits.npz") as splits:
        train = np.flatnonzero(np.isin(selected, splits["train"]))
        validation = np.searchsorted(selected, splits["validation"])
    _, groups = dataset.player_groups(raw["players"][selected[train]])
    generator = np.random.default_rng(20261112)
    positions = []
    for group in groups:
        if len(group) >= 15:
            positions.append(train[generator.permutation(group)[:15]])
    if len(positions) < 600:
        raise ValueError(
            "Need at least 600 training players with fifteen games"
        )
    return np.stack(positions), validation


def stage_matrix(
    bank: pair_features.CandidateBank, weight: float
) -> np.ndarray:
    """Computes centroid ensemble retrieval for the aligned validation pool."""
    matrices = []
    for count in evaluation.COUNTS:
        first = pair_features.normalize(bank.queries_a[:, :count].mean(1))
        second = pair_features.normalize(bank.queries_b[:, :count].mean(1))
        matrices.append(
            (1 - weight) * (first @ bank.centroids_a.T)
            + weight * (second @ bank.centroids_b.T)
        )
    return np.stack(matrices)


def choose_weight(bank: pair_features.CandidateBank) -> tuple[float, list]:
    """Selects dimension blending on style validation with a fixed small grid."""
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        ranks = evaluation.correct_ranks(stage_matrix(bank, weight))
        score = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0).mean()
        trials.append({"weight": weight, "score": float(score)})
    best = max(
        trials, key=lambda row: (row["score"], -abs(row["weight"] - 0.5))
    )
    return best["weight"], trials


def assemble(
    rows: list[pair_features.PairRows], counts: list[int]
) -> PairDataset:
    """Stacks complete query groups in their original deterministic order."""
    return PairDataset(
        np.stack([row.features for row in rows]),
        np.asarray([row.label for row in rows], dtype=np.int64),
        np.stack([row.top_ids for row in rows]),
        np.stack([row.scores for row in rows]),
        np.asarray(counts, dtype=np.int64),
        np.asarray([row.truth_position for row in rows], dtype=np.int64),
    )


def synthetic_pairs(
    bank: pair_features.CandidateBank,
    weight: float,
    owners: int,
) -> PairDataset:
    """Mines natural Top-20 candidates from a fresh 600-way pool per query."""
    generator = np.random.default_rng(20261112)
    queries = generator.choice(len(bank.centroids_a), owners, replace=False)
    rows = []
    counts = []
    for owner in queries:
        for count in evaluation.COUNTS:
            candidates = generator.choice(
                len(bank.centroids_a), 600, replace=False
            )
            if owner not in candidates:
                candidates[0] = owner
            task = pair_features.Task(
                int(owner), count, candidates, truth_owner=int(owner)
            )
            rows.append(pair_features.make_rows((bank, task, weight, "both")))
            counts.append(count)
    return assemble(rows, counts)


def validation_pairs(
    bank: pair_features.CandidateBank, weight: float
) -> PairDataset:
    """Keeps all validation questions, including misses outside Top-20."""
    if len(bank.centroids_a) != 600:
        raise ValueError("The v2 validation pool must contain 600 players")
    rows = []
    counts = []
    for count in evaluation.COUNTS:
        for owner in range(600):
            task = pair_features.Task(
                owner, count, np.arange(600), truth_owner=owner
            )
            rows.append(pair_features.make_rows((bank, task, weight, "both")))
            counts.append(count)
    return assemble(rows, counts)


def stage_recall(pairs: PairDataset) -> dict:
    """Reports Top-K recall, accounting for every natural stage-one miss."""
    if (pairs.truth_positions < 0).any():
        raise ValueError("Recall reporting requires explicit answer labels")
    order = np.argsort(-pairs.scores, axis=1, kind="stable")
    ranks = np.argmax(order == pairs.truth_positions[:, None], axis=1) + 1
    return {
        str(limit): {
            str(count): float((ranks[pairs.counts == count] <= limit).mean())
            for count in evaluation.COUNTS
        }
        for limit in (5, 10, 20, 30, 50)
    }


def feature_columns(branch: str) -> np.ndarray:
    """Selects one or both game-matching families, retaining stage features."""
    if branch == "a":
        return np.arange(20)
    if branch == "b":
        return np.r_[np.arange(7), np.arange(20, 33)]
    if branch == "both":
        return np.arange(33)
    raise ValueError(f"Unknown game-matching branch: {branch!r}")
