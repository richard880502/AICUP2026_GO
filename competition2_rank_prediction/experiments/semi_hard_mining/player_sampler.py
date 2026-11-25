"""Builds player-disjoint decision data and differentiable bag indices.

Used by the companion training module; cached inputs remain read-only.
"""

import dataclasses
import pathlib

import numpy as np

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error

SEED = 20261108


@dataclasses.dataclass
class Provenance:
    """Stores training normalization and the data inventory.

    Attributes:
        normalization: Training-only mean and standard deviation.
        audit: Player counts and exclusion information.
    """

    normalization: dict[str, np.ndarray]
    audit: dict[str, object]


@dataclasses.dataclass
class PlayerData:
    """Contains standardized games and complete player partitions.

    Attributes:
        tokens: Standardized decision tokens for each game.
        masks: Valid token masks.
        labels: Per-game integer rank labels.
        rows: Player ID to all game positions.
        splits: Disjoint player arrays for train/inner/evaluation.
        questions: Fixed query indices for validation and evaluation.
        provenance: Training normalization and audit information.
    """

    tokens: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    rows: dict[str, np.ndarray]
    splits: dict[str, np.ndarray]
    questions: dict[str, list[np.ndarray]]
    provenance: Provenance


def _merged(root: pathlib.Path) -> dict[str, np.ndarray]:
    directories = (
        root / "features/feat96/train",
        root / "experiments/dense_player_bags/features",
    )
    parts = [rank_lib.load_features(directory) for directory in directories]
    merged = {
        key: np.concatenate((parts[0][key], parts[1][key])) for key in parts[0]
    }
    _, unique = np.unique(merged["game_key"], return_index=True)
    return {key: value[np.sort(unique)] for key, value in merged.items()}


def _partitions(
    rows: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[str]]:
    conflicting = []
    dense: dict[int, list[str]] = {rank: [] for rank in range(10)}
    for player, indices in rows.items():
        ranks = np.unique(labels[indices])
        if len(ranks) != 1:
            conflicting.append(player)
        elif len(indices) >= 20:
            dense[int(ranks[0])].append(player)
    generator = np.random.default_rng(SEED)
    inner: list[str] = []
    evaluation: list[str] = []
    for rank in range(10):
        players = generator.permutation(sorted(dense[rank]))
        count = int(0.15 * len(players))
        if count < 5:
            raise ValueError(f"Insufficient dense players for rank {rank}")
        inner.extend(players[:count].tolist())
        evaluation.extend(players[count : 2 * count].tolist())
    excluded = set(inner) | set(evaluation) | set(conflicting)
    training = sorted(set(rows) - excluded)
    return {
        "train": np.asarray(training),
        "inner": np.asarray(inner),
        "evaluation": np.asarray(evaluation),
    }, conflicting


def load_data(root: pathlib.Path) -> PlayerData:
    """Creates deterministic disjoint splits and training-only normalization."""
    features = _merged(root)
    players = features["player_id"]
    labels = rank_lib.rank_ids(features["rank"])
    rows = {
        str(player): np.flatnonzero(players == player)
        for player in np.unique(players)
    }
    splits, conflicting = _partitions(rows, labels)
    if set(splits["train"]) & (
        set(splits["inner"]) | set(splits["evaluation"])
    ):
        raise ValueError("Player split overlap")
    tokens = transformer_rank.decision_tokens(features["tokens"])
    masks = features["masks"].copy()
    training = np.isin(players, splits["train"])
    mean, scale = transformer_rank.standardise(
        tokens[training], masks[training]
    )
    tokens = np.where(masks[..., None], (tokens - mean) / scale, 0).astype(
        np.float32
    )
    masks[~masks.any(axis=1), 0] = True
    questions = _questions(rows, splits)
    audit = _audit(players, rows, labels, splits, conflicting)
    return PlayerData(
        tokens,
        masks,
        labels,
        rows,
        splits,
        questions,
        Provenance({"mean": mean, "std": scale}, audit),
    )


def _questions(
    rows: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    generator = np.random.default_rng(SEED + 1)
    questions: dict[str, list[np.ndarray]] = {}
    for name in ("inner", "evaluation"):
        questions[name] = []
        for player in splits[name]:
            count = int(generator.integers(5, 21))
            questions[name].append(
                generator.choice(rows[str(player)], count, replace=False)
            )
    return questions


def _audit(
    players: np.ndarray,
    rows: dict[str, np.ndarray],
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    conflicting: list[str],
) -> dict[str, object]:
    return {
        "games": len(players),
        "players": len(rows),
        "excluded_inconsistent_players": conflicting,
        "split_player_counts": {
            name: len(values) for name, values in splits.items()
        },
        "split_rank_counts": {
            name: np.bincount(
                [labels[rows[str(player)][0]] for player in values],
                minlength=10,
            ).tolist()
            for name, values in splits.items()
        },
        "train_dense_players": sum(
            len(rows[str(player)]) >= 20 for player in splits["train"]
        ),
        "initialization": "from scratch; no upstream checkpoints",
        "fresh_outer_cohort": False,
    }


def sample_pairs(
    data: PlayerData,
    generator: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Samples four distinct players per rank with 5–20-game bags."""
    pools = {}
    for rank in (4, 5):
        pools[rank] = [
            str(player)
            for player in data.splits["train"]
            if len(data.rows[str(player)]) >= 20
            and data.labels[data.rows[str(player)][0]] == rank
        ]
    low = generator.choice(pools[4], 4, replace=False)
    high = generator.choice(pools[5], 4, replace=False)
    pairs = []
    for low_player, high_player in zip(low, high):
        bags = []
        for player in (low_player, high_player):
            count = int(generator.integers(5, 21))
            bags.append(
                generator.choice(data.rows[str(player)], count, replace=False)
            )
        pairs.append((bags[0], bags[1]))
    return pairs
