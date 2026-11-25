"""Audits dense player bags and restricts normalization to optimizer players.

Example: data.load_dense(research_root) returns audited train/inner/eval data.
"""

import pathlib

import numpy as np

from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)


def load_dense(root: pathlib.Path) -> player_sampler.PlayerData:
    """Reuses fixed partitions but excludes sparse players from optimization."""
    data = player_sampler.load_data(root)
    dense = np.asarray(
        [
            player
            for player in data.splits["train"]
            if len(data.rows[str(player)]) >= 20
        ]
    )
    rows = np.concatenate([data.rows[str(player)] for player in dense])
    selected = data.tokens[rows][data.masks[rows]].astype(np.float64)
    mean, std = selected.mean(axis=0), selected.std(axis=0) + 1e-6
    data.tokens = np.where(
        data.masks[..., None], (data.tokens - mean) / std, 0
    ).astype(np.float32)
    previous = data.provenance.normalization
    data.provenance.normalization = {
        "mean": previous["mean"] + previous["std"] * mean,
        "std": previous["std"] * std,
    }
    data.splits["train"] = dense
    data.provenance.audit["source_split_player_counts"] = data.provenance.audit[
        "split_player_counts"
    ]
    data.provenance.audit["split_player_counts"] = {
        name: len(values) for name, values in data.splits.items()
    }
    data.provenance.audit["split_rank_counts"] = {
        name: np.bincount(
            [data.labels[data.rows[str(player)][0]] for player in values],
            minlength=10,
        ).tolist()
        for name, values in data.splits.items()
    }
    data.provenance.audit["optimizer_dense_players"] = len(dense)
    data.provenance.audit["optimizer_games"] = len(rows)
    data.provenance.audit["normalization_dense_optimizer_only"] = True
    data.provenance.audit["sparse_players_optimized"] = False
    for name in ("inner", "evaluation"):
        if set(dense) & set(data.splits[name]):
            raise ValueError(f"Dense optimizer overlaps {name} players")
    return data


def sample_bags(
    data: player_sampler.PlayerData, generator: np.random.Generator
) -> list[np.ndarray]:
    """Draws eight distinct optimizer players and genuine variable-size queries."""
    selected = generator.choice(data.splits["train"], 8, replace=False)
    bags = []
    for player in selected:
        size = int(generator.choice((5, 10, 15, 20)))
        bags.append(
            generator.choice(data.rows[str(player)], size, replace=False)
        )
    return bags
