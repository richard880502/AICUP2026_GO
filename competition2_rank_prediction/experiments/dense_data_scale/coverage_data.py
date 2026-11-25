"""Validates coverage-pilot keys and deterministic player-disjoint partitions.

Use validate_features before training; partial extraction is never accepted.
"""

import pathlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_manifest,
)


def partitions(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Reserves ten players per rank using the preregistered stable order."""
    result: dict[str, set[str]] = {"optimizer": set(), "inner": set()}
    if frame.groupby("player_id")["rank"].nunique().max() != 1:
        raise ValueError("A player has conflicting rank labels")
    for _, group in frame.groupby("rank"):
        players = sorted(
            group["player_id"].unique(), key=coverage_manifest.stable_order
        )
        if len(players) != 50:
            raise ValueError("Expected fifty players per rank")
        result["inner"].update(players[:10])
        result["optimizer"].update(players[10:])
    if len(result["inner"]) != 100 or len(result["optimizer"]) != 400:
        raise ValueError("Expected ten ranks and 500 total players")
    if result["inner"] & result["optimizer"]:
        raise ValueError("Inner and optimizer players overlap")
    return result


def read_manifest(directory: pathlib.Path) -> pd.DataFrame:
    """Loads a complete eight-game manifest without changing its row order."""
    paths = sorted(directory.glob("games_*.csv"))
    if len(paths) != 8:
        raise ValueError("Expected eight source shards")
    frame = pd.concat(
        [pd.read_csv(path, dtype=str) for path in paths], ignore_index=True
    )
    if len(frame) != 4000 or frame["game_key"].duplicated().any():
        raise ValueError("Expected 4000 unique source game keys")
    if not frame.groupby("player_id").size().eq(8).all():
        raise ValueError("Expected eight games per player")
    return frame


def validate_features(directory: pathlib.Path, manifest: pd.DataFrame) -> None:
    """Rejects incomplete, misaligned or nonfinite cached decision features."""
    seen: set[str] = set()
    expected = manifest.set_index("game_key")
    for path in sorted(directory.glob("feat_*.npz")):
        with np.load(path) as payload:
            if np.asarray(payload["profiles"]).tolist() != [
                "normal",
                "rank_5k",
                "rank_1d",
                "rank_5d",
            ]:
                raise ValueError(f"Incorrect policy profile order: {path}")
            keys = np.asarray(payload["game_key"], dtype=str)
            tokens = np.asarray(payload["tokens"])
            masks = np.asarray(payload["masks"])
            if tokens.shape != (len(keys), 96, 4, 3):
                raise ValueError(f"Wrong token schema: {path}")
            if masks.shape != (len(keys), 96) or not masks.any(axis=1).all():
                raise ValueError(f"Missing valid decisions: {path}")
            if not np.isfinite(tokens).all():
                raise ValueError(f"Nonfinite decision feature: {path}")
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError(f"Duplicate cached game keys: {path}")
            if not set(keys).issubset(expected.index):
                raise ValueError(f"Unexpected cached game keys: {path}")
            for name in ("player_id", "rank"):
                if not np.array_equal(
                    np.asarray(payload[name], dtype=str),
                    expected.loc[keys, name].to_numpy(),
                ):
                    raise ValueError(f"Misaligned {name}: {path}")
            seen.update(keys)
    if seen != set(expected.index):
        raise ValueError(f"Incomplete extraction: {len(seen)}/4000 games")
