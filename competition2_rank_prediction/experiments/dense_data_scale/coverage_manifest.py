"""Builds nested two/eight-game manifests for a fixed 500-player pilot.

Run from the repository root with python -m followed by this module path.
Only original training players are eligible; validation players are excluded.
"""

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def cached_players(directory: pathlib.Path) -> set[str]:
    """Reads anonymous player IDs from existing feature shards."""
    players: set[str] = set()
    for path in sorted(directory.glob("feat_*.npz")):
        with np.load(path) as data:
            players.update(np.asarray(data["player_id"], dtype=str).tolist())
    if not players:
        raise ValueError(f"No cached players found: {directory}")
    return players


def stable_order(value: str) -> str:
    """Orders identifiers independently of source row order."""
    return hashlib.sha256(f"coverage20261109:{value}".encode()).hexdigest()


def select_rank(path: pathlib.Path, train: set[str]) -> pd.DataFrame:
    """Selects eight deterministic games for fifty eligible players."""
    frame = pd.read_csv(path, dtype=str)
    frame = frame[frame["player_id"].isin(train)].copy()
    frame["game_key"] = frame["rank"] + "_" + frame["game_id"]
    frame = frame.drop_duplicates("game_key")
    counts = frame.groupby("player_id").size()
    eligible = sorted(counts[counts >= 8].index, key=stable_order)
    if len(eligible) < 50:
        raise ValueError(f"Insufficient eligible players: {path}")
    frame = frame[frame["player_id"].isin(eligible[:50])].copy()
    frame["selection_order"] = frame["game_key"].map(stable_order)
    frame = frame.sort_values(["player_id", "selection_order"])
    frame = frame.groupby("player_id", sort=False).head(8).copy()
    frame["coverage_index"] = frame.groupby("player_id").cumcount()
    return frame


def write_shards(output: pathlib.Path, expanded: pd.DataFrame) -> None:
    """Writes nested source shards into a newly created output directory."""
    columns = ["game_key", "player_id", "rank", "color", "sgf_content"]
    output.mkdir(parents=True)
    for cap in (2, 8):
        selected = expanded[expanded["coverage_index"] < cap][columns]
        directory = output / f"cap{cap}"
        directory.mkdir()
        for start in range(0, len(selected), 500):
            selected.iloc[start : start + 500].to_csv(
                directory / f"games_{start // 500:04d}.csv",
                index=False,
            )


def main() -> None:
    """Writes isolated pilot shards, or refuses an existing output directory."""
    root = pathlib.Path.cwd()
    research = root / "runtime/competition2_research"
    output = root / "production/game_coverage_manifest_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing manifest: {output}")
    train = cached_players(research / "features/feat96/train")
    validation = cached_players(research / "features/feat96/val")
    if train & validation:
        raise ValueError("Training and validation players overlap")
    frames = []
    source_hashes = {}
    levels = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
    for rank in levels:
        path = research / "data/training" / f"train_{rank}.csv"
        source_hashes[rank] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(select_rank(path, train))
        print(f"Selected {rank}: 50 players, 400 games", flush=True)
    expanded = pd.concat(frames, ignore_index=True)
    if expanded["game_key"].duplicated().any():
        raise ValueError("Duplicate game keys in pilot")
    if set(expanded["player_id"]) & validation:
        raise ValueError("Pilot includes validation players")
    write_shards(output, expanded)
    report = {
        "players": 500,
        "cap2_games": 1000,
        "cap8_games": 4000,
        "validation_overlap": 0,
        "nested_selection": True,
        "source_sha256": source_hashes,
        "script_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
