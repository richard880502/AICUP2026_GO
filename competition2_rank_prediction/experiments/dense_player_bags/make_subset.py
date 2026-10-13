"""Builds extra dense-feature shards for multi-game player training.

The existing dense training set already contains two games from nearly every
training player. This script chooses a fixed player subset and writes only the
additional games needed to reach 20 games per selected player. Existing
validation and holdout players are always excluded.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.make_subset
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

import rank_lib


def load_existing_training() -> pd.DataFrame:
    """Loads game/player pairs from the existing two-game dense subset."""
    paths = sorted(
        (rank_lib.ROOT / "features" / "shards" / "train").glob(
            "games_*.csv",
        ),
    )
    if not paths:
        raise FileNotFoundError("Existing dense training shards were not found")
    return pd.concat(
        [
            pd.read_csv(path, usecols=["game_key", "player_id", "rank"])
            for path in paths
        ],
        ignore_index=True,
    )


def validation_players() -> set[str]:
    """Returns every player in the fixed external validation cohort."""
    players: set[str] = set()
    directory = rank_lib.ROOT / "features" / "shards" / "val"
    for path in sorted(directory.glob("games_*.csv")):
        frame = pd.read_csv(path, usecols=["player_id"])
        players.update(frame["player_id"].astype(str))
    return players


def select_rank_games(
    rank: str,
    existing: pd.DataFrame,
    excluded: set[str],
    players_per_rank: int,
    games_per_player: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Selects players and only their not-yet-extracted target games."""
    path = rank_lib.ROOT / "data" / "training" / f"train_{rank}.csv"
    frame = pd.read_csv(path)
    frame["game_key"] = rank + "_" + frame["game_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str)
    counts = frame.groupby("player_id").size()
    eligible = np.asarray([
        player
        for player in counts[counts >= games_per_player].index.astype(str)
        if player not in excluded
    ])
    if len(eligible) < players_per_rank:
        raise ValueError(
            f"Rank {rank!r} has only {len(eligible)} eligible players",
        )
    selected = sorted(
        rng.choice(eligible, size=players_per_rank, replace=False).tolist(),
    )
    existing_keys = set(
        existing.loc[
            existing["player_id"].astype(str).isin(selected),
            "game_key",
        ].astype(str),
    )
    target_parts = []
    final_counts = {}
    for player in selected:
        player_games = frame[frame["player_id"] == player]
        known = player_games[player_games["game_key"].isin(existing_keys)]
        needed = games_per_player - len(known)
        available = player_games[~player_games["game_key"].isin(existing_keys)]
        random_state = int(rng.integers(0, 2**31 - 1))
        added = available.sample(n=needed, random_state=random_state)
        target_parts.append(added)
        final_counts[player] = len(known) + len(added)
    result = pd.concat(target_parts, ignore_index=True)
    columns = ["game_key", "player_id", "rank", "color", "sgf_content"]
    return result[columns], selected, final_counts


def write_shards(
    frame: pd.DataFrame,
    directory: pathlib.Path,
    shard_size: int,
    seed: int,
) -> int:
    """Writes shuffled, rank-mixed CSV shards and returns their count."""
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("games_*.csv")):
        raise FileExistsError(
            f"Refusing to overwrite existing shards in {directory}",
        )
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    for start in range(0, len(shuffled), shard_size):
        shard = start // shard_size
        shuffled.iloc[start:start + shard_size].to_csv(
            directory / f"games_{shard:04d}.csv",
            index=False,
        )
    return (len(shuffled) + shard_size - 1) // shard_size


def main() -> None:
    """Creates the targeted player-bag extraction cohort."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-rank", type=int, default=200)
    parser.add_argument("--games-per-player", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            rank_lib.ROOT
            / "experiments"
            / "dense_player_bags"
            / "shards"
        ),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    existing = load_existing_training()
    excluded = validation_players()
    holdout_path = (
        rank_lib.ROOT / "features" / "holdout" / "players.txt"
    )
    excluded.update(holdout_path.read_text(encoding="utf-8").split())

    frames = []
    selected_players = []
    counts: dict[str, int] = {}
    for rank in rank_lib.LEVELS:
        frame, selected, rank_counts = select_rank_games(
            rank,
            existing,
            excluded,
            args.players_per_rank,
            args.games_per_player,
            rng,
        )
        frames.append(frame)
        selected_players.extend(selected)
        counts.update(rank_counts)
        print(
            f"{rank}: players={len(selected)} additional_games={len(frame)}",
            flush=True,
        )
    additional = pd.concat(frames, ignore_index=True)
    shard_count = write_shards(
        additional,
        args.output,
        args.shard_size,
        args.seed,
    )
    if set(selected_players) & excluded:
        raise RuntimeError("Selected players overlap validation or holdout")
    if set(counts.values()) != {args.games_per_player}:
        raise RuntimeError("Not every selected player reaches the target count")
    manifest = {
        "players": len(selected_players),
        "players_per_rank": args.players_per_rank,
        "target_games_per_player": args.games_per_player,
        "existing_games": len(selected_players) * args.games_per_player
        - len(additional),
        "additional_games": len(additional),
        "shards": shard_count,
        "seed": args.seed,
        "validation_or_holdout_overlap": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "players.txt").write_text(
        "\n".join(selected_players) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
