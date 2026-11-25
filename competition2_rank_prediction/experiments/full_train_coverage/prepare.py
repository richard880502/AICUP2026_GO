"""Prepares additional, player-safe dense feature extraction shards.

The script samples additional source games for all players in the existing
dense training split. It excludes previously extracted games and validation
players, then writes rank-mixed shards for the KataGo extractor.

Example:
    python -m competition2_rank_prediction.experiments.full_train_coverage.prepare
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
SHARD_SIZE = 500


@dataclasses.dataclass(frozen=True)
class SelectionState:
    """Contains player-safe training metadata for the fixed experiment."""

    known_games: dict[str, set[str]]
    players_by_rank: dict[str, set[str]]
    validation_players: int


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Contains the options required to sample one rank's games."""

    player_ids: set[str]
    known_games: dict[str, set[str]]
    additional_games: int
    random_generator: np.random.Generator


def project_root() -> pathlib.Path:
    """Returns the repository root for this experiment module."""
    return pathlib.Path(__file__).resolve().parents[3]


def read_player_shards(directory: pathlib.Path) -> pd.DataFrame:
    """Loads player, rank, and game-key metadata from a shard directory."""
    paths = sorted(directory.glob("games_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No game metadata shards found in {directory!r}")
    return pd.concat(
        [
            pd.read_csv(
                path,
                usecols=["player_id", "rank", "game_key"],
                dtype={"player_id": str, "rank": str, "game_key": str},
            )
            for path in paths
        ],
        ignore_index=True,
    )


def select_rank_games(
    rank: str,
    data_path: pathlib.Path,
    config: SamplingConfig,
) -> pd.DataFrame:
    """Samples unseen games for each selected player at one rank."""
    games = pd.read_csv(
        data_path,
        usecols=["player_id", "game_id", "rank", "color", "sgf_content"],
        dtype={"player_id": str, "game_id": str, "rank": str},
    )
    games = games.loc[games["player_id"].isin(config.player_ids)].copy()
    games["game_key"] = rank + "_" + games["game_id"]
    selected = []
    observed_players: set[str] = set()
    for player_id, player_games in games.groupby("player_id", sort=False):
        observed_players.add(player_id)
        previous = config.known_games.get(player_id, set())
        available = player_games.loc[~player_games["game_key"].isin(previous)]
        if available.empty:
            continue
        sample_count = min(config.additional_games, len(available))
        random_state = int(
            config.random_generator.integers(0, 2**31 - 1),
        )
        sampled = available.sample(
            n=sample_count,
            replace=False,
            random_state=random_state,
        )
        selected.append(sampled)
    missing_players = config.player_ids - observed_players
    if missing_players:
        raise ValueError(
            f"Rank {rank!r} has {len(missing_players)} players without source rows",
        )
    if not selected:
        raise ValueError(f"Rank {rank!r} has no additional games to sample")
    result = pd.concat(selected, ignore_index=True)
    return result[
        ["game_key", "player_id", "rank", "color", "sgf_content"]
    ]


def write_shards(
    games: pd.DataFrame,
    output: pathlib.Path,
    seed: int,
) -> int:
    """Writes shuffled extraction shards and returns their count."""
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("games_*.csv")):
        raise FileExistsError(f"Refusing to overwrite game shards in {output!r}")
    shuffled = games.sample(frac=1, random_state=seed).reset_index(drop=True)
    for offset in range(0, len(shuffled), SHARD_SIZE):
        shard_index = 1000 + offset // SHARD_SIZE
        shard = shuffled.iloc[offset : offset + SHARD_SIZE]
        shard.to_csv(output / f"games_{shard_index:04d}.csv", index=False)
    return (len(shuffled) + SHARD_SIZE - 1) // SHARD_SIZE


def parse_args() -> argparse.Namespace:
    """Parses the fixed data-selection options."""
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--additional-games-per-player",
        type=int,
        default=6,
    )
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "production/full_train_coverage_20261109/shards",
    )
    return parser.parse_args()


def load_selection_state(research: pathlib.Path) -> SelectionState:
    """Loads and validates the training and validation player manifests."""
    train_metadata = read_player_shards(research / "features/shards/train")
    validation_metadata = read_player_shards(
        research / "features/shards/val",
    )
    train_players = set(train_metadata["player_id"])
    validation_players = set(validation_metadata["player_id"])
    if train_players & validation_players:
        raise ValueError("Training and external-validation players overlap")
    rank_per_player = train_metadata.groupby("player_id")["rank"].nunique()
    if (rank_per_player != 1).any():
        raise ValueError("A training player appears with multiple rank labels")
    known_games = {
        player_id: set(player_rows["game_key"])
        for player_id, player_rows in train_metadata.groupby("player_id")
    }
    players_by_rank = {
        rank: set(
            train_metadata.loc[
                train_metadata["rank"] == rank,
                "player_id",
            ],
        )
        for rank in LEVELS
    }
    if sum(len(players) for players in players_by_rank.values()) != len(
        known_games,
    ):
        raise ValueError("Training players are not uniquely assigned to ranks")
    return SelectionState(
        known_games=known_games,
        players_by_rank=players_by_rank,
        validation_players=len(validation_players),
    )


def select_additional_games(
    root: pathlib.Path,
    state: SelectionState,
    additional_games: int,
    seed: int,
) -> pd.DataFrame:
    """Samples the fixed number of additional games for every training player."""
    random_generator = np.random.default_rng(seed)
    selected = []
    for rank in LEVELS:
        config = SamplingConfig(
            player_ids=state.players_by_rank[rank],
            known_games=state.known_games,
            additional_games=additional_games,
            random_generator=random_generator,
        )
        rank_games = select_rank_games(
            rank,
            root / f"data/training/train_{rank}.csv",
            config,
        )
        selected.append(rank_games)
        print(
            f"{rank}: {len(config.player_ids)} players, "
            f"{len(rank_games)} added games",
            flush=True,
        )
    additions = pd.concat(selected, ignore_index=True)
    if additions["game_key"].duplicated().any():
        raise ValueError("Selected game keys are not unique")
    return additions


def write_manifest(
    output: pathlib.Path,
    state: SelectionState,
    additions: pd.DataFrame,
    shards: int,
    args: argparse.Namespace,
) -> None:
    """Writes the reproducible selection summary next to the shards."""
    games_per_player = additions.groupby("player_id").size().reindex(
        state.known_games,
        fill_value=0,
    )
    coverage = games_per_player.value_counts().sort_index().to_dict()
    manifest = {
        "seed": args.seed,
        "additional_games_per_player": args.additional_games_per_player,
        "training_players": len(state.known_games),
        "external_validation_players": state.validation_players,
        "added_games": len(additions),
        "players_with_fewer_than_target": int(
            (games_per_player < args.additional_games_per_player).sum(),
        ),
        "added_games_per_player_histogram": {
            str(count): int(players)
            for count, players in coverage.items()
        },
        "shards": shards,
        "overlap_with_existing_games": 0,
        "overlap_with_external_validation_players": 0,
        "levels": {
            rank: len(state.players_by_rank[rank])
            for rank in LEVELS
        },
    }
    (output / "players.txt").write_text(
        "\n".join(sorted(state.known_games)) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    """Selects additional games without changing source data or prior features."""
    args = parse_args()
    if args.additional_games_per_player <= 0:
        raise ValueError("--additional-games-per-player must be positive")
    root = project_root()
    research = root / "runtime/competition2_research"
    state = load_selection_state(research)
    additions = select_additional_games(
        root,
        state,
        args.additional_games_per_player,
        args.seed,
    )
    shards = write_shards(additions, args.output, args.seed)
    write_manifest(
        args.output,
        state,
        additions,
        shards,
        args,
    )


if __name__ == "__main__":
    main()
