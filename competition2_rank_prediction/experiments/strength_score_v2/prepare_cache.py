"""Builds a compact player-balanced cache from existing 960-D rich shards."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.trunk_style import train as style_train

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def shard_layout(directory: pathlib.Path) -> tuple[list[pathlib.Path], list[int]]:
    paths = sorted(directory.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no rich shards in {directory}")
    sizes = []
    for path in paths:
        with np.load(path) as arrays:
            sizes.append(len(arrays["masks"]))
    return paths, sizes


def corpus_metadata(base_count: int, extra_count: int) -> tuple[np.ndarray, ...]:
    everything = np.load(style_train.OPEN / "engine_inputs/global_indices.npy")
    base_global = np.asarray(everything[:base_count], dtype=np.int64)
    extra_global = np.load(style_train.ROOT / "inputs/corpus2_global_indices.npy")
    if len(extra_global) != extra_count:
        raise ValueError(
            f"corpus2 rows {extra_count} != global index rows {len(extra_global)}"
        )
    global_rows = np.concatenate((base_global, extra_global.astype(np.int64)))
    raw = dataset.load_arrays(style_train.OPEN / "raw_cache", ("players",))
    players = raw["players"][global_rows].astype(str)
    ranks = (global_rows // 100000).astype(np.int64)
    return players, ranks, global_rows


def select_games(
    players: np.ndarray,
    ranks: np.ndarray,
    minimum_games: int,
    games_per_player: int,
    max_players_per_rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for row, player in enumerate(players):
        groups.setdefault(str(player), []).append(row)

    labels: dict[str, int] = {}
    eligible: list[list[str]] = [[] for _ in LEVELS]
    for player, rows in groups.items():
        labels_found = np.unique(ranks[np.asarray(rows, dtype=np.int64)])
        if len(labels_found) != 1:
            raise ValueError(f"inconsistent labels for {player}")
        label = int(labels_found[0])
        labels[player] = label
        if len(rows) >= minimum_games:
            eligible[label].append(player)

    rng = np.random.default_rng(seed)
    chosen_players: list[str] = []
    for label, names in enumerate(eligible):
        names = sorted(names)
        if max_players_per_rank and len(names) > max_players_per_rank:
            names = sorted(
                str(name)
                for name in rng.choice(
                    np.asarray(names),
                    size=max_players_per_rank,
                    replace=False,
                )
            )
        if not names:
            raise ValueError(f"no eligible players for {LEVELS[label]}")
        chosen_players.extend(names)

    player_index = {player: index for index, player in enumerate(chosen_players)}
    source_rows: list[int] = []
    game_player: list[int] = []
    for player in chosen_players:
        rows = np.asarray(groups[player], dtype=np.int64)
        take = min(games_per_player, len(rows))
        selected = rng.choice(rows, size=take, replace=False)
        source_rows.extend(int(row) for row in selected)
        game_player.extend([player_index[player]] * take)

    source = np.asarray(source_rows, dtype=np.int64)
    owner = np.asarray(game_player, dtype=np.int32)
    order = np.argsort(source, kind="stable")
    return (
        source[order],
        owner[order],
        np.asarray(chosen_players),
        np.asarray([labels[p] for p in chosen_players], dtype=np.int64),
    )


def copy_rows(
    paths: list[pathlib.Path],
    sizes: list[int],
    source_rows: np.ndarray,
    output: pathlib.Path,
    moves_per_game: int,
) -> None:
    tokens_out = np.lib.format.open_memmap(
        output / "tokens.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(source_rows), moves_per_game, 960),
    )
    masks_out = np.lib.format.open_memmap(
        output / "masks.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(source_rows), moves_per_game),
    )
    masks_out[:] = False
    cursor = 0
    copied = 0
    for path, size in zip(paths, sizes):
        stop = cursor + size
        left = np.searchsorted(source_rows, cursor, side="left")
        right = np.searchsorted(source_rows, stop, side="left")
        if right > left:
            with np.load(path) as arrays:
                source_tokens = np.asarray(arrays["tokens"])
                source_masks = np.asarray(arrays["masks"])
                if source_tokens.shape[-1] != 960:
                    raise ValueError(f"{path} is not 960-D rich tokens")
                for target in range(left, right):
                    local = int(source_rows[target] - cursor)
                    valid = np.flatnonzero(source_masks[local])
                    tokens_out[target] = 0
                    if len(valid):
                        if len(valid) > moves_per_game:
                            valid = valid[
                                np.linspace(
                                    0,
                                    len(valid) - 1,
                                    moves_per_game,
                                    dtype=np.int64,
                                )
                            ]
                        count = len(valid)
                        tokens_out[target, :count] = source_tokens[local, valid]
                        masks_out[target, :count] = True
                    copied += 1
        cursor = stop
        print(
            json.dumps(
                {"shard": path.name, "copied": copied, "total": len(source_rows)}
            ),
            flush=True,
        )
    if copied != len(source_rows):
        raise ValueError(f"copied {copied}/{len(source_rows)} selected games")
    tokens_out.flush()
    masks_out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-rich",
        type=pathlib.Path,
        default=style_train.ROOT / "corpus_rich",
    )
    parser.add_argument(
        "--extra-rich",
        type=pathlib.Path,
        default=style_train.ROOT / "corpus2_rich",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/strength_score_v2/train_cache"
        ),
    )
    parser.add_argument("--minimum-games", type=int, default=4)
    parser.add_argument("--games-per-player", type=int, default=12)
    parser.add_argument("--moves-per-game", type=int, default=48)
    parser.add_argument("--max-players-per-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base_paths, base_sizes = shard_layout(args.base_rich)
    extra_paths, extra_sizes = shard_layout(args.extra_rich)
    players, ranks, global_rows = corpus_metadata(
        sum(base_sizes), sum(extra_sizes)
    )
    source_rows, game_player, player_id, player_rank = select_games(
        players,
        ranks,
        args.minimum_games,
        args.games_per_player,
        args.max_players_per_rank,
        args.seed,
    )
    copy_rows(
        base_paths + extra_paths,
        base_sizes + extra_sizes,
        source_rows,
        args.output,
        args.moves_per_game,
    )
    np.savez_compressed(
        args.output / "metadata.npz",
        game_player_index=game_player,
        player_id=player_id,
        player_rank=player_rank,
        source_corpus_row=source_rows,
        source_global_row=global_rows[source_rows],
        levels=np.asarray(LEVELS),
    )
    counts = np.bincount(player_rank, minlength=len(LEVELS))
    report = {
        "players": int(len(player_id)),
        "games": int(len(source_rows)),
        "moves_per_game": args.moves_per_game,
        "players_by_rank": {
            level: int(counts[index]) for index, level in enumerate(LEVELS)
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
