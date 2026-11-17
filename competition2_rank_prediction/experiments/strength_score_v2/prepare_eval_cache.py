"""Compacts extracted holdout3 960-D rich tokens into mmap-friendly arrays."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def main() -> None:
    """Samples a fixed number of valid moves per holdout game."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=pathlib.Path, required=True)
    parser.add_argument("--rich-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "strength_score_v2/holdout3_cache"
        ),
    )
    parser.add_argument("--moves-per-game", type=int, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.games.open(encoding="utf-8", newline="") as source:
        games = list(csv.DictReader(source))
    shards = sorted(args.rich_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no rich shards in {args.rich_dir}")

    tokens_out = np.lib.format.open_memmap(
        args.output / "tokens.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(games), args.moves_per_game, 960),
    )
    masks_out = np.lib.format.open_memmap(
        args.output / "masks.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(games), args.moves_per_game),
    )
    masks_out[:] = False

    offset = 0
    valid_games = np.zeros(len(games), dtype=np.bool_)
    for path in shards:
        with np.load(path) as arrays:
            shard_tokens = np.asarray(arrays["tokens"])
            shard_masks = np.asarray(arrays["masks"])
            record_ids = np.asarray(arrays["record_ids"]).astype(str)
            if shard_tokens.shape[-1] != 960:
                raise ValueError(f"{path} is not a 960-D rich-token shard")
            stop = offset + len(shard_masks)
            expected = np.asarray(
                [row["game_id"] for row in games[offset:stop]]
            )
            if not np.array_equal(record_ids, expected):
                raise ValueError(f"record id mismatch at {path}")
            for local in range(len(shard_masks)):
                row = offset + local
                tokens_out[row] = 0
                valid = np.flatnonzero(shard_masks[local])
                if not len(valid):
                    continue
                if len(valid) > args.moves_per_game:
                    pick = np.linspace(
                        0,
                        len(valid) - 1,
                        args.moves_per_game,
                        dtype=np.int64,
                    )
                    valid = valid[pick]
                count = len(valid)
                tokens_out[row, :count] = shard_tokens[local, valid]
                masks_out[row, :count] = True
                valid_games[row] = True
            offset = stop
            print(
                json.dumps(
                    {
                        "shard": path.name,
                        "rows": offset,
                        "total": len(games),
                    }
                ),
                flush=True,
            )
    if offset != len(games):
        raise ValueError(
            f"rich shards contain {offset} rows, CSV has {len(games)}"
        )
    tokens_out.flush()
    masks_out.flush()

    np.savez_compressed(
        args.output / "metadata.npz",
        game_id=np.asarray([row["game_id"] for row in games]),
        question_id=np.asarray([row["question_id"] for row in games]),
        player_id=np.asarray([row["player_id"] for row in games]),
        rank=np.asarray(
            [LEVELS.index(row["rank"]) for row in games],
            dtype=np.int64,
        ),
        fold=np.asarray([int(row["fold"]) for row in games], dtype=np.int64),
        valid_game=valid_games,
        levels=np.asarray(LEVELS),
    )
    report = {
        "games": len(games),
        "valid_games": int(valid_games.sum()),
        "empty_games": int((~valid_games).sum()),
        "moves_per_game": args.moves_per_game,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "
", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
