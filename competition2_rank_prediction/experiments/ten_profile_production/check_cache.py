"""Validate that a ten-profile extraction is suitable for the matched P1 test."""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

EXPECTED = (
    "normal",
    "rank_11k", "rank_8k", "rank_5k", "rank_2k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d",
)
PRODUCTION_INDICES = (0, 3, 5, 9)
PRODUCTION_NAMES = ("normal", "rank_5k", "rank_1d", "rank_5d")


def main() -> None:
    """Checks every shard's profile order, shape, finiteness and metadata alignment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument(
        "--shard-dir",
        type=pathlib.Path,
        help="optional games_*.csv source; require one feature file per source shard",
    )
    args = parser.parse_args()

    shards = sorted(args.features.glob("feat_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no feat_*.npz under {args.features}")
    if args.shard_dir is not None:
        source = sorted(args.shard_dir.glob("games_*.csv"))
        expected = {
            path.name.replace("games_", "feat_").replace(".csv", ".npz")
            for path in source
        }
        actual = {path.name for path in shards}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"shard coverage mismatch: source={len(source)} features={len(shards)} "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    games = 0
    samples: set[int] = set()
    for path in shards:
        with np.load(path, allow_pickle=True) as saved:
            profiles = tuple(str(value) for value in saved["profiles"])
            if profiles != EXPECTED:
                raise ValueError(
                    f"{path}: profile order {profiles} != expected {EXPECTED}"
                )
            tokens = np.asarray(saved["tokens"])
            masks = np.asarray(saved["masks"])
            if tokens.ndim != 4 or tokens.shape[2:] != (11, 3):
                raise ValueError(f"{path}: unexpected token shape {tokens.shape}")
            if masks.shape != tokens.shape[:2]:
                raise ValueError(
                    f"{path}: masks {masks.shape} do not match tokens {tokens.shape[:2]}"
                )
            for key in ("game_key", "player_id", "rank"):
                if len(saved[key]) != len(tokens):
                    raise ValueError(f"{path}: {key} length mismatch")
            if not np.isfinite(tokens).all():
                raise ValueError(f"{path}: non-finite token")
            games += len(tokens)
            samples.add(tokens.shape[1])

    selected = tuple(EXPECTED[index] for index in PRODUCTION_INDICES)
    if selected != PRODUCTION_NAMES:
        raise AssertionError(
            f"production indices map to {selected}, expected {PRODUCTION_NAMES}"
        )

    print(
        f"{args.features}: {len(shards)} shards, {games:,} games, "
        f"samples={sorted(samples)}, profiles OK"
    )
    print(
        "matched 3-profile baseline uses indices",
        ",".join(str(index) for index in PRODUCTION_INDICES),
        "=>",
        selected,
    )


if __name__ == "__main__":
    main()
