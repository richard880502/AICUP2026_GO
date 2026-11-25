# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Caches exact binary board inputs for matched frozen/adapted rank probes.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Six CPU workers encode training and historical validation; no GPU is used.
"""

import concurrent.futures
import csv
import functools
import hashlib
import json
import multiprocessing
import pathlib
import sys

import numpy as np

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import equivalence
from competition2_rank_prediction.experiments.human_trunk import extract

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def encode(research: str, config: dict, game: tuple[str, str]) -> dict:
    """Encodes SGF/color without exposing true rank to metadata construction."""
    moves = extract_rank_features.parse_moves(game[0])
    turns = extract_rank_features.katago.sample_player_turns(moves, game[1], 8)
    if not turns:
        raise ValueError("No eligible own decisions in selected pilot game")
    arrays = equivalence.make_inputs(
        extract.worker_modules(research), config, moves, turns
    )
    if not np.isin(arrays[0], (0, 1)).all():
        raise ValueError(
            "Spatial features are not binary; packing would be lossy"
        )
    rows = np.minimum(np.arange(8), len(turns) - 1)
    points = np.asarray(
        [
            361 if moves[turn].x is None else moves[turn].y * 19 + moves[turn].x
            for turn in turns
        ],
        dtype=np.int16,
    )
    packed = np.packbits(arrays[0][rows].astype(np.uint8), axis=-1)
    recovered = np.unpackbits(packed, axis=-1)[..., :19]
    if not np.array_equal(recovered, arrays[0][rows]):
        raise ValueError("Binary packing round trip failed")
    return {
        "spatial_packed": packed,
        "global": arrays[1][rows].astype(np.float32),
        "meta": arrays[2][rows].astype(np.float32),
        "points": points[rows],
        "valid": np.arange(8) < len(turns),
    }


def write_shard(
    pool, operation, games: list, destination: pathlib.Path
) -> None:
    """Writes one bounded500-game batch after all input round trips pass."""
    encoded = list(
        pool.map(
            operation,
            [(game["sgf_content"], game["color"]) for game in games],
            chunksize=8,
        )
    )
    values: dict = {
        name: np.stack([item[name] for item in encoded])
        for name in ("spatial_packed", "global", "meta", "points", "valid")
    }
    for name in ("game_key", "player_id", "rank"):
        values[name] = np.asarray([game[name] for game in games], dtype=str)
    np.savez_compressed(destination, **values)
    print(f"Saved {destination.name}: {len(games)} games", flush=True)


def main() -> None:
    """Creates isolated exact-input caches, refusing existing output directories."""
    root = pathlib.Path.cwd()
    research = root / "runtime/competition2_research"
    output = root / "production/human_finetune_inputs_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing input cache: {output}")
    operation = functools.partial(
        encode,
        str(research),
        bridge.checkpoint(
            research / "models/human_trunk/b18c384nbt-humanv0.ckpt"
        )["config"],
    )
    sources = {
        "train": root / "production/game_coverage_manifest_20261109/cap8",
        "evaluation": research / "features/shards/val",
    }
    output.mkdir(parents=True)
    manifest = {"samples": 8, "metadata": "neutral fixed1d; no rank input"}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for split, source in sources.items():
            destination = output / split
            destination.mkdir()
            count = 0
            for path in sorted(source.glob("games_*.csv")):
                manifest[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                with path.open(encoding="utf-8", newline="") as handle:
                    games = list(csv.DictReader(handle))
                for start in range(0, len(games), 500):
                    write_shard(
                        pool,
                        operation,
                        games[start : start + 500],
                        destination / f"inputs_{count:04d}.npz",
                    )
                    count += 1
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
