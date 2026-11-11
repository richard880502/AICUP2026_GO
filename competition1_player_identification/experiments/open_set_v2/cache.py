"""Caches training-only move tokens and canonical game fingerprints.

Example from the repository root:
    python -m competition1_player_identification.experiments.open_set_v2.cache

Shards are independent and resumable. No official test files are read.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import pathlib
import re
import time

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts

PROPERTY_PATTERN = re.compile(r"([A-Z]+)((?:\[(?:\\.|[^\]\\])*\])+)")
VALUE_PATTERN = re.compile(r"\[((?:\\.|[^\]\\])*)\]")
SETUP_PROPERTIES = frozenset(
    (
        "SZ",
        "AB",
        "AW",
    )
)
RANKS = (
    "D",
    "C",
    "B",
    "A",
    "1D",
    "2D",
    "3D",
    "4D",
    "5D",
    "6D",
)
MAX_MOVES = 384


def parse_game(content: str) -> tuple[str, np.ndarray, int]:
    """Retains setup and every move for hashing, and bounded sequence tokens."""
    canonical = []
    moves = []
    for name, encoded in PROPERTY_PATTERN.findall(content):
        values = VALUE_PATTERN.findall(encoded)
        if name in SETUP_PROPERTIES:
            canonical.append(name + json.dumps(values, separators=(",", ":")))
        elif name in ("B", "W"):
            if len(values) != 1:
                raise ValueError(f"Move property has multiple values: {name!r}")
            coordinate = values[0]
            canonical.append(name + coordinate)
            position = 362
            if coordinate not in ("", "tt"):
                if len(coordinate) != 2:
                    raise ValueError(f"Malformed coordinate: {coordinate!r}")
                column, row = (ord(value) - ord("a") for value in coordinate)
                if not (0 <= column < 19 and 0 <= row < 19):
                    raise ValueError(f"Non-19x19 coordinate: {coordinate!r}")
                position = column * 19 + row + 1
            moves.append(position + (362 if name == "W" else 0))
    if not moves:
        raise ValueError("Training game contains no moves")
    tokens = np.zeros(MAX_MOVES, dtype=np.uint16)
    retained = min(len(moves), MAX_MOVES)
    tokens[:retained] = moves[:retained]
    digest = hashlib.sha256("|".join(canonical).encode()).hexdigest()
    return (
        digest,
        tokens,
        len(
            moves,
        ),
    )


def cache_file(task: tuple[str, str]) -> dict:
    """Converts one rank CSV, verifying its header and record fields."""
    source = pathlib.Path(task[0])
    destination = pathlib.Path(task[1])
    if destination.exists():
        return {"file": str(destination), "status": "existing"}
    token_rows = []
    columns: dict[str, list] = {
        name: []
        for name in (
            "players",
            "game_ids",
            "colors",
            "hashes",
            "lengths",
        )
    }
    started = time.monotonic()
    with source.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row["color"] not in ("B", "W") or row["rank"] not in RANKS:
                raise ValueError(f"Invalid rank/color in {row['game_id']!r}")
            digest, tokens, length = parse_game(row["sgf_content"])
            token_rows.append(tokens)
            columns["players"].append(row["player_id"])
            columns["game_ids"].append(row["game_id"])
            columns["colors"].append(int(row["color"] == "W"))
            columns["hashes"].append(digest)
            columns["lengths"].append(length)
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    arrays["tokens"] = np.stack(token_rows)
    arrays["rank"] = np.full(
        len(token_rows),
        RANKS.index(source.stem[6:]),
        np.int8,
    )
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            tokens=arrays["tokens"],
            players=arrays["players"],
            game_ids=arrays["game_ids"],
            colors=arrays["colors"],
            hashes=arrays["hashes"],
            lengths=arrays["lengths"],
            rank=arrays["rank"],
        )
    temporary.replace(destination)
    return {
        "file": str(destination),
        "games": len(token_rows),
        "seconds": time.monotonic() - started,
        "status": "created",
    }


def main() -> None:
    """Builds independent rank shards with bounded process parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/training",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output",
        default="production/open_set_v2_20261111/raw_cache",
        type=pathlib.Path,
    )
    artifacts.add_workers(parser)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for rank in RANKS:
        tasks.append(
            (
                str(arguments.source / f"train_{rank}.csv"),
                str(arguments.output / f"{rank}.npz"),
            )
        )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=arguments.workers,
    ) as executor:
        for result in executor.map(cache_file, tasks):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
