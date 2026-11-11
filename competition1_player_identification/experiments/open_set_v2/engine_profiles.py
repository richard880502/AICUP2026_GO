"""Verifies complete engine shards and materializes training-only rich profiles.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.engine_profiles

A partially extracted corpus is rejected. Original source game ordering and
feature schema are checked before the memory-mapped profile matrix is written.
"""

import concurrent.futures
import csv
import json
import pathlib

import numpy as np

from competition1_player_identification.production_pipeline import data
from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset


def expected_ids(root: pathlib.Path) -> list[str]:
    """Reads identifiers in the immutable, selected training-corpus order."""
    with (root / "engine_inputs/games.csv").open(
        encoding="utf-8",
        newline="",
    ) as source:
        return [row["game_id"] for row in csv.DictReader(source)]


def profile_tasks(root: pathlib.Path, workers: int) -> list[pathlib.Path]:
    """Runs bounded CPU aggregation with the production 64-decision schema."""
    metadata = json.loads((root / "engine_features/metadata.json").read_text())
    if metadata["samples_per_game"] != 64 or len(metadata["features"]) != 41:
        raise ValueError(
            "Expected 64 decisions and the established 41 channels"
        )
    paths = sorted((root / "engine_features/candidates").glob("shard_*.npz"))
    count = len(np.load(root / "engine_inputs/global_indices.npy"))
    if len(paths) != (count + 127) // 128:
        raise ValueError("Engine extraction is incomplete")
    destination = root / "engine_profile_shards"
    destination.mkdir(exist_ok=True)
    tasks = [
        (
            path,
            destination / path.name,
            tuple(metadata["features"]),
        )
        for path in paths
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        return list(executor.map(data.profile_shard, tasks))


def merge_profiles(root: pathlib.Path, paths: list[pathlib.Path]) -> dict:
    """Checks source identifiers and profile values before publishing a matrix."""
    identifiers = expected_ids(root)
    destination = root / "engine_profiles.npy"
    if destination.exists():
        raise FileExistsError(f"Profile matrix already exists: {destination}")
    temporary = root / "engine_profiles.tmp.npy"
    result = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(identifiers), 9198),
    )
    offset = 0
    for path in paths:
        with np.load(path) as payload:
            profiles = np.asarray(payload["profiles"])
            count = len(profiles)
            if (
                np.asarray(payload["record_ids"]).tolist()
                != identifiers[offset : offset + count]
            ):
                raise ValueError(f"Profile/source record mismatch: {path}")
            if profiles.shape[1] != 9198 or not np.isfinite(profiles).all():
                raise ValueError(
                    f"Invalid profile schema or numeric values: {path}"
                )
            result[offset : offset + count] = profiles
            offset += count
    if offset != len(identifiers):
        raise ValueError(f"Expected {len(identifiers)} profiles; got {offset}")
    result.flush()
    del result
    temporary.replace(destination)
    report = {
        "games": offset,
        "features": 9198,
        "status": "passed",
    }
    artifacts.write_report(root / "engine_profile_audit.json", report)
    return report


def main() -> None:
    """Validates and materializes all engine profile shards after extraction."""
    parser = artifacts.root_parser(__doc__, dataset.ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    arguments = parser.parse_args()
    paths = profile_tasks(arguments.root, arguments.workers)
    print(json.dumps(merge_profiles(arguments.root, paths), indent=2))


if __name__ == "__main__":
    main()
