"""Loads verified production shards and caches the established rich profiles.

Used by ``python -m competition1_player_identification.production_pipeline.run``.
"""

import concurrent.futures
import csv
import dataclasses
import json
import pathlib

import numpy as np

from competition1_player_identification import cache_rich_profiles


@dataclasses.dataclass
class Candidates:
    """Stores profiles grouped in the published candidate-player order."""

    players: np.ndarray
    profiles: np.ndarray


@dataclasses.dataclass
class Queries:
    """Stores flattened query profiles and their question membership."""

    identifiers: np.ndarray
    counts: np.ndarray
    groups: np.ndarray
    profiles: np.ndarray


@dataclasses.dataclass
class Pool:
    """Stores candidate and query data without any query-answer labels."""

    candidates: Candidates
    queries: Queries


def profile_shard(task: tuple) -> pathlib.Path:
    """Caches one immutable decision shard as rich game profiles."""
    source, destination, names = task
    if destination.exists():
        return destination
    keep = [
        index for index, name in enumerate(names)
        if name not in cache_rich_profiles.DROP_FEATURES
    ]
    selected_names = tuple(names[index] for index in keep)
    with np.load(source) as arrays:
        profiles = cache_rich_profiles.make_profile_chunk(
            np.asarray(arrays["tokens"])[:, :, keep],
            arrays["masks"],
            selected_names,
            cache_rich_profiles.make_profile_config((1, 2, 4, 8, 16, 32)),
        )
        payload = {
            "profiles": profiles.astype(np.float32),
            "group_indices": arrays["group_indices"],
            "record_ids": arrays["record_ids"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **payload)
    temporary.replace(destination)
    return destination


def load_group(directory: pathlib.Path, expected_ids: list[str]) -> dict:
    """Checks the exact record order before concatenating profile shards."""
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("profiles", "group_indices", "record_ids")
    }
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as arrays:
            for key in parts:
                parts[key].append(arrays[key])
    if not parts["profiles"]:
        raise ValueError(f"No profile shards in {directory!r}")
    result = {key: np.concatenate(values) for key, values in parts.items()}
    if result["record_ids"].tolist() != expected_ids:
        raise ValueError(f"Incomplete or reordered feature records: {directory!r}")
    if not np.isfinite(result["profiles"]).all():
        raise ValueError(f"Nonfinite game profiles: {directory!r}")
    return result


def source_ids(metadata: dict, group: str) -> list[str]:
    """Reads expected record IDs directly from the authorized input CSV."""
    key = "candidates" if group == "candidates" else "questions"
    with pathlib.Path(metadata[key]).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if group == "candidates":
        return [row["game_id"] for row in rows]
    identifiers = []
    for row in rows:
        for offset in range(1, int(row["num_games"]) + 1):
            identifiers.append(f"{row['question_id']}:{offset}")
    return identifiers


def load_pool(root: pathlib.Path, workers: int) -> Pool:
    """Builds profiles and checks every candidate/query feature is present."""
    metadata = json.loads((root / "features/metadata.json").read_text())
    tasks = []
    for group in ("candidates", "queries"):
        for path in sorted((root / "features" / group).glob("shard_*.npz")):
            tasks.append((path, root / "profiles" / group / path.name,
                          metadata["features"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(profile_shard, tasks))
    candidate = load_group(root / "profiles/candidates",
                           source_ids(metadata, "candidates"))
    query = load_group(root / "profiles/queries", source_ids(metadata, "queries"))
    grouped = []
    for index in range(len(metadata["player_ids"])):
        grouped.append(candidate["profiles"][candidate["group_indices"] == index])
    if len({len(rows) for rows in grouped}) != 1 or len(grouped[0]) < 10:
        raise ValueError("Candidate players must have equal, complete game counts")
    counts = np.asarray(metadata["query_game_counts"])
    if not np.array_equal(np.bincount(query["group_indices"]), counts):
        raise ValueError("Query game counts disagree with extraction metadata")
    return Pool(
        Candidates(np.asarray(metadata["player_ids"]), np.stack(grouped)),
        Queries(np.asarray(metadata["question_ids"]), counts,
                query["group_indices"], query["profiles"]),
    )
