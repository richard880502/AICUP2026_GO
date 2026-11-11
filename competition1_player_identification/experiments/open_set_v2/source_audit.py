"""Checks source CSV identity, board settings, and cache alignment.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.source_audit

This check rejects unsupported setup stones and fingerprints source files.
HD is reported; the inherited fixed-komi engine representation is a proxy.
"""

import collections
import concurrent.futures
import csv
import hashlib
import json
import pathlib
import re

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import cache
from competition1_player_identification.experiments.open_set_v2 import dataset

SETTING_PATTERN = re.compile(r"([A-Z]+)\[([^\]]*)\]")


def source_digest(source: pathlib.Path) -> str:
    """Fingerprints source bytes without retaining the complete CSV."""
    digest = hashlib.sha256()
    with source.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def root_settings(content: str) -> dict:
    """Rejects board setups unsupported by the engine request wrapper."""
    metadata = dict(SETTING_PATTERN.findall(content.split(";", 2)[1]))
    if metadata.get("SZ") != "19" or int(metadata.get("HD", "0")) < 0:
        raise ValueError(f"Unsupported board settings: {metadata!r}")
    if "AB" in metadata or "AW" in metadata:
        raise ValueError("Setup stones require a different engine request path")
    return metadata


def audit_rank(rank: str) -> dict:
    """Checks one entire rank CSV against its corresponding raw cache."""
    source = pathlib.Path("data/training") / f"train_{rank}.csv"
    with np.load(dataset.ROOT / "raw_cache" / f"{rank}.npz") as arrays:
        players = np.asarray(arrays["players"])
        identifiers = np.asarray(arrays["game_ids"])
        colors = np.asarray(arrays["colors"])
        lengths = np.asarray(arrays["lengths"])
    settings: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    rows = 0
    with source.open(encoding="utf-8", newline="") as input_file:
        for index, row in enumerate(csv.DictReader(input_file)):
            if (
                row["player_id"] != players[index]
                or row["game_id"] != identifiers[index]
                or row["color"] != ("W" if colors[index] else "B")
                or row["rank"] != rank
            ):
                raise ValueError(f"Source/cache mismatch: {source}:{index}")
            for name, value in root_settings(row["sgf_content"]).items():
                settings[name][value] += 1
            rows += 1
    if rows != len(players):
        raise ValueError(f"Source/cache size mismatch: {source}")
    return {
        "rank": rank,
        "rows": rows,
        "sha256": source_digest(source),
        "board_settings": {
            key: dict(values) for key, values in settings.items()
        },
        "move_length_quantiles": np.quantile(
            lengths,
            (
                0,
                0.1,
                0.5,
                0.9,
                1,
            ),
        ).tolist(),
        "status": "passed",
    }


def main() -> None:
    """Runs a full independent source audit with one worker per rank file."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        reports = list(executor.map(audit_rank, cache.RANKS))
    payload = {
        "status": "passed",
        "files": reports,
        "engine_representation": "moves-only Japanese 6.5-komi proxy",
        "limitation": "HD is not converted to assumed initial stones or komi",
    }
    destination = dataset.ROOT / "source_audit.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if [entry["sha256"] for entry in previous["files"]] != [
            entry["sha256"] for entry in reports
        ]:
            raise ValueError(
                "Source files changed; existing raw caches cannot be reused"
            )
    (dataset.ROOT / "source_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
