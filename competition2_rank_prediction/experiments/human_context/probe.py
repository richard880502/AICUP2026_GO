"""Checks the installed engine's HumanSL context fields before large extraction.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.human_context.probe
"""

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error

PROFILES = ("rank_5k", "rank_1d", "rank_5d")
FIELDS = (
    "humanWinrate",
    "humanScoreMean",
    "humanScoreStdev",
    "humanStWrError",
    "humanStScoreError",
)


def main() -> None:
    """Writes a small capability report, not a rank-prediction score."""
    args = common.parse_arguments(
        __doc__ or "", "human_context_probe_20261109", True
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite probe: {args.output}")
    shard = sorted((args.research / "features/shards/val").glob("games_*.csv"))[
        0
    ]
    with shard.open(encoding="utf-8", newline="") as handle:
        game = next(csv.DictReader(handle))
    moves = extract_rank_features.parse_moves(game["sgf_content"])
    turns = extract_rank_features.katago.sample_player_turns(
        moves, game["color"], 8
    )
    settings = extract_rank_features.Settings(8, 1, 1, PROFILES)
    queries = extract_rank_features.make_queries(
        "probe", moves, turns, settings
    )[1:]
    client = start_client(args)
    try:
        responses = client.collect(client.submit(queries))
    finally:
        client.close()
    report = summarize(responses, len(turns))
    report["source_game"] = {
        key: value for key, value in game.items() if key != "sgf_content"
    }
    args.output.mkdir(parents=True)
    (args.output / "probe.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "all_fields_present": report["all_fields_present"],
                "mean_profile_range": report["mean_profile_range"],
            }
        ),
        flush=True,
    )


def summarize(
    responses: dict[str, list[dict[str, object]]], count: int
) -> dict[str, Any]:
    """Checks required fields and reports how much profiles change them."""
    records: dict[str, list[dict[str, Any]]] = {}
    for index, profile in enumerate(PROFILES):
        records[profile] = []
        for response in responses.get(f"probe:h{index}", []):
            root = response.get("rootInfo", {})
            if not isinstance(root, dict):
                raise ValueError("Engine rootInfo is not a dictionary")
            records[profile].append({"turn": response["turnNumber"], **root})
        records[profile].sort(key=lambda row: row["turn"])
    available = all(len(rows) == count for rows in records.values())
    for rows in records.values():
        for row in rows:
            available = available and all(field in row for field in FIELDS)
    ranges = {}
    if available:
        for field in FIELDS:
            parts = []
            for profile in PROFILES:
                parts.append([float(row[field]) for row in records[profile]])
            values = np.asarray(parts, dtype=np.float64)
            ranges[field] = float(np.ptp(values, axis=0).mean())
    return {
        "profiles": PROFILES,
        "fields": FIELDS,
        "all_fields_present": available,
        "mean_profile_range": ranges,
        "root_info": records,
        "scope": "single-game capability probe; no accuracy claim",
    }


def start_client(
    args: argparse.Namespace,
) -> extract_rank_features.PipelinedKataGo:
    """Starts the read-only deterministic engine with complete root information."""
    return extract_rank_features.start_engine(
        args.research,
        int(args.device.split(":")[-1]),
        keep_all=True,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
