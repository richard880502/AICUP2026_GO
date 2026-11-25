"""Benchmarks actual32visit analysis and checks searched/raw value fields.

Example: python -m competition2_rank_prediction.experiments.search_regret.probe
This is API/cost validation, not a rank-score experiment.
"""

import contextlib
import csv
import itertools
import json
import pathlib
import sys
import tempfile
import time
from typing import Any, Iterator

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error

ROOT_FIELDS = ("visits", "winrate", "scoreLead", "rawWinrate", "rawLead")
VISITS = 32


@contextlib.contextmanager
def engine(args) -> Iterator[Any]:
    """Starts a deterministic normal-only engine onGPU1.

    Yields:
        Full-response client with no HumanSL model loaded.
    """
    if args.device != "cuda:1":
        raise ValueError("Search pilot is restricted toGPU1")
    with tempfile.TemporaryDirectory(
        prefix="aicup_search_regret_"
    ) as directory:
        config = pathlib.Path(directory) / "analysis.cfg"
        template = (
            args.research / "configs/analysis_fast.cfg.tmpl"
        ).read_text()
        config.write_text(
            template.replace("@DEVICE@", "1")
            + "\nnnRandomize = false\nnnRandSeed = search_regret\n"
            + "forDeterministicTesting = true\n"
        )
        with extract_rank_features.katago.KataGoClient(
            args.research / "c1/scripts/run_katago.sh",
            config,
            args.research
            / "c1/models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
        ) as client:
            yield client


def query_game(game: dict[str, str], identity: str) -> dict[str, object]:
    """Uses fixed positions for cost measurement, not the score pilot."""
    moves = extract_rank_features.parse_moves(game["sgf_content"])
    turns = []
    for color in ("B", "W"):
        turns.extend(
            extract_rank_features.katago.sample_player_turns(moves, color, 4)
        )
    return extract_rank_features.katago.make_analysis_query(
        identity,
        moves,
        sorted(set(turns)),
        VISITS,
        extract_rank_features.katago.AnalysisQueryConfig(True),
    ) | {"includeOwnership": False}


def summarize(responses: dict) -> dict:
    """Requires raw/root/child values and records actual search visits."""
    roots = []
    children = []
    for rows in responses.values():
        for response in rows:
            root = response["rootInfo"]
            if not all(field in root for field in ROOT_FIELDS):
                raise ValueError(
                    "Search root lacks required searched/raw fields"
                )
            candidates = sorted(
                response["moveInfos"], key=lambda row: row["order"]
            )
            if not candidates:
                raise ValueError("Search returned no candidate move values")
            child = candidates[0]
            if not all(
                field in child for field in ("scoreLead", "winrate", "visits")
            ):
                raise ValueError(
                    "Best searched child lacks score/winrate/visits"
                )
            roots.append(root)
            children.append(child)
    if not roots:
        raise ValueError("Search probe returned no positions")
    actual = np.asarray([root["visits"] for root in roots], dtype=float)
    shifts = np.asarray(
        [abs(root["scoreLead"] - root["rawLead"]) for root in roots]
    )
    if not np.isfinite(shifts).all() or actual.min() < VISITS:
        raise ValueError(
            "Nonfinite search values or actual visit count below32"
        )
    return {
        "positions": len(roots),
        "actual_visits_min": float(actual.min()),
        "actual_visits_max": float(actual.max()),
        "mean_abs_searched_minus_raw_lead": float(shifts.mean()),
        "root_fields": sorted(roots[0]),
        "child_fields": sorted(children[0]),
        "scope": "fixed16game cost/API probe, not rank utility",
    }


def main() -> None:
    """Measures a warmed fixed batch separately from model startup."""
    args = common.parse_arguments(
        __doc__ or "", "search_regret_probe_20261109", True
    )
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite search probe: {args.output}"
        )
    with (args.research / "features/shards/val/games_0000.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        games = list(itertools.islice(csv.DictReader(handle), 16))
    queries = [query_game(game, str(index)) for index, game in enumerate(games)]
    with engine(args) as client:
        client.analyze([query_game(games[-1], "warmup")])
        started = time.perf_counter()
        responses = client.analyze(queries)
        elapsed = time.perf_counter() - started
    report = summarize(responses)
    report["analysis_seconds_after_warmup"] = elapsed
    report["positions_per_second"] = report["positions"] / elapsed
    report["projected_8000game_seconds_at16positions"] = (
        8000 * 16 / report["positions_per_second"]
    )
    args.output.mkdir(parents=True)
    (args.output / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
