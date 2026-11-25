# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Extracts searched/raw regret from fixed surprising own/opponent decisions.

Example: python -m competition2_rank_prediction.experiments.search_regret.extract
Outputs compact feature and raw-value audit shards, not new rank predictions.
"""

import csv
import dataclasses
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.search_regret import features
from competition2_rank_prediction.experiments.search_regret import probe
from competition2_rank_prediction.experiments.search_regret import selection

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error
import rank_lib  # pylint: disable=wrong-import-position,import-error


@dataclasses.dataclass
class Prepared:
    """Stores selection, not ground-truth rank, for one query.

    Attributes:
        identity: Engine request ID.
        length: Number of parsed moves.
        color: Labelled player's actual color.
        own: Selected own turn/surprisal pairs.
        opponent: Selected opponent turn/surprisal pairs.
    """

    identity: str
    length: int
    color: str
    own: list[tuple[int, float]]
    opponent: list[tuple[int, float]]


@dataclasses.dataclass
class Result:
    """Stores compact features and their independently recomputable audit trace.

    Attributes:
        values: Raw14/search12features.
        decisions: Six per-decision raw/search/context values.
        source: Eight BLACK-perspective source values per decision.
        visits: Pre-root/post-root/best-child visits.
        turns: Selected actual-move indices.
        masks: Whether a selected decision was returned and valid.
    """

    values: np.ndarray
    decisions: np.ndarray
    source: np.ndarray
    visits: np.ndarray
    turns: np.ndarray
    masks: np.ndarray


def cache(directory: pathlib.Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loads selection arrays from immutable production96decision caches."""
    payload = rank_lib.load_features(directory)
    if payload["tokens"].shape[1] != 96:
        raise ValueError(
            "Search selection requires the original96decision cache"
        )
    return {
        str(key): (
            payload["tokens"][index, :, 0, 0].copy(),
            payload["masks"][index].copy(),
        )
        for index, key in enumerate(payload["game_key"])
    }


def prepare(
    game: dict[str, str], identity: str, caches: tuple[dict, dict]
) -> tuple[Prepared, dict[str, object] | None]:
    """Builds one32visit request for both roles and actual post-move turns."""
    moves = extract_rank_features.parse_moves(game["sgf_content"])
    color = game["color"]
    opposite = "W" if color == "B" else "B"
    own = selection.select_turns(moves, color, *caches[0][game["game_key"]])
    opponent = selection.select_turns(
        moves, opposite, *caches[1][game["game_key"]]
    )
    prepared = Prepared(identity, len(moves), color, own, opponent)
    turns = sorted({pair[0] for pair in own + opponent})
    if not turns:
        return prepared, None
    query = extract_rank_features.katago.make_analysis_query(
        identity,
        moves,
        turns,
        probe.VISITS,
        extract_rank_features.katago.AnalysisQueryConfig(True),
    )
    query["includeOwnership"] = False
    query["includePolicy"] = False
    return prepared, query


def assemble(prepared: Prepared, responses: list[dict]) -> Result:
    """Computes features and retains each decision's source values/visits."""
    result = Result(
        np.zeros(26),
        np.zeros((2, 4, 6)),
        np.zeros((2, 4, 8)),
        np.zeros((2, 4, 3), dtype=np.int32),
        np.full((2, 4), -1, dtype=np.int32),
        np.zeros((2, 4), dtype=bool),
    )
    lookup = {int(response["turnNumber"]): response for response in responses}
    raw, searched = [], []
    for role, selected in enumerate((prepared.own, prepared.opponent)):
        color = (
            prepared.color
            if role == 0
            else ("W" if prepared.color == "B" else "B")
        )
        values = []
        for index, (turn, surprise) in enumerate(selected):
            result.turns[role, index] = turn
            if turn not in lookup or turn + 1 not in lookup:
                continue
            result.decisions[role, index] = features.decision_values(
                lookup[turn],
                lookup[turn + 1],
                color,
                surprise,
                turn / max(prepared.length - 1, 1),
            )
            result.source[role, index], result.visits[role, index] = (
                features.source_values(lookup[turn], lookup[turn + 1])
            )
            result.masks[role, index] = True
            values.append(result.decisions[role, index])
        role_raw, role_search = features.aggregate(values)
        raw.append(role_raw)
        searched.append(role_search)
    result.values = np.concatenate(raw + searched)
    return result


def extract_games(
    client, games: list[dict[str, str]], caches: tuple[dict, dict]
) -> list[Result]:
    """Runs bounded batches without inspecting rank outcomes."""
    results = []
    for start in range(0, len(games), 32):
        prepared, queries = [], []
        for index in range(start, min(start + 32, len(games))):
            item, query = prepare(games[index], str(index), caches)
            prepared.append(item)
            if query is not None:
                queries.append(query)
        responses = client.analyze(queries) if queries else {}
        for item in prepared:
            results.append(assemble(item, responses.get(item.identity, [])))
        if (start + 32) % 128 == 0:
            print(
                f"searched {min(start + 32, len(games))}/{len(games)} "
                "games in shard",
                flush=True,
            )
    return results


def write_shard(
    path: pathlib.Path, games: list[dict[str, str]], results: list[Result]
) -> None:
    """Atomically saves descriptors/source traces with immutable game keys."""
    arrays: dict[str, Any] = {}
    for name in ("values", "decisions", "source", "visits", "turns", "masks"):
        arrays[name] = np.stack([getattr(result, name) for result in results])
    for name in ("game_key", "player_id", "rank", "color"):
        arrays[name] = np.asarray([game[name] for game in games])
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, allow_pickle=False, **arrays)
    temporary.rename(path)


def record_manifest(args) -> None:
    """Prevents mixed resumes and records source/selection-cache hashes."""
    source = pathlib.Path(__file__).resolve().parent
    hashes = {}
    for path in (
        source / "extract.py",
        source / "features.py",
        source / "selection.py",
        source / "probe.py",
    ):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ("feat96/val", "feat96opp/val"):
        for path in sorted(
            (args.research / f"features/{directory}").glob("feat_*.npz")
        ):
            with path.open("rb") as handle:
                hashes[str(path)] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    inputs = sorted((args.research / "features/shards/val").glob("games_*.csv"))
    inputs.extend(
        [
            args.research / "configs/analysis_fast.cfg.tmpl",
            args.research
            / "c1/models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
            pathlib.Path(extract_rank_features.__file__),
            pathlib.Path(extract_rank_features.katago.__file__),
        ]
    )
    for path in inputs:
        with path.open("rb") as handle:
            hashes[str(path)] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    encoded = json.dumps(
        {"visits": 32, "decisions_per_role": 4, "hashes": hashes}, indent=2
    )
    path = args.output / "manifest.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError(
            "Source/cache configuration differs from existing search shards"
        )
    path.write_text(encoded)


def main() -> None:
    """Writes eight resumable original-validation feature shards."""
    args = common.parse_arguments(__doc__ or "", "search_regret_20261109", True)
    args.output.mkdir(parents=True, exist_ok=True)
    record_manifest(args)
    caches = (
        cache(args.research / "features/feat96/val"),
        cache(args.research / "features/feat96opp/val"),
    )
    shards = sorted((args.research / "features/shards/val").glob("games_*.csv"))
    if (
        len(shards) != 8
        or len(caches[0]) != 8000
        or set(caches[0]) != set(caches[1])
    ):
        raise ValueError(
            "Expected eight shards and matching8000game selection caches"
        )
    with probe.engine(args) as client:
        for shard in shards:
            destination = args.output / shard.name.replace(
                "games_", "feat_"
            ).replace(".csv", ".npz")
            if destination.exists():
                continue
            with shard.open(encoding="utf-8", newline="") as handle:
                games = list(csv.DictReader(handle))
            results = extract_games(client, games, caches)
            write_shard(destination, games, results)
            print(f"saved {destination.name}: {len(games)} games", flush=True)
    (args.output / "EXTRACTION_COMPLETE").write_text(
        "All original validation shards completed.\n"
    )


if __name__ == "__main__":
    main()
