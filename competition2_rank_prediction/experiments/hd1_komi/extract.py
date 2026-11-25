"""Extracts paired64visit final-position values for validation HD1games only.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
HD is read from updated authorized training rows, not old metadata-free shards.
"""

import csv
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.search_regret import probe

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error

HD = re.compile(r"HD\[(\d+)\]")
LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def handicap(row: dict, original: dict) -> int:
    """Requires unchanged actor metadata and an explicit updated HD header."""
    for field in ("player_id", "rank", "color"):
        if row[field] != original[field]:
            raise ValueError(
                f"Original/updated actor metadata differs: {field}"
            )
    match = HD.search(row["sgf_content"])
    if match is None:
        raise ValueError("Updated source lacks required HD header")
    return int(match.group(1))


def selected_games(research) -> tuple[list, dict]:
    """Checks every original game key and preserves exact moves/actor labels."""
    cached = {}
    for path in sorted((research / "features/shards/val").glob("games_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["game_key"] in cached:
                    raise ValueError("Duplicate original validation game key")
                cached[row["game_key"]] = row
    if len(cached) != 8000:
        raise ValueError("Expected8000original validation games")
    selected = []
    seen = set()
    hashes = {}
    for rank in LEVELS:
        path = research / "data/training" / f"train_{rank}.csv"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = rank + "_" + row["game_id"]
                if key not in cached:
                    continue
                if key in seen:
                    raise ValueError("Repeated updated-source validation game")
                seen.add(key)
                if handicap(row, cached[key]) != 1:
                    continue
                if extract_rank_features.parse_moves(
                    row["sgf_content"]
                ) != extract_rank_features.parse_moves(
                    cached[key]["sgf_content"]
                ):
                    raise ValueError(f"Updated-source moves differ: {key}")
                selected.append({"game_key": key, **row})
    if seen != set(cached) or len(selected) != 1542:
        raise ValueError("HD lookup or fixed1542-game cohort differs")
    return selected, hashes


def queries(games: list, start: int) -> list:
    """Builds paired final-root queries differing only in komi and request ID."""
    result = []
    for offset, game in enumerate(games):
        moves = extract_rank_features.parse_moves(game["sgf_content"])
        if not moves:
            raise ValueError("HD1selected game has no moves")
        for condition, komi in enumerate((6.5, 0.5)):
            query = extract_rank_features.katago.make_analysis_query(
                f"{start+offset}:{condition}", moves, [len(moves)], 64
            )
            query.update(komi=komi, includePolicy=False, includeOwnership=False)
            result.append(query)
    return result


def root_values(response: dict) -> np.ndarray:
    """Checks actual visits and retains Black-perspective values for auditing."""
    root = response["rootInfo"]
    result = np.asarray(
        [root["visits"], root["winrate"], root["scoreLead"]], dtype=float
    )
    if (
        not np.isfinite(result).all()
        or result[0] < 64
        or not 0 <= result[1] <= 1
    ):
        raise ValueError("Missing/invalid64visit HD1analysis root")
    return result


def main() -> None:
    """Writes paired roots without modifying the original final-result cache."""
    args = common.parse_arguments(__doc__ or "", "hd1_komi_20261109", True)
    if args.output.exists():
        raise FileExistsError(f"Refusing existing HD1experiment: {args.output}")
    games, hashes = selected_games(args.research)
    values = np.zeros((len(games), 2, 3))
    args.output.mkdir(parents=True)
    with probe.engine(args) as client:
        for start in range(0, len(games), 64):
            batch = games[start : start + 64]
            responses = client.analyze(queries(batch, start))
            for offset in range(len(batch)):
                for condition in range(2):
                    rows = responses[f"{start+offset}:{condition}"]
                    if len(rows) != 1:
                        raise ValueError(
                            "Expected exactly one final-root response"
                        )
                    values[start + offset, condition] = root_values(rows[0])
            print(
                f"Paired HD1roots {start+len(batch)}/{len(games)}", flush=True
            )
    np.savez_compressed(
        args.output / "paired_roots.npz",
        roots=values,
        game_key=np.asarray([game["game_key"] for game in games]),
        player=np.asarray([game["player_id"] for game in games]),
        color=np.asarray([game["color"] for game in games]),
        komi=np.asarray([6.5, 0.5]),
    )
    hashes[str(__file__)] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(hashes, indent=2))
    (args.output / "EXTRACTION_COMPLETE").touch()


if __name__ == "__main__":
    main()
