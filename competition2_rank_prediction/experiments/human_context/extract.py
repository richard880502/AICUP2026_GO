# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Extracts HumanSL context and actual-move transitions with normal controls.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.human_context.extract
"""

import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

from competition2_rank_prediction.experiments.human_context import probe
from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error

NORMAL_FIELDS = (
    "rawWinrate",
    "rawScoreSelfplay",
    "rawScoreSelfplayStdev",
    "rawStWrError",
    "rawStScoreError",
)
SAMPLES = 16


def root_values(
    root: object, fields: tuple[str, ...], color: str
) -> np.ndarray:
    """Converts BLACK-perspective engine values to mover-perspective features."""
    if color not in ("B", "W"):
        raise ValueError(f"Unexpected labelled player color: {color!r}")
    if not isinstance(root, dict):
        raise ValueError("Engine rootInfo is not a dictionary")
    if len(fields) != 5:
        raise ValueError("Context conversion requires exactly five fields")
    values = np.asarray(
        [float(root[name]) for name in fields], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite context value from engine")
    if color == "W":
        values[0] = 1 - values[0]
        values[1] = -values[1]
    values[[1, 2, 4]] /= 20
    return values.astype(np.float32)


def prepare_game(
    game: dict[str, str], request_id: str
) -> tuple[list[int], list[dict[str, object]]]:
    """Builds three rank-conditioned requests over pre/post actual-move turns."""
    moves = extract_rank_features.parse_moves(game["sgf_content"])
    turns = extract_rank_features.katago.sample_player_turns(
        moves, game["color"], SAMPLES
    )
    settings = extract_rank_features.Settings(SAMPLES, 1, 1, probe.PROFILES)
    queries = extract_rank_features.make_queries(
        request_id, moves, turns, settings
    )[1:]
    for query in queries:
        query["analyzeTurns"] = sorted(
            set(turns) | {turn + 1 for turn in turns}
        )
        query["includeOwnership"] = False
    return turns, queries


def game_tokens(
    game: dict[str, str],
    turns: list[int],
    responses: dict[str, list[dict[str, object]]],
    request_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns sixteen normal/human transition tokens with explicit masks."""
    tokens = np.zeros((SAMPLES, 4, 10), dtype=np.float32)
    masks = np.zeros((SAMPLES, 4), dtype=bool)
    for profile in range(3):
        lookup = {
            int(str(row["turnNumber"])): row.get("rootInfo", {})
            for row in responses.get(f"{request_id}:h{profile}", [])
        }
        streams = [(profile + 1, probe.FIELDS)]
        if profile == 0:
            streams.append((0, NORMAL_FIELDS))
        for sample, turn in enumerate(turns):
            if turn not in lookup or turn + 1 not in lookup:
                continue
            for column, fields in streams:
                before = root_values(lookup[turn], fields, game["color"])
                after = root_values(lookup[turn + 1], fields, game["color"])
                tokens[sample, column] = np.concatenate(
                    (before, after - before)
                )
                masks[sample, column] = True
    return tokens, masks


def extract_shard(
    client, games: list[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Batches games so parsing work does not require keeping a whole shard's JSON."""
    tokens, masks = [], []
    for start in range(0, len(games), 16):
        prepared = []
        queries = []
        for index in range(start, min(start + 16, len(games))):
            turns, requests = prepare_game(games[index], str(index))
            prepared.append((index, turns))
            if turns:
                queries.extend(requests)
        responses = client.collect(client.submit(queries)) if queries else {}
        for index, turns in prepared:
            values, valid = game_tokens(
                games[index], turns, responses, str(index)
            )
            tokens.append(values)
            masks.append(valid)
        if (start + 16) % 128 == 0:
            print(
                f"extracted {min(start + 16, len(games))}/{len(games)} games in shard",
                flush=True,
            )
    return np.stack(tokens), np.stack(masks)


def main() -> None:
    """Writes resumable independent feature shards with game-key metadata."""
    args = common.parse_arguments(__doc__ or "", "human_context_20261109", True)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "samples": SAMPLES,
        "profiles": probe.PROFILES,
        "fields": probe.FIELDS,
        "extractor_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
    }
    manifest = args.output / "extraction_settings.json"
    encoded = json.dumps(settings, indent=2)
    if manifest.exists() and manifest.read_text() != encoded:
        raise ValueError("Extraction settings differ from existing shards")
    manifest.write_text(encoded)
    client = probe.start_client(args)
    try:
        for shard in sorted(
            (args.research / "features/shards/val").glob("games_*.csv")
        ):
            destination = args.output / shard.name.replace(
                "games_", "feat_"
            ).replace(".csv", ".npz")
            if destination.exists():
                continue
            with shard.open(encoding="utf-8", newline="") as handle:
                games = list(csv.DictReader(handle))
            tokens, masks = extract_shard(client, games)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                tokens=tokens,
                masks=masks,
                game_key=np.asarray([game["game_key"] for game in games]),
                player_id=np.asarray([game["player_id"] for game in games]),
                rank=np.asarray([game["rank"] for game in games]),
            )
            temporary.rename(destination)
            print(f"saved {destination.name}: {len(games)} games", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
