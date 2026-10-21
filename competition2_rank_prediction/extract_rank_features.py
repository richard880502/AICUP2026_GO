"""Extracts per-move KataGo and multi-rank HumanSL policy features for rank prediction.

For each game, samples evenly spaced decisions of the labelled player's colour (or, with
--opponent, of the other colour) and records surprisal / log policy rank / entropy of the
played move under the normal KataGo policy and under every HumanSL rank profile.

Example:
    c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/train \
      --output features/feat/train --samples-per-game 32

Shards are resumable: a shard whose feat_*.npz exists is skipped. KATAGO_STDERR=<file> keeps the
engine's own log (e.g. CUDA out-of-memory) instead of discarding it.
"""

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import typing

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "c1"))

# The competition-1 package is found through the sys.path entry above.
from go_retrieval import katago  # noqa: E402  pylint: disable=wrong-import-position,import-error

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
# One HumanSL profile per competition level, near the middle of each level's range.
PROFILES = (
    "rank_11k", "rank_8k", "rank_5k", "rank_2k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d",
)
FEATURES_PER_MODEL = 3  # surprisal, log policy rank, policy entropy
BATCH_GAMES = 32


@dataclasses.dataclass(frozen=True)
class Settings:
    """Per-run extraction settings."""

    samples: int
    max_visits: int
    depth: int
    profiles: tuple[str, ...] = PROFILES


def parse_arguments() -> argparse.Namespace:
    """Parses extraction settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=int, default=1,
                        help="CUDA device index for this worker's KataGo")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3, help="batches kept in flight")
    parser.add_argument("--opponent", action="store_true",
                        help="extract the OPPONENT's decisions instead of the labelled player's")
    parser.add_argument("--profiles", default=",".join(PROFILES),
                        help="comma list of HumanSL profiles to query")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--limit-games", type=int, default=0,
                        help="debug: only the first N games of each shard")
    parser.add_argument("--samples-per-game", type=int, default=32)
    parser.add_argument("--max-visits", type=int, default=1)
    return parser.parse_args()


def make_queries(
    request_id: str,
    moves: list[katago.SgfMove],
    turns: list[int],
    max_visits: int,
    profiles: tuple[str, ...] = PROFILES,
) -> list[dict[str, object]]:
    """One normal-model query plus one query per HumanSL profile."""
    queries = [katago.make_analysis_query(request_id, moves, turns, max_visits)]
    for index, profile in enumerate(profiles):
        queries.append(
            katago.make_analysis_query(
                f"{request_id}:h{index}",
                moves,
                turns,
                max_visits,
                katago.AnalysisQueryConfig(False, profile),
            ),
        )
    return queries


def policy_features(
    moves: list[katago.SgfMove],
    turns: list[int],
    responses: list[dict[str, object]],
    key: str,
) -> np.ndarray:
    """Returns [len(turns), 3] features of the played move under one policy stream."""
    by_turn = {int(str(r["turnNumber"])): r for r in responses}
    return np.asarray(
        [katago.make_policy_features(moves[t], by_turn[t][key], key) for t in turns],
        dtype=np.float32,
    )


@dataclasses.dataclass
class _Tickets:
    """Book-keeping of the batches in flight."""

    remaining: dict[int, int] = dataclasses.field(default_factory=dict)
    ticket_of: dict[str, int] = dataclasses.field(default_factory=dict)
    ids_of: dict[int, list[str]] = dataclasses.field(default_factory=dict)
    responses: dict[int, dict[str, list[dict[str, object]]]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(dict))
    next_ticket: int = 0


class PipelinedKataGo:
    """KataGo analysis process that accepts new batches while earlier ones are still running.

    ``submit`` returns a ticket immediately; ``collect`` blocks until every response of that
    ticket has arrived. A reader thread keeps only the policy field each query needs, which
    keeps the parsed responses small.
    """

    def __init__(self, command: list[str], keep_all: bool = False):
        self.keep_all = keep_all
        log_path = os.environ.get("KATAGO_STDERR")
        # Both handles live as long as the engine; they are closed in close().
        self.stderr = (open(log_path, "a", encoding="utf-8")  # pylint: disable=consider-using-with
                       if log_path else None)
        self.process = subprocess.Popen(  # pylint: disable=consider-using-with
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr if self.stderr else subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.lock = threading.Condition()
        self.tickets = _Tickets()
        self.error: str | None = None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _keep(self, response: dict, request_id: str) -> dict:
        """The part of a response the features need."""
        if self.keep_all:  # value features need rootInfo and ownership too
            return {k: response[k] for k in ("turnNumber", "policy", "rootInfo", "ownership")
                    if k in response}
        key = "humanPolicy" if ":h" in request_id else "policy"
        return {"turnNumber": response["turnNumber"], key: response[key]}

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            response = json.loads(line)
            with self.lock:
                if "error" in response:
                    self.error = str(response["error"])
                    self.lock.notify_all()
                    return
                if response.get("isDuringSearch"):
                    continue
                request_id = str(response["id"])
                ticket = self.tickets.ticket_of[request_id]
                self.tickets.responses[ticket].setdefault(request_id, []).append(
                    self._keep(response, request_id))
                self.tickets.remaining[ticket] -= 1
                if self.tickets.remaining[ticket] == 0:
                    self.lock.notify_all()
        with self.lock:
            self.error = self.error or "KataGo exited"
            self.lock.notify_all()

    def submit(self, queries: list[dict[str, object]]) -> int:
        """Sends queries and returns a ticket for ``collect``."""
        assert self.process.stdin is not None
        with self.lock:
            ticket = self.tickets.next_ticket
            self.tickets.next_ticket += 1
            self.tickets.remaining[ticket] = sum(
                len(typing.cast(list, q["analyzeTurns"])) for q in queries)
            self.tickets.ids_of[ticket] = [str(query["id"]) for query in queries]
            for request_id in self.tickets.ids_of[ticket]:
                self.tickets.ticket_of[request_id] = ticket
        for query in queries:
            self.process.stdin.write(json.dumps(query) + "\n")
        self.process.stdin.flush()
        return ticket

    def collect(self, ticket: int) -> dict[str, list[dict[str, object]]]:
        """Blocks until every response of ``ticket`` has arrived and returns them by request id."""
        with self.lock:
            while self.tickets.remaining[ticket] > 0:
                if self.error:
                    raise RuntimeError(f"KataGo failed: {self.error}")
                self.lock.wait(timeout=5)
            del self.tickets.remaining[ticket]
            for request_id in self.tickets.ids_of.pop(ticket):
                del self.tickets.ticket_of[request_id]
            return self.tickets.responses.pop(ticket)

    def close(self) -> None:
        """Stops the engine."""
        self.process.terminate()
        self.process.wait(timeout=30)
        if self.stderr:
            self.stderr.close()


def prepare_batch(games: pd.DataFrame, start: int,
                  settings: Settings) -> tuple[list[tuple], list[dict[str, object]]]:
    """Parses one batch of games and builds its engine queries."""
    prepared = []
    queries: list[dict[str, object]] = []
    for offset, row in enumerate(games.iloc[start:start + BATCH_GAMES].itertuples()):
        moves = katago.parse_sgf_moves(row.sgf_content)
        turns = katago.sample_player_turns(moves, row.color, settings.samples)
        request_id = str(start + offset)
        prepared.append((start + offset, moves, turns, request_id))
        if turns:
            queries.extend(make_queries(request_id, moves, turns, settings.max_visits,
                                        settings.profiles))
    return prepared, queries


def fill_batch(responses: dict[str, list[dict[str, object]]], prepared: list, n_profiles: int,
               out: tuple[np.ndarray, np.ndarray]) -> None:
    """Writes one collected batch's features into out = (tokens, masks)."""
    tokens, masks = out
    for index, moves, turns, request_id in prepared:
        if not turns:
            continue
        columns = [policy_features(moves, turns, responses[request_id], "policy")]
        for h in range(n_profiles):
            columns.append(policy_features(moves, turns, responses[f"{request_id}:h{h}"],
                                           "humanPolicy"))
        tokens[index, :len(turns)] = np.stack(columns, axis=1)
        masks[index, :len(turns)] = True


def extract_games(client: PipelinedKataGo, games: pd.DataFrame, settings: Settings,
                  label: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns decision features [G, samples, 1+profiles, 3] and a validity mask [G, samples].

    Up to ``settings.depth`` batches are in flight, so the engine keeps working while Python
    turns the previous batch's responses into features.
    """
    shape = (len(games), settings.samples, 1 + len(settings.profiles), FEATURES_PER_MODEL)
    out = (np.zeros(shape, np.float32), np.zeros((len(games), settings.samples), np.bool_))
    started = time.time()
    inflight: collections.deque = collections.deque()

    def finish(entry: tuple) -> None:
        ticket, prepared = entry
        if ticket is not None:
            fill_batch(client.collect(ticket), prepared, len(settings.profiles), out)

    for start in range(0, len(games), BATCH_GAMES):
        prepared, queries = prepare_batch(games, start, settings)
        inflight.append((client.submit(queries) if queries else None, prepared))
        if len(inflight) >= settings.depth:
            finish(inflight.popleft())
    while inflight:
        finish(inflight.popleft())
    elapsed = time.time() - started
    print(f"[{label}] {len(games)} games  {elapsed:.0f}s  {len(games) / elapsed:.2f} games/s",
          flush=True)
    return out


def save_features(path: pathlib.Path, games: pd.DataFrame, tokens: np.ndarray,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  masks: np.ndarray, profiles: tuple[str, ...] = PROFILES) -> None:
    """Atomically writes one feature shard."""
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        tokens=tokens,
        masks=masks,
        game_key=np.asarray(games["game_key"], dtype="U"),
        player_id=np.asarray(games["player_id"], dtype="U"),
        rank=np.asarray(games["rank"].fillna(""), dtype="U"),
        profiles=np.asarray(("normal",) + profiles),
    )
    temporary.rename(path)


def start_engine(root: pathlib.Path, device: int, human: bool = True,
                 keep_all: bool = False) -> PipelinedKataGo:
    """Starts one KataGo process on ``device`` using the fast config template."""
    template = (root / "configs/analysis_fast.cfg.tmpl").read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False,
                                     encoding="utf-8") as config:
        config.write(template.replace("@DEVICE@", str(device)))
    models = root / "c1/models/katago"
    return PipelinedKataGo([
        str(root / "c1/scripts/run_katago.sh"), "analysis", "-config", config.name,
        "-model", str(models / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"),
        *(["-human-model", str(models / "b18c384nbt-humanv0.bin.gz")] if human else []),
    ], keep_all=keep_all)


def output_of(output_dir: pathlib.Path, shard: pathlib.Path) -> pathlib.Path:
    """games_XXXX.csv -> <output_dir>/feat_XXXX.npz"""
    return output_dir / shard.name.replace("games_", "feat_").replace(".csv", ".npz")


def main() -> None:
    """Processes this worker's share of the games_*.csv shards that have no feature file yet."""
    args = parse_arguments()
    settings = Settings(args.samples_per_game, args.max_visits, args.depth,
                        tuple(args.profiles.split(",")))
    print("profiles:", settings.profiles, "samples per game:", settings.samples, flush=True)
    root = pathlib.Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)

    pending = [shard for shard in sorted(args.shard_dir.glob("games_*.csv"))
               if not output_of(args.output, shard).exists()]
    if args.limit_shards:
        pending = pending[:args.limit_shards]
    pending = pending[args.worker::args.num_workers]
    print(f"worker {args.worker}/{args.num_workers}: {len(pending)} shard(s) to extract",
          flush=True)
    if not pending:
        return
    client = start_engine(root, args.device)
    try:
        for shard in pending:
            games = pd.read_csv(shard)
            if args.limit_games:
                games = games.head(args.limit_games)
            if args.opponent:
                # the other colour's decisions; labels and keys stay those of the labelled player
                games = games.assign(color=games["color"].map({"B": "W", "W": "B"}))
            tokens, masks = extract_games(client, games, settings, shard.name)
            save_features(output_of(args.output, shard), games, tokens, masks, settings.profiles)
    finally:
        client.close()


if __name__ == "__main__":
    main()
