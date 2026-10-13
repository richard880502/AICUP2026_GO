"""Extracts per-move KataGo and multi-rank HumanSL policy features for rank prediction.

For each game, samples evenly spaced decisions of the labelled player's colour and
records surprisal / log policy rank / entropy of the played move under the normal
KataGo policy and under every HumanSL rank profile.

Example:
    c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/train \
      --output features/feat/train --samples-per-game 32

Shards are resumable: a shard whose feat_*.npz exists is skipped.
"""

import argparse
import collections
import json
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "c1"))

from go_retrieval import katago  # noqa: E402  pylint: disable=wrong-import-position

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
# One HumanSL profile per competition level, near the middle of each level's range.
PROFILES = (
    "rank_11k", "rank_8k", "rank_5k", "rank_2k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d",
)
FEATURES_PER_MODEL = 3  # surprisal, log policy rank, policy entropy


def parse_arguments() -> argparse.Namespace:
    """Parses extraction settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=int, default=1, help="CUDA device index for this worker's KataGo")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3, help="batches kept in flight")
    parser.add_argument("--opponent", action="store_true", help="extract the OPPONENT's decisions instead of the labelled player's")
    parser.add_argument("--profiles", default=",".join(PROFILES), help="comma list of HumanSL profiles to query")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--limit-games", type=int, default=0, help="debug: only the first N games of each shard")
    parser.add_argument("--samples-per-game", type=int, default=32)
    parser.add_argument("--max-visits", type=int, default=1)
    return parser.parse_args()


def make_queries(
    request_id: str,
    moves: list[katago.SgfMove],
    turns: list[int],
    max_visits: int,
) -> list[dict[str, object]]:
    """One normal-model query plus one query per HumanSL profile."""
    queries = [katago.make_analysis_query(request_id, moves, turns, max_visits)]
    for index, profile in enumerate(PROFILES):
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
    by_turn = {int(r["turnNumber"]): r for r in responses}
    return np.asarray(
        [katago.make_policy_features(moves[t], by_turn[t][key], key) for t in turns],
        dtype=np.float32,
    )


class PipelinedKataGo:
    """KataGo analysis process that accepts new batches while earlier ones are still running.

    ``submit`` returns a ticket immediately; ``collect`` blocks until every response of that
    ticket has arrived. A reader thread keeps only the policy field each query needs, which
    keeps the parsed responses small.
    """

    def __init__(self, command: list[str], keep_all: bool = False):
        self.keep_all = keep_all
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.lock = threading.Condition()
        self.remaining: dict[int, int] = {}
        self.ticket_of: dict[str, int] = {}
        self.ids_of: dict[int, list[str]] = {}
        self.responses: dict[int, dict[str, list[dict[str, object]]]] = collections.defaultdict(dict)
        self.error: str | None = None
        self.next_ticket = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

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
                key = "humanPolicy" if ":h" in request_id else "policy"
                if self.keep_all:  # value features need rootInfo and ownership too
                    kept = {k: response[k] for k in ("turnNumber", "policy", "rootInfo", "ownership") if k in response}
                else:
                    kept = {"turnNumber": response["turnNumber"], key: response[key]}
                ticket = self.ticket_of[request_id]
                self.responses[ticket].setdefault(request_id, []).append(kept)
                self.remaining[ticket] -= 1
                if self.remaining[ticket] == 0:
                    self.lock.notify_all()
        with self.lock:
            self.error = self.error or "KataGo exited"
            self.lock.notify_all()

    def submit(self, queries: list[dict[str, object]]) -> int:
        """Sends queries and returns a ticket for ``collect``."""
        assert self.process.stdin is not None
        with self.lock:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.remaining[ticket] = sum(len(q["analyzeTurns"]) for q in queries)
            self.ids_of[ticket] = [str(query["id"]) for query in queries]
            for request_id in self.ids_of[ticket]:
                self.ticket_of[request_id] = ticket
        for query in queries:
            self.process.stdin.write(json.dumps(query) + "\n")
        self.process.stdin.flush()
        return ticket

    def collect(self, ticket: int) -> dict[str, list[dict[str, object]]]:
        with self.lock:
            while self.remaining[ticket] > 0:
                if self.error:
                    raise RuntimeError(f"KataGo failed: {self.error}")
                self.lock.wait(timeout=5)
            del self.remaining[ticket]
            for request_id in self.ids_of.pop(ticket):
                del self.ticket_of[request_id]
            return self.responses.pop(ticket)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=30)


def prepare_batch(games: pd.DataFrame, start: int, samples: int, max_visits: int):
    """Parses one batch of games and builds its engine queries."""
    prepared = []
    queries: list[dict[str, object]] = []
    for offset, row in enumerate(games.iloc[start:start + BATCH_GAMES].itertuples()):
        moves = katago.parse_sgf_moves(row.sgf_content)
        turns = katago.sample_player_turns(moves, row.color, samples)
        request_id = str(start + offset)
        prepared.append((start + offset, moves, turns, request_id))
        if turns:
            queries.extend(make_queries(request_id, moves, turns, max_visits))
    return prepared, queries


BATCH_GAMES = 32


def extract_games(
    client: PipelinedKataGo,
    games: pd.DataFrame,
    samples: int,
    depth: int,
    max_visits: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns decision features [G, samples, 1+profiles, 3] and a validity mask [G, samples].

    Up to ``depth`` batches are in flight, so the engine keeps working while Python turns the
    previous batch's responses into features.
    """
    n_models = 1 + len(PROFILES)
    tokens = np.zeros((len(games), samples, n_models, FEATURES_PER_MODEL), np.float32)
    masks = np.zeros((len(games), samples), np.bool_)
    started = time.time()
    inflight: collections.deque = collections.deque()

    def finish(ticket: int, prepared: list) -> None:
        responses = client.collect(ticket)
        for index, moves, turns, request_id in prepared:
            if not turns:
                continue
            columns = [policy_features(moves, turns, responses[request_id], "policy")]
            for h in range(len(PROFILES)):
                columns.append(
                    policy_features(moves, turns, responses[f"{request_id}:h{h}"], "humanPolicy"),
                )
            tokens[index, :len(turns)] = np.stack(columns, axis=1)
            masks[index, :len(turns)] = True

    for start in range(0, len(games), BATCH_GAMES):
        prepared, queries = prepare_batch(games, start, samples, max_visits)
        inflight.append((client.submit(queries) if queries else None, prepared))
        if len(inflight) >= depth:
            ticket, done = inflight.popleft()
            if ticket is not None:
                finish(ticket, done)
    while inflight:
        ticket, done = inflight.popleft()
        if ticket is not None:
            finish(ticket, done)
    elapsed = time.time() - started
    print(f"[{label}] {len(games)} games  {elapsed:.0f}s  {len(games) / elapsed:.2f} games/s", flush=True)
    return tokens, masks


def save_features(path: pathlib.Path, games: pd.DataFrame, tokens: np.ndarray, masks: np.ndarray) -> None:
    """Atomically writes one feature shard."""
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        tokens=tokens,
        masks=masks,
        game_key=np.asarray(games["game_key"], dtype="U"),
        player_id=np.asarray(games["player_id"], dtype="U"),
        rank=np.asarray(games["rank"].fillna(""), dtype="U"),
        profiles=np.asarray(("normal",) + PROFILES),
    )
    temporary.rename(path)


def start_engine(root: pathlib.Path, device: int, human: bool = True, keep_all: bool = False) -> PipelinedKataGo:
    """Starts one KataGo process on ``device`` using the fast config template."""
    template = (root / "configs/analysis_fast.cfg.tmpl").read_text()
    config = tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False)
    config.write(template.replace("@DEVICE@", str(device)))
    config.close()
    return PipelinedKataGo([
        str(root / "c1/scripts/run_katago.sh"), "analysis", "-config", config.name,
        "-model", str(root / "c1/models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"),
        *(["-human-model", str(root / "c1/models/katago/b18c384nbt-humanv0.bin.gz")] if human else []),
    ], keep_all=keep_all)


def main() -> None:
    """Processes this worker's share of the games_*.csv shards that have no feature file yet."""
    global PROFILES
    args = parse_arguments()
    PROFILES = tuple(args.profiles.split(","))
    print("profiles:", PROFILES, "samples per game:", args.samples_per_game, flush=True)
    root = pathlib.Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [
        shard for shard in sorted(args.shard_dir.glob("games_*.csv"))
        if not (args.output / shard.name.replace("games_", "feat_").replace(".csv", ".npz")).exists()
    ]
    if args.limit_shards:
        pending = pending[:args.limit_shards]
    pending = pending[args.worker::args.num_workers]
    print(f"worker {args.worker}/{args.num_workers}: {len(pending)} shard(s) to extract", flush=True)
    if not pending:
        return
    client = start_engine(root, args.device)
    try:
        for shard in pending:
            games = pd.read_csv(shard)
            if args.limit_games:
                games = games.head(args.limit_games)
            tokens, masks = extract_games(
                client, games, args.samples_per_game, args.depth, args.max_visits, shard.name,
            )
            save_features(
                args.output / shard.name.replace("games_", "feat_").replace(".csv", ".npz"),
                games, tokens, masks,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
