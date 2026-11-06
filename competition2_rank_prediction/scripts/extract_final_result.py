"""Game-result proxy for the shift residual.

The SGF records no result, handicap or komi, so KataGo evaluates the FINAL position of every game
(normal model, 64 visits, Japanese rules, komi 6.5) from the labelled player's side, plus a few
SGF-derived facts.

Per game (features/final_result/<split>/feat_XXXX.npz, arrays game_key/player_id/rank/values/names):
  final_winrate        KataGo win rate of the labelled player in the final position
  final_score_lead     KataGo score lead of the labelled player in the final position (points)
  n_moves              number of moves (incl. passes)
  last_move_by_player  1 if the labelled player made the last move
  ended_two_passes     1 if the game ends with two passes (counted game rather than resignation)
  leading_black_stones number of consecutive black moves at the start (> 1 = handicap stones)
Example: c1/.venv/bin/python scripts/extract_final_result.py --shard-dir features/shards/val \
    --output features/final_result/val --device 0
"""
import argparse
import collections
import pathlib
import sys
import time

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# extract_rank_features lives in the project root, which is put on sys.path just above.
import extract_rank_features as base  # noqa: E402  pylint: disable=wrong-import-position,import-error
from extract_rank_features import katago  # noqa: E402  pylint: disable=wrong-import-position,import-error

NAMES = ["final_winrate", "final_score_lead", "n_moves", "last_move_by_player", "ended_two_passes",
         "leading_black_stones"]
BATCH = 64


def sgf_facts(moves: list, color: str) -> list[float]:
    """n_moves, last_move_by_player, ended_two_passes, leading_black_stones."""
    n = len(moves)
    lead = 0
    for move in moves:
        if move.color != "B":
            break
        lead += 1
    two_pass = float(n >= 2 and moves[-1].x is None and moves[-2].x is None)
    return [float(n), float(n > 0 and moves[-1].color == color), two_pass, float(lead)]


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--shard-dir", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--visits", type=int, default=64)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true",
                    help="opt-in: fixed board symmetry (nnRandomize = false)")
    return ap.parse_args()


def fill_results(responses: dict, prepared: list, values: np.ndarray) -> None:
    """Writes the final-position win rate / score lead (reported for Black) from the labelled
    player's side into values[:, :2]."""
    for i, rid, color in prepared:
        response = responses.get(rid)
        if not response:
            continue
        root = response[-1]["rootInfo"]
        sign = 1.0 if color == "B" else -1.0
        values[i, 0] = root["winrate"] if color == "B" else 1.0 - root["winrate"]
        values[i, 1] = sign * root["scoreLead"]


def shard_values(client: base.PipelinedKataGo, games: pd.DataFrame, visits: int) -> np.ndarray:
    """[G, len(NAMES)] features of one shard (NaN where a game has no move)."""
    values = np.full((len(games), len(NAMES)), np.nan, np.float32)
    inflight: collections.deque = collections.deque()

    def finish(entry: tuple) -> None:
        ticket, prepared = entry
        if ticket is not None:
            fill_results(client.collect(ticket), prepared, values)

    for start in range(0, len(games), BATCH):
        prepared, queries = [], []
        for offset, row in enumerate(games.iloc[start:start + BATCH].itertuples()):
            i = start + offset
            moves = base.parse_moves(row.sgf_content)
            values[i, 2:] = sgf_facts(moves, row.color)
            if moves:
                queries.append(katago.make_analysis_query(str(i), moves, [len(moves)], visits))
                prepared.append((i, str(i), row.color))
        inflight.append((client.submit(queries) if queries else None, prepared))
        if len(inflight) >= 3:
            finish(inflight.popleft())
    while inflight:
        finish(inflight.popleft())
    return values


def main() -> None:
    """Processes this worker's share of the shards that have no output yet."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def output_of(shard: pathlib.Path) -> pathlib.Path:
        return args.output / shard.name.replace("games_", "feat_").replace(".csv", ".npz")

    pending = [s for s in sorted(args.shard_dir.glob("games_*.csv"))
               if not output_of(s).exists()][args.worker::args.num_workers]
    print(f"{len(pending)} shard(s)", flush=True)
    if not pending:
        return
    client = base.start_engine(ROOT, args.device, human=False, keep_all=True,
                               deterministic=args.deterministic)
    try:
        for shard in pending:
            games = pd.read_csv(shard)
            started = time.time()
            values = shard_values(client, games, args.visits)
            np.savez(output_of(shard), game_key=np.asarray(games["game_key"], dtype="U"),
                     player_id=np.asarray(games["player_id"], dtype="U"),
                     rank=np.asarray(games["rank"].fillna(""), dtype="U"), values=values,
                     names=np.asarray(NAMES))
            print(f"[{shard.name}] {len(games)} games {time.time() - started:.0f}s", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
