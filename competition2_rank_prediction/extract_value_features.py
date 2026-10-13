"""Value / ownership decision features (competition 1's transition tokens) for the same games.

For each sampled decision of the labelled player: KataGo's policy surprisal / rank / entropy, win rate and
score lead before and after the move (their drop = how much the move cost), ownership change geometry.
Uses normal-model queries only (about 5x cheaper than the multi-profile HumanSL extraction).

Example:
    c1/.venv/bin/python extract_value_features.py --shard-dir features/shards/val --output features/feat_value/val --device 0
"""
import argparse
import collections
import pathlib
import time

import numpy as np
import pandas as pd

import extract_rank_features as base
from extract_rank_features import katago  # competition 1's module (path set up there)

BATCH_GAMES = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--samples-per-game", type=int, default=32)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--limit-games", type=int, default=0)
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [s for s in sorted(args.shard_dir.glob("games_*.csv"))
               if not (args.output / s.name.replace("games_", "feat_").replace(".csv", ".npz")).exists()]
    if args.limit_shards:
        pending = pending[:args.limit_shards]
    pending = pending[args.worker::args.num_workers]
    print(f"worker {args.worker}/{args.num_workers}: {len(pending)} shard(s)", flush=True)
    if not pending:
        return
    names = katago.decision_feature_names(True)
    n_features = len(names)
    client = base.start_engine(root, args.device, human=False, keep_all=True)
    try:
        for shard in pending:
            games = pd.read_csv(shard)
            if args.limit_games:
                games = games.head(args.limit_games)
            samples = args.samples_per_game
            tokens = np.zeros((len(games), samples, n_features), np.float32)
            masks = np.zeros((len(games), samples), np.bool_)
            inflight: collections.deque = collections.deque()
            started = time.time()

            def finish(ticket, prepared):
                responses = client.collect(ticket)
                for index, moves, turns, color, request_id in prepared:
                    if not turns:
                        continue
                    built = katago.build_decision_tokens(
                        moves, color, turns, katago.DecisionResponses(responses[request_id], {}), True)
                    tokens[index, :len(built)] = built
                    masks[index, :len(built)] = True

            for start in range(0, len(games), BATCH_GAMES):
                prepared, queries = [], []
                for offset, row in enumerate(games.iloc[start:start + BATCH_GAMES].itertuples()):
                    moves = katago.parse_sgf_moves(row.sgf_content)
                    turns = katago.sample_player_turns(moves, row.color, samples)
                    rid = str(start + offset)
                    prepared.append((start + offset, moves, turns, row.color, rid))
                    if turns:
                        queries.append(katago.make_analysis_query(rid, moves, turns, 1, katago.AnalysisQueryConfig(True)))
                inflight.append((client.submit(queries) if queries else None, prepared))
                if len(inflight) >= args.depth:
                    ticket, done = inflight.popleft()
                    if ticket is not None:
                        finish(ticket, done)
            while inflight:
                ticket, done = inflight.popleft()
                if ticket is not None:
                    finish(ticket, done)
            elapsed = time.time() - started
            print(f"[{shard.name}] {len(games)} games  {elapsed:.0f}s  {len(games) / elapsed:.2f} games/s", flush=True)
            base.save_features(args.output / shard.name.replace("games_", "feat_").replace(".csv", ".npz"), games, tokens, masks)
            np.save(args.output / "feature_names.npy", np.asarray(names))
    finally:
        client.close()


if __name__ == "__main__":
    main()
