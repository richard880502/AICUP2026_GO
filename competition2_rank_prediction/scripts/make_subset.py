"""Builds player-disjoint train / validation game shards for feature extraction.

validation cohort: --val-players-per-rank players per rank, --val-games games each
(so 5-20 game questions can be simulated); train: --train-games games for every other player.
Shards are shuffled across ranks so any prefix of shards is rank-balanced.

Example:
    c1/.venv/bin/python scripts/make_subset.py --val-players-per-rank 40 --train-games 2
"""
import argparse, pathlib
import numpy as np, pandas as pd

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--val-players-per-rank", type=int, default=40)
ap.add_argument("--val-games", type=int, default=20)
ap.add_argument("--train-games", type=int, default=2)
ap.add_argument("--min-val-games", type=int, default=20, help="val players need at least this many games")
ap.add_argument("--shard-size", type=int, default=1000)
ap.add_argument("--seed", type=int, default=20261003)
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/shards")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

parts = {"train": [], "val": []}
for rank in LEVELS:
    d = pd.read_csv(ROOT / f"data/training/train_{rank}.csv")
    d["game_key"] = rank + "_" + d["game_id"]
    counts = d.groupby("player_id").size()
    eligible = counts[counts >= a.min_val_games].index.to_numpy()
    val_players = set(rng.choice(eligible, size=min(a.val_players_per_rank, len(eligible)), replace=False))
    d = d.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    is_val = d["player_id"].isin(val_players)
    parts["val"].append(d[is_val].groupby("player_id").head(a.val_games))
    parts["train"].append(d[~is_val].groupby("player_id").head(a.train_games))
    print(rank, "val players", len(val_players), "train players", d[~is_val].player_id.nunique(), flush=True)

for split, frames in parts.items():
    df = pd.concat(frames, ignore_index=True)[["game_key", "player_id", "rank", "color", "sgf_content"]]
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    out = a.out / split
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(df), a.shard_size):
        df.iloc[i:i + a.shard_size].to_csv(out / f"games_{i // a.shard_size:04d}.csv", index=False)
    print(split, len(df), "games in", -(-len(df) // a.shard_size), "shards")
