"""Builds an untouched internal 'official-format' test set (the user's idea: our own public/private).

Picks --players players that (a) have at least 20 games, (b) are not in the validation cohort, and writes
  features/holdout/players.txt                 players that every retrained model must exclude
  features/holdout/tests/rank_prediction_test_{public,private}.csv  + labels.csv   (official layout)
Each player gives one question of k ~ U{5..20} games, like the official 5-20 game questions.
"""
import argparse, pathlib
import numpy as np, pandas as pd

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--players", type=int, default=400)
ap.add_argument("--seed", type=int, default=20261004)
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/holdout")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

val_players = set()
for p in sorted((ROOT / "features/shards/val").glob("games_*.csv")):
    val_players |= set(pd.read_csv(p, usecols=["player_id"])["player_id"])
per_rank = a.players // len(LEVELS)  # rank-balanced like the official sets
chosen, frames = [], []
for rank in LEVELS:
    d = pd.read_csv(ROOT / f"data/training/train_{rank}.csv")
    counts = d.groupby("player_id").size()
    eligible = np.array([p for p in counts[counts >= 20].index if p not in val_players], dtype=object)
    pick = rng.choice(eligible, size=per_rank, replace=False)
    chosen += list(pick)
    frames.append(d[d["player_id"].isin(set(pick))])
    print(rank, "eligible", len(eligible), "picked", len(pick), flush=True)
games = pd.concat(frames, ignore_index=True)
order = rng.permutation(len(chosen))
chosen = [chosen[i] for i in order]
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "players.txt").write_text("\n".join(chosen) + "\n")
(a.out / "tests").mkdir(exist_ok=True)
labels, rows = [], {"public": [], "private": []}
for q, pid in enumerate(chosen):
    split = "public" if q < len(chosen) // 2 else "private"
    g = games[games["player_id"] == pid].sample(frac=1, random_state=int(rng.integers(1 << 30))).head(int(rng.integers(5, 21)))
    qid = f"{'pub' if split == 'public' else 'priv'}_q_{len(rows[split]) + 1:04d}"
    row = {"question_id": qid, "num_games": len(g)}
    for i in range(20):
        row[f"sgf_{i + 1}"] = g.iloc[i]["sgf_content"] if i < len(g) else ""
    for i in range(20):
        row[f"color_{i + 1}"] = g.iloc[i]["color"] if i < len(g) else ""
    rows[split].append(row); labels.append((qid, g.iloc[0]["rank"]))
for split, r in rows.items():
    pd.DataFrame(r).to_csv(a.out / f"tests/rank_prediction_test_{split}.csv", index=False)
pd.DataFrame(labels, columns=["question_id", "rank"]).to_csv(a.out / "tests/labels.csv", index=False)
print("wrote", len(labels), "questions;", sum(len(r) for r in rows.values()), "total; games:",
      sum(int(x["num_games"]) for r in rows.values() for x in r))
