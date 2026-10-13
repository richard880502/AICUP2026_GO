"""Builds a rehearsal 'official-format' test set from validation players (labels kept aside).

Writes rank_prediction_test_{public,private}.csv (question_id, num_games, sgf_i, color_i) plus labels.csv, so the
whole inference pipeline can be exercised before the real test set is released.
"""
import argparse, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--questions", type=int, default=40, help="per split")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
games = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "features/shards/val").glob("games_*.csv"))], ignore_index=True)
players = rng.permutation(np.asarray(games["player_id"].unique(), dtype=object))
a.out.mkdir(parents=True, exist_ok=True)
labels, cursor = [], 0
for split in ("public", "private"):
    rows = []
    for q in range(a.questions):
        pid = players[cursor]; cursor += 1
        g = games[games["player_id"] == pid].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        g = g.head(int(rng.integers(5, 21)))
        qid = f"{'pub' if split == 'public' else 'priv'}_q_{q + 1:04d}"
        row = {"question_id": qid, "num_games": len(g)}
        for i in range(20):
            row[f"sgf_{i + 1}"] = g.iloc[i]["sgf_content"] if i < len(g) else ""
        for i in range(20):
            row[f"color_{i + 1}"] = g.iloc[i]["color"] if i < len(g) else ""
        rows.append(row); labels.append((qid, g.iloc[0]["rank"]))
    pd.DataFrame(rows).to_csv(a.out / f"rank_prediction_test_{split}.csv", index=False)
pd.DataFrame(labels, columns=["question_id", "rank"]).to_csv(a.out / "labels.csv", index=False)
print("wrote", 2 * a.questions, "questions to", a.out)
