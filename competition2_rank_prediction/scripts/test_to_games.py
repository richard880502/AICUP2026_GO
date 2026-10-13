"""Explodes official rank_prediction_test_{public,private}.csv into shard CSVs.

Each question becomes one pseudo-player (player_id = question_id) so the same extractor and
the same question-level pooling code apply. Example:
    c1/.venv/bin/python scripts/test_to_games.py --tests-dir /path/to/tests
"""
import argparse, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
ap.add_argument("--shard-size", type=int, default=1000)
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/shards/test")
a = ap.parse_args()

rows = []
for split in ("public", "private"):
    t = pd.read_csv(a.tests_dir / f"rank_prediction_test_{split}.csv")
    for r in t.itertuples(index=False):
        r = r._asdict()
        for i in range(1, int(r["num_games"]) + 1):
            sgf = r.get(f"sgf_{i}")
            if isinstance(sgf, str) and sgf.strip():
                rows.append({"game_key": f"{r['question_id']}#{i}", "player_id": r["question_id"],
                             "rank": "", "color": r[f"color_{i}"], "sgf_content": sgf})
df = pd.DataFrame(rows)
a.out.mkdir(parents=True, exist_ok=True)
for i in range(0, len(df), a.shard_size):
    df.iloc[i:i + a.shard_size].to_csv(a.out / f"games_{i // a.shard_size:04d}.csv", index=False)
print(len(df), "games,", df.player_id.nunique(), "questions")
