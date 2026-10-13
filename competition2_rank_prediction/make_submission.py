"""Predicts the official public+private questions and writes submission_rank.csv (800 rows).

Steps after the test CSVs are released:
    c1/.venv/bin/python scripts/test_to_games.py --tests-dir <dir with rank_prediction_test_*.csv>
    c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/test --output features/feat/test --samples-per-game 32
    c1/.venv/bin/python make_submission.py --tests-dir <same dir>
"""
import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

import rank_lib as lib

ap = argparse.ArgumentParser()
ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
ap.add_argument("--features", type=pathlib.Path, default=lib.ROOT / "features/feat/test")
ap.add_argument("--models", type=pathlib.Path, default=lib.ROOT / "results/final_models.pkl")
ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "submission_rank.csv")
ap.add_argument("--pool", default="logmean", choices=("mean", "logmean"))
a = ap.parse_args()

feats = lib.load_features(a.features)
models = pickle.load(open(a.models, "rb"))
x = lib.game_features(feats["tokens"], feats["masks"])
prob = np.mean([m.predict_proba(x) for m in models.values()], axis=0)

question_ids = pd.concat([pd.read_csv(a.tests_dir / f"rank_prediction_test_{s}.csv", usecols=["question_id"])
                          for s in ("public", "private")])["question_id"].tolist()
rows = []
for qid in question_ids:
    idx = np.flatnonzero(feats["player_id"] == qid)
    if len(idx) == 0:  # no parsable game: fall back to the middle class rather than dropping the question
        rows.append((qid, lib.LEVELS[4]))
        continue
    rows.append((qid, lib.LEVELS[int(lib.decode(lib.pool(prob[idx], a.pool)[None])[0])]))
sub = pd.DataFrame(rows, columns=["question_id", "pred_rank"])
assert len(sub) == 800 and sub.question_id.is_unique, f"unexpected submission size {len(sub)}"
sub.to_csv(a.out, index=False)
print("wrote", a.out, sub.pred_rank.value_counts().to_dict())
