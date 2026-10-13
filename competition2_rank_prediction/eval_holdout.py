"""Scores the untouched internal holdout test: the tutorial's method vs my pipeline.

Needs the per-game probabilities written by `predict_test.py --dump` on the holdout questions and the
holdout-excluded retrains (suffix _h). Fusion weights and the aggregator come from the *validation* players only,
so nothing here was ever fitted on a holdout player. Paired bootstrap over questions gives the intervals.

Example:
    c1/.venv/bin/python predict_test.py --tests-dir features/holdout/tests --feat features/feat/holdout \
      --value features/feat_value/holdout --moves features/moves/holdout_moves.npz --expect 400 --suffix _h \
      --dump features/holdout/probs.npz --out features/holdout/submission.csv
    c1/.venv/bin/python eval_holdout.py
"""
import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

import blend_eval
import rank_lib as lib


def question_score(pred, truth):
    gap = abs(pred - truth)
    return 1.0 if gap == 0 else (lib.ADJACENT if gap == 1 else 0.0)


def boot(a, b, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, default=lib.ROOT / "features/holdout/probs.npz")
    ap.add_argument("--labels", type=pathlib.Path, default=lib.ROOT / "features/holdout/tests/labels.csv")
    ap.add_argument("--suffix", default="_h")
    ap.add_argument("--results", type=pathlib.Path, default=lib.ROOT / "results")
    args = ap.parse_args()
    sfx = args.suffix
    models = [f"transformer_v{sfx}", f"move{sfx}", f"resnet5{sfx}", f"resnet8c{sfx}"]
    z = np.load(args.dump, allow_pickle=True)
    probs = {m: np.clip(z[m], 1e-6, 1.0) for m in models}
    by_q = {}
    for i, q in enumerate(z["player_id"]):
        by_q.setdefault(str(q), []).append(i)
    labels = pd.read_csv(args.labels)
    qids = labels["question_id"].tolist()
    truth = np.array([lib.LEVELS.index(r) for r in labels["rank"]])
    ks = np.array([len(by_q[q]) for q in qids])

    # fusion weights + aggregator fitted on the validation players only
    val_logs, y = [], None
    for m in models:
        v = np.load(args.results / f"val_probs_{m}.npz", allow_pickle=True)
        val_logs.append(np.log(np.clip(v["transformer" if m.startswith("transformer_v") else m], 1e-6, 1.0)))
        y = v["y"]
    weights = blend_eval.fit_weights(np.stack(val_logs), y)
    print("fusion weights:", dict(zip(models, weights.round(3))))
    test_logs = np.stack([np.log(probs[m]) for m in models])
    fused = blend_eval.fuse(test_logs, weights)
    agg = pickle.load(open(args.results / f"aggregator{sfx}.pkl", "rb"))
    assert agg["models"] == models, (agg["models"], models)

    def decode(p):
        return int(lib.decode(p[None])[0])

    methods = {
        "resnet5 snapshot avg (official recipe), argmax": lambda idx: int(probs[models[2]][idx].mean(0).argmax()),
        "resnet5 snapshot avg + expected decode": lambda idx: decode(probs[models[2]][idx].mean(0)),
        "resnet8c (cosine), argmax": lambda idx: int(probs[models[3]][idx].mean(0).argmax()),
        "resnet8c (cosine), expected decode": lambda idx: decode(probs[models[3]][idx].mean(0)),
        "transformer_v, expected decode": lambda idx: decode(probs[models[0]][idx].mean(0)),
        "fusion (3-4 models), logmean, expected": lambda idx: decode(lib.pool(fused[idx], "logmean")),
        "fusion + learned aggregator": lambda idx: decode(agg["clf"].predict_proba(agg["scaler"].transform(
            np.concatenate([test_logs[:, idx].mean(1).reshape(-1), [1 / np.sqrt(len(idx)), np.log(len(idx))]])[None]))[0]),
    }
    scores = {}
    print(f"{len(qids)} holdout questions; k mean {ks.mean():.1f}")
    print(f"{'method':44s} {'score':>7s} {'+-SE':>6s} {'k5-9':>6s} {'k10-14':>7s} {'k15-20':>7s}")
    for name, fn in methods.items():
        sc = np.array([question_score(fn(by_q[q]), t) for q, t in zip(qids, truth)])
        scores[name] = sc
        print(f"{name:44s} {sc.mean():7.4f} {sc.std(ddof=1) / np.sqrt(len(sc)):6.3f} {sc[ks < 10].mean():6.3f} "
              f"{sc[(ks >= 10) & (ks < 15)].mean():7.3f} {sc[ks >= 15].mean():7.3f}")
    base = scores["resnet5 snapshot avg (official recipe), argmax"]
    print("\npaired bootstrap over questions, minus the tutorial method (mean, 95% interval):")
    for name, sc in scores.items():
        if sc is not base:
            m, lo, hi = boot(sc, base)
            print(f"  {name:44s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
