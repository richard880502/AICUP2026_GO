"""Fuses per-game probabilities from several models and scores 5-20 game questions honestly.

Fusion = softmax(sum_m w_m * log p_m): a weighted geometric mean with weights fitted by minimising
per-game NLL. Validation players are split into two halves; weights are fitted on one half and the
question score is measured on the other (then swapped), so no validation label picks its own weights.

Example:
    c1/.venv/bin/python blend_eval.py --models gbm:blend transformer:transformer move:move resnet:resnet
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import minimize

import rank_lib as lib
from train_eval import simulate_questions


def load(models, results):
    probs, ref = {}, None
    for spec in models:
        tag, key = spec.split(":")
        z = np.load(results / f"val_probs_{tag}.npz", allow_pickle=True)
        if ref is None:
            ref = {k: z[k] for k in ("y", "player", "game_key")}
        assert (z["game_key"] == ref["game_key"]).all(), f"{tag}: game order differs"
        probs[tag] = np.clip(z[key], 1e-6, 1.0)
    return probs, ref


def fit_weights(logs, y):
    """logs: [M, N, 10] log-probs. Returns non-negative weights minimising mean NLL of the fused softmax."""
    M = logs.shape[0]

    def nll(w):
        z = np.tensordot(np.abs(w), logs, 1)
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())

    best = minimize(nll, np.full(M, 1.0 / M), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-5})
    return np.abs(best.x)


def fuse(logs, w):
    z = np.tensordot(w, logs, 1)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tag:key pairs, e.g. transformer:transformer")
    ap.add_argument("--results", type=pathlib.Path, default=lib.ROOT / "results")
    ap.add_argument("--questions-per-player", type=int, default=60)
    ap.add_argument("--out", default="blend_report.json")
    args = ap.parse_args()
    probs, ref = load(args.models, args.results)
    tags = list(probs)
    y, player = ref["y"], ref["player"]
    logs = np.stack([np.log(probs[t]) for t in tags])
    players = np.unique(player)
    rng = np.random.default_rng(0)
    half = {p: i % 2 for i, p in enumerate(rng.permutation(players))}
    side = np.array([half[p] for p in player])
    report = {"models": tags, "single": {}, "fused": {}, "weights": []}

    def questions(prob, mask, seed):
        out = {}
        for rule, rows in simulate_questions(prob[mask], player[mask], y[mask], args.questions_per_player, seed).items():
            out[rule] = float(np.mean([r[1] for r in rows]))
        return out

    for t in tags:  # each single model, scored over the same two held-out halves for a fair comparison
        scores = [questions(probs[t], side == s, s) for s in (0, 1)]
        report["single"][t] = {r: float(np.mean([sc[r] for sc in scores])) for r in scores[0]}
    fused_scores = []
    for s in (0, 1):  # fit on the other half, score on this one
        w = fit_weights(logs[:, side != s], y[side != s])
        report["weights"].append(dict(zip(tags, map(float, w))))
        fused_scores.append(questions(fuse(logs, w), side == s, s))
    report["fused"] = {r: float(np.mean([sc[r] for sc in fused_scores])) for r in fused_scores[0]}
    print("weights per fold:", report["weights"])
    for t, sc in report["single"].items():
        print(f"{t:14s} mean/expected {sc['mean/expected']:.4f}  logmean/expected {sc['logmean/expected']:.4f}")
    print(f"{'FUSED':14s} mean/expected {report['fused']['mean/expected']:.4f}  logmean/expected {report['fused']['logmean/expected']:.4f}")
    (args.results / args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
