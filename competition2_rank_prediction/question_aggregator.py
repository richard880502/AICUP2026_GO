"""Learns how to pool the per-game probabilities of one question (player-grouped cross-validation).

Baseline: weighted log-probability fusion (weights fitted on the training folds) + log-mean pooling + expected-score
decoding. Learned: multinomial logistic regression on the per-model mean log-probabilities of the sampled games
(+ k), trained on questions sampled from the training-fold players and scored on questions from unseen players.

Example:
    c1/.venv/bin/python question_aggregator.py --models transformer_v:transformer move:move resnet5:resnet5
"""
import argparse
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import blend_eval
import rank_lib as lib


def sample_questions(player_rows, rng, per_player):
    out = []
    for pid, rows in player_rows.items():
        for _ in range(per_player):
            k = int(rng.integers(5, min(20, len(rows)) + 1))
            out.append((pid, rng.choice(rows, size=k, replace=False)))
    return out


def question_features(logs, qs):
    """logs [M, N, 10] per-game log-probs -> [Q, 3? M*10 + 2] features, labels are looked up by the caller."""
    feats = []
    for _, idx in qs:
        mean_logs = logs[:, idx].mean(1).reshape(-1)  # per-model mean log-prob
        feats.append(np.concatenate([mean_logs, [1 / np.sqrt(len(idx)), np.log(len(idx))]]))
    return np.asarray(feats)


def fit_final(logs, y, player, tags, c, per_player, seed, path):
    """Fits the aggregator on questions from every validation player and pickles it."""
    import pickle
    players = np.array(sorted(set(player)))
    rows_of = {p: np.flatnonzero(player == p) for p in players}
    qs = sample_questions(rows_of, np.random.default_rng(seed), per_player)
    x = question_features(logs, qs)
    labels = np.array([y[rows_of[p][0]] for p, _ in qs])
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=600).fit(scaler.transform(x), labels)
    pickle.dump({"scaler": scaler, "clf": clf, "models": tags, "c": c}, open(path, "wb"))
    print(f"saved aggregator ({len(qs)} questions, C={c}) -> {path}")


def bootstrap(who, a, b, reps=3000, seed=0):
    """Paired bootstrap over players: mean and 95% interval of mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(who)
    sums_a = np.array([a[who == u].sum() for u in uniq]); sums_b = np.array([b[who == u].sum() for u in uniq])
    n = np.array([(who == u).sum() for u in uniq])
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        diffs.append(sums_a[pick].sum() / n[pick].sum() - sums_b[pick].sum() / n[pick].sum())
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=pathlib.Path, default=None, help="fit on all validation players and pickle")
    ap.add_argument("--final-c", type=float, default=0.02)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--results", type=pathlib.Path, default=lib.ROOT / "results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-per-player", type=int, default=150)
    ap.add_argument("--test-per-player", type=int, default=60)
    ap.add_argument("--c", type=float, nargs="+", default=[0.03, 0.3, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    probs, ref = blend_eval.load(args.models, args.results)
    tags = list(probs)
    y, player = ref["y"], ref["player"]
    logs = np.stack([np.log(probs[t]) for t in tags])
    if args.save is not None:
        fit_final(logs, y, player, tags, args.final_c, args.train_per_player, args.seed, args.save)
        if args.skip_cv:
            return
    players = np.array(sorted(set(player)))
    rng = np.random.default_rng(args.seed)
    fold_of = dict(zip(rng.permutation(players), np.arange(len(players)) % args.folds))
    rows_of = {p: np.flatnonzero(player == p) for p in players}
    label_of = {p: y[rows_of[p][0]] for p in players}

    results = {"official (resnet5, mean softmax, argmax)": [], "baseline": [], **{f"learned C={c}": [] for c in args.c}}
    ks_all, who_all = [], []
    for fold in range(args.folds):
        tr_p = [p for p in players if fold_of[p] != fold]
        te_p = [p for p in players if fold_of[p] == fold]
        train_games = np.concatenate([rows_of[p] for p in tr_p])
        w = blend_eval.fit_weights(logs[:, train_games], y[train_games])
        q_tr = sample_questions({p: rows_of[p] for p in tr_p}, rng, args.train_per_player)
        q_te = sample_questions({p: rows_of[p] for p in te_p}, np.random.default_rng(1000 + fold), args.test_per_player)
        y_te = np.array([label_of[p] for p, _ in q_te])
        ks_all.append(np.array([len(i) for _, i in q_te]))
        who_all.append(np.array([p for p, _ in q_te]))
        if "resnet5" in tags:  # the tutorial's own rule: average the softmax of the games, take the argmax
            r = tags.index("resnet5")
            off = np.array([np.exp(logs[r][i]).mean(0).argmax() for _, i in q_te])
            results["official (resnet5, mean softmax, argmax)"].append(np.where(off == y_te, 1.0, np.where(np.abs(off - y_te) == 1, lib.ADJACENT, 0.0)))
        # baseline: fuse per game, then log-mean pool, then expected-score decode
        fused = blend_eval.fuse(logs, w)
        base = np.array([lib.decode(lib.pool(fused[i], "logmean")[None])[0] for _, i in q_te])
        results["baseline"].append(lib.competition_score(base, y_te) if False else np.where(base == y_te, 1.0, np.where(np.abs(base - y_te) == 1, lib.ADJACENT, 0.0)))
        x_tr, x_te = question_features(logs, q_tr), question_features(logs, q_te)
        y_tr = np.array([label_of[p] for p, _ in q_tr])
        scaler = StandardScaler().fit(x_tr)
        for c in args.c:
            clf = LogisticRegression(C=c, max_iter=400).fit(scaler.transform(x_tr), y_tr)
            pred = lib.decode(clf.predict_proba(scaler.transform(x_te)))
            results[f"learned C={c}"].append(np.where(pred == y_te, 1.0, np.where(np.abs(pred - y_te) == 1, lib.ADJACENT, 0.0)))
        print(f"fold {fold} done", flush=True)
    ks = np.concatenate(ks_all)
    who = np.concatenate(who_all)
    print(f"{'method':18s} {'all':>7s} {'k5-9':>7s} {'k10-14':>7s} {'k15-20':>7s}")
    for name, parts in results.items():
        sc = np.concatenate(parts)
        print(f"{name[:18]:18s} {sc.mean():7.4f} {sc[ks < 10].mean():7.3f} {sc[(ks >= 10) & (ks < 15)].mean():7.3f} {sc[ks >= 15].mean():7.3f}")
    official = np.concatenate(results["official (resnet5, mean softmax, argmax)"]) if "resnet5" in tags else None
    if official is not None:
        for name in results:
            if name.startswith("learned") or name == "baseline":
                d = bootstrap(who, np.concatenate(results[name]), official)
                print(f"BOOTSTRAP {name:16s} minus official: {d[0]:+.4f}  95% interval [{d[1]:+.4f}, {d[2]:+.4f}]")


if __name__ == "__main__":
    main()
