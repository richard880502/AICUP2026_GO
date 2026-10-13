"""Trains per-game rank classifiers and scores them on simulated 5-20 game questions.

Example:
    c1/.venv/bin/python train_eval.py --train features/feat/train --val features/feat/val
    c1/.venv/bin/python train_eval.py --train features/feat/train --val features/feat/val --final
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rank_lib as lib


def fit_models(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, object]:
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=seed)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
    return {"hgb": hgb.fit(x, y), "logistic": logistic.fit(x, y)}


def game_probs(models: dict[str, object], x: np.ndarray) -> dict[str, np.ndarray]:
    probs = {name: model.predict_proba(x) for name, model in models.items()}
    probs["blend"] = np.mean(list(probs.values()), axis=0)
    return probs


def simulate_questions(prob, player, truth, questions_per_player, seed):
    """Samples 5-20 game questions from each validation player; returns scores per pooling rule."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[int, float]]] = {}
    for pid in np.unique(player):
        rows = np.flatnonzero(player == pid)
        for _ in range(questions_per_player):
            k = int(rng.integers(5, min(20, len(rows)) + 1))
            chosen = rng.choice(rows, size=k, replace=False)
            for how in ("mean", "logmean"):
                pooled = lib.pool(prob[chosen], how)[None]
                for decoder, fn in (("argmax", lambda p: p.argmax(1)), ("expected", lib.decode)):
                    out.setdefault(f"{how}/{decoder}", []).append((k, lib.competition_score(fn(pooled), truth[chosen[:1]])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--val", type=pathlib.Path, required=True)
    ap.add_argument("--final", action="store_true", help="refit on train+val and save the model")
    ap.add_argument("--questions-per-player", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    train, val = lib.load_features(a.train), lib.load_features(a.val)
    xt, yt = lib.game_features(train["tokens"], train["masks"]), lib.rank_ids(train["rank"])
    xv, yv = lib.game_features(val["tokens"], val["masks"]), lib.rank_ids(val["rank"])
    print(f"train {xt.shape}  val {xv.shape}", flush=True)
    assert not set(train["player_id"]) & set(val["player_id"]), "train/val players overlap"

    models = fit_models(xt, yt, a.seed)
    probs = game_probs(models, xv)
    np.savez(a.out / "val_probs_gbm.npz", y=yv, player=val["player_id"], game_key=val["game_key"], **probs)
    report = {"n_train_games": int(len(yt)), "n_val_games": int(len(yv)), "per_game": {}, "question": {}}
    for name, p in probs.items():
        report["per_game"][name] = {"top1": float((p.argmax(1) == yv).mean()),
                                    "score": lib.competition_score(p.argmax(1), yv)}
        print(f"per-game {name:8s} top1 {report['per_game'][name]['top1']:.4f} "
              f"score {report['per_game'][name]['score']:.4f}")
    for name, p in probs.items():
        sims = simulate_questions(p, val["player_id"], yv, a.questions_per_player, a.seed)
        for rule, rows in sims.items():
            ks, sc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            report["question"][f"{name}:{rule}"] = {
                "all": float(sc.mean()), "k5-9": float(sc[ks < 10].mean()),
                "k10-14": float(sc[(ks >= 10) & (ks < 15)].mean()), "k15-20": float(sc[ks >= 15].mean())}
            print(f"question {name:8s} {rule:14s} {sc.mean():.4f}  (k5-9 {sc[ks < 10].mean():.3f}, "
                  f"k15-20 {sc[ks >= 15].mean():.3f})")
    (a.out / "val_report.json").write_text(json.dumps(report, indent=2))

    if a.final:
        x, y = np.concatenate([xt, xv]), np.concatenate([yt, yv])
        final = fit_models(x, y, a.seed)
        with open(a.out / "final_models.pkl", "wb") as handle:
            pickle.dump(final, handle)
        print("saved", a.out / "final_models.pkl")


if __name__ == "__main__":
    main()
