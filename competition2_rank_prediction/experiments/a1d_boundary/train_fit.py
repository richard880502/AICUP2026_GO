"""Exp 1/2 re-evaluated without the small-sample CV bias of run.py.

run.py fits the extra features inside 5-fold CV over only 80 validation players per boundary; with
weak features that CV is biased far below 0.5 (leaving players out shifts the training class means
away from them). Here the feature combiners are fitted on the TRAINING cohort instead (2 games per
player, 1,500+ A and ~400 1D players), frozen, and scored once on the cached validation questions
(results/question_features.csv from run.py). The only thing fitted on validation is the 1-weight
blend s + w * z, with label-stratified player-disjoint CV (2 coefficients, negligible bias), and
that is reported next to the frozen z alone.
Run from the project root after run.py: c1/.venv/bin/python experiments/a1d_boundary/train_fit.py
"""
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import cross_val_predict  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
# `experiments` must be imported before rank_lib, which puts c1/ (with its own `experiments`
# package) in front of sys.path. run.py: Exp 1 / Exp 2 feature definitions, AUC and bootstrap.
from experiments.a1d_boundary import run as a1d  # noqa: E402  pylint: disable=wrong-import-position,import-error
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error

OUT, L, BOUNDARIES = a1d.OUT, a1d.L, a1d.BOUNDARIES


def question_features(old: tuple[np.ndarray, np.ndarray], dense: tuple[np.ndarray, np.ndarray],
                      value: tuple[np.ndarray, np.ndarray], vnames: list[str]) -> dict[str, Any]:
    """Same definitions as run.py relative_probability / error_profile, for one question's games;
    each argument but vnames is a (tokens, masks) pair."""
    f = a1d.relative_probability(*old)
    f.update(a1d.error_profile(dense, value, vnames))
    return f


def training_questions() -> pd.DataFrame:
    """Training cohort: one 2-game "question" per player of the boundary ranks."""
    t96 = lib.load_features(ROOT / "features/feat96/train")
    old = lib.load_features(ROOT / "features/feat/train")
    val_ = lib.load_features(ROOT / "features/feat_value/train")
    vnames = list(np.load(ROOT / "features/feat_value/train/feature_names.npy"))
    io = {k: i for i, k in enumerate(old["game_key"])}
    iv = {k: i for i, k in enumerate(val_["game_key"])}
    keep = np.array([k in io and k in iv for k in t96["game_key"]])
    print(f"training games with all three feature sets: {keep.sum()} / {len(keep)}")
    y96 = lib.rank_ids(t96["rank"])
    wanted = {r for b in BOUNDARIES for r in b}
    rows = []
    for p in np.unique(t96["player_id"]):
        g = np.flatnonzero((t96["player_id"] == p) & keep)
        if len(g) == 0 or y96[g[0]] not in wanted:
            continue
        go = np.array([io[k] for k in t96["game_key"][g]])
        gv = np.array([iv[k] for k in t96["game_key"][g]])
        f = {"player": p, "y": y96[g[0]], "k": len(g)}
        f.update(question_features((old["tokens"][go], old["masks"][go]),
                                   (t96["tokens"][g], t96["masks"][g]),
                                   (val_["tokens"][gv], val_["masks"][gv]), vnames))
        rows.append(f)
    return pd.DataFrame(rows)


def blend_oof(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Label-stratified, player-disjoint CV of a small logistic model (s + one frozen score)."""
    return a1d.cv_auc(df, cols, c=1.0, max_iter=100)


def labelled(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Rows of two adjacent ranks with hi = 1 for the higher one."""
    out = df[df["y"].isin([lo, hi])].reset_index(drop=True).copy()
    out["hi"] = (out["y"] == hi).astype(int)
    return out


def evaluate_set(train: pd.DataFrame, va: pd.DataFrame, cols: list[str],
                 s_oof: np.ndarray) -> dict[str, Any]:
    """Fits one feature set on the training questions, freezes it as z and scores it (alone and
    blended with s) on the validation questions."""
    yv = va["hi"].values
    sc = StandardScaler().fit(train[cols].values)
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(train[cols].values),
                                                       train["hi"].values)
    # in-train CV (many players, one question each: ordinary CV is fine here)
    tr_cv = cross_val_predict(LogisticRegression(C=0.5, max_iter=3000),
                              sc.transform(train[cols].values), train["hi"].values, cv=5,
                              method="decision_function")
    va["z"] = clf.decision_function(sc.transform(va[cols].fillna(train[cols].median()).values))
    both = blend_oof(va, ["s", "z"])
    return {"train_cv_auc": float(a1d.fast_auc(train["hi"].values, tr_cv)),
            "val_auc_frozen_z": float(a1d.fast_auc(yv, va["z"].values)),
            "val_auc_s_plus_z": float(a1d.fast_auc(yv, both)),
            "val_auc_s_cv": float(a1d.fast_auc(yv, s_oof)),
            "delta_vs_s": a1d.player_boot_auc_diff(va, both, s_oof),
            "corr_s_z": float(np.corrcoef(va["s"], va["z"])[0, 1])}


def evaluate_boundary(train_all: pd.DataFrame, val_all: pd.DataFrame, lo: int,
                      hi: int) -> dict[str, Any]:
    """Every feature set at one boundary, printed and returned."""
    exp1, exp2 = a1d.extra_columns(val_all)
    sets = {"exp1 relative probability": exp1, "exp2 error profile": exp2,
            "exp1 + exp2": exp1 + exp2,
            "exp2 surprisal only": [c for c in exp2 if c.startswith("surp_")],
            "exp2 score/winrate loss only": [c for c in exp2 if not c.startswith("surp_")]}
    train = labelled(train_all.dropna(), lo, hi)
    va = labelled(val_all, lo, hi)
    res: dict[str, Any] = {
        "train_players": {L[lo]: int((train["hi"] == 0).sum()),
                          L[hi]: int((train["hi"] == 1).sum())},
        "val_questions": int(len(va)), "val_players": int(va["player"].nunique()),
        "auc_s": float(a1d.fast_auc(va["hi"].values, va["s"].values)), "sets": {}}
    print(f"\n=== {L[lo]}/{L[hi]}: fitted on {res['train_players']} training players "
          f"(2 games each); scored on {len(va)} validation questions / "
          f"{res['val_players']} players")
    print(f"    s (production strength)            val AUC {res['auc_s']:.3f}")
    s_oof = blend_oof(va, ["s"])
    for sname, cols in sets.items():
        r = evaluate_set(train, va, cols, s_oof)
        d = r["delta_vs_s"]
        res["sets"][sname] = r
        print(f"    {sname:32s} train-CV AUC {r['train_cv_auc']:.3f}   "
              f"val AUC frozen z {r['val_auc_frozen_z']:.3f}   s+z {r['val_auc_s_plus_z']:.3f} "
              f"(s alone, same CV {r['val_auc_s_cv']:.3f}; diff {d[0]:+.3f} [{d[1]:+.3f}, "
              f"{d[2]:+.3f}])   corr(s,z) {r['corr_s_z']:+.2f}")
    return res


def main() -> None:
    """Fits on the training cohort, scores the validation questions, writes train_fit.json."""
    train_all = training_questions()
    val_all = pd.read_csv(OUT / "question_features.csv", dtype={"player": str})
    report = {f"{L[lo]}/{L[hi]}": evaluate_boundary(train_all, val_all, lo, hi)
              for lo, hi in BOUNDARIES}
    (OUT / "train_fit.json").write_text(json.dumps(report, indent=2))
    print("\nsaved", OUT / "train_fit.json")


if __name__ == "__main__":
    main()
