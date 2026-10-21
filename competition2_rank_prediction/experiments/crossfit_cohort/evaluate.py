"""Pre-registered evaluation on the 2,000-player cross-fitted cohort (see PREREGISTRATION.md).

Run once, after run_extract_train.sh and run_opp_models.sh:
c1/.venv/bin/python experiments/crossfit_cohort/evaluate.py

For each half j the *_c{j} models (trained without half j) score only half j's questions.
Everything is fitted on the validation cohort. Baseline = the default pipeline (predict_test.py:
d96 + resnet8c + d96 on opponent moves, val-fitted weights, val offsets from 10 questions per
player, log-mean pooling, expected-score decode).
C1 = shift residual (opponent_meta 'shift lam=0.01 [E1+E2+E3]', now residual_shift.ShiftResidual);
C2 = opponent-specific member opp_scratch_c{j} replacing the borrowed one; C1+C2 exploratory.
Holm over {C1, C2}, one-sided, alpha 0.05, paired bootstrap over questions (= players).
"""
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
import blend_eval  # noqa: E402  pylint: disable=wrong-import-position,import-error
import predict_test as pt  # noqa: E402  pylint: disable=wrong-import-position,import-error
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error
import residual_shift  # noqa: E402  pylint: disable=wrong-import-position,import-error
import transformer_rank as tr  # noqa: E402  pylint: disable=wrong-import-position,import-error

HERE = ROOT / "experiments/crossfit_cohort"
RES = ROOT / "results"
OUT = HERE / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
W = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
VARIANTS = (("baseline", "opp", False), ("C1", "opp", True), ("C2", "oppC2", False),
            ("C1+C2", "oppC2", True))


def ckpt_probs(tag: str, feats: dict, keys: np.ndarray) -> np.ndarray:
    """Mean probability of results/<tag>_seed*.pt on feats, rows in the order of keys
    (uniform where a game has no decision)."""
    idx = {k: i for i, k in enumerate(feats["game_key"])}
    rows = np.array([idx[k] for k in keys])
    x, valid = tr.decision_tokens(feats["tokens"][rows]), feats["masks"][rows].copy()
    empty = ~valid.any(1)
    valid[empty, 0] = True
    probs = []
    for path in sorted(RES.glob(f"{tag}_seed*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = tr.RankTransformer(x.shape[2], x.shape[1], a["d_model"], a["heads"], a["ffn"],
                               a["layers"], a["dropout"], a.get("head", "clm")).to(DEV)
        m.load_state_dict(ck["state"])
        xs = np.where(valid[..., None], (x - ck["mean"]) / ck["std"], 0).astype(np.float32)
        probs.append(tr.predict(m, xs, valid, DEV))
    assert len(probs) == 3, tag
    p = np.mean(probs, 0)
    p[empty] = 0.1
    return p


def score(qprob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-question competition score of the expected-score decode."""
    gap = np.abs((qprob @ W).argmax(1) - y)
    return np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))


def offsets_like_predict_test(vfused: np.ndarray, vplayer: np.ndarray) -> np.ndarray:
    """Per-rank offsets from 10 simulated 5-20 game questions per validation player (seed 0)."""
    rng = np.random.default_rng(0)
    vq = []
    for _ in range(10):
        for pid in np.unique(vplayer):
            r = np.flatnonzero(vplayer == pid)
            pick = rng.choice(r, size=min(int(rng.integers(5, 21)), len(r)), replace=False)
            vq.append(lib.pool(vfused[pick], "logmean"))
    return lib.fit_class_offsets(np.stack(vq))


def resnet_val_file(sfx: str) -> pathlib.Path:
    """Resnet validation probabilities = snapshot mean of epochs 5-8 (as production)."""
    vres = RES / f"val_probs_resnet8c{sfx}.npz"
    best = RES / f"val_probs_resnet8c{sfx}_bestepoch.npz"
    if not best.exists():
        shutil.copy(vres, best)
        v = np.load(best, allow_pickle=True)
        tok, ln, col = pt.aligned_moves(ROOT / "features/moves/all_moves.npz", v["game_key"])
        arrays: dict[str, Any] = {"y": v["y"], "player": v["player"], "game_key": v["game_key"],
                                  f"resnet8c{sfx}": pt.resnet_probs(tok, ln, col, DEV,
                                                                    f"resnet8c{sfx}")}
        np.savez(vres, **arrays)
    return vres


def member_dump(sfx: str) -> pathlib.Path:
    """Default pipeline per-game members on all holdout3 games (+ caches
    val_probs_d96_c{j}_opp.npz)."""
    dump = OUT / f"probs{sfx}.npz"
    if not dump.exists():
        r = subprocess.run([sys.executable, "predict_test.py",
                            "--tests-dir", "features/holdout3/tests",
                            "--feat", "features/feat96/holdout3",
                            "--moves", "features/moves/holdout3_moves.npz",
                            "--opp-feat", "features/feat96opp/holdout3", "--suffix", sfx,
                            "--expect", "2000", "--out", str(OUT / f"sub_all{sfx}.csv"),
                            "--dump", str(dump)],
                           cwd=ROOT, capture_output=True, text=True, check=False)
        assert dump.exists(), r.stdout + r.stderr
    return dump


def aligned_val(path: pathlib.Path, key: str, vkeys: np.ndarray) -> np.ndarray:
    """Validation member probabilities aligned to the d96 validation order (clipped)."""
    z = np.load(path, allow_pickle=True)
    i = {k: n for n, k in enumerate(z["game_key"])}
    return np.clip(z[key][[i[k] for k in vkeys]], 1e-6, 1)


class Half(NamedTuple):
    """Everything one cross-fitted half needs: validation / test members and questions."""
    sfx: str
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    vy: np.ndarray
    vplayer: np.ndarray
    vfin: np.ndarray
    tfin: np.ndarray
    tq: list[np.ndarray]
    ty: np.ndarray
    test_qids: np.ndarray


def game_rows(player_ids: np.ndarray) -> dict[str, list[int]]:
    """Question id -> its game rows in the dump."""
    by_q: dict[str, list[int]] = {}
    for i, q in enumerate(player_ids):
        by_q.setdefault(str(q), []).append(i)
    return by_q


def load_half(j: int, lab: pd.DataFrame, h3_opp: dict,
              fins: tuple[dict, dict]) -> tuple[Half, np.ndarray]:
    """Members, final-result features and questions of half j; also its rows in lab."""
    sfx = f"_c{j}"
    vres = resnet_val_file(sfx)
    d = np.load(member_dump(sfx), allow_pickle=True)
    keys = d["game_key"]
    vd = np.load(RES / f"val_probs_d96{sfx}.npz", allow_pickle=True)
    vkeys = vd["game_key"]
    val = {"d96": aligned_val(RES / f"val_probs_d96{sfx}.npz", "transformer", vkeys),
           "res": aligned_val(vres, f"resnet8c{sfx}", vkeys),
           "opp": aligned_val(RES / f"val_probs_d96{sfx}_opp.npz", "transformer", vkeys),
           "oppC2": aligned_val(RES / f"val_probs_opp_scratch{sfx}.npz", "transformer", vkeys)}
    test = {"d96": d[f"d96{sfx}"], "res": d[f"resnet8c{sfx}"], "opp": d[f"d96{sfx}_opp"],
            "oppC2": ckpt_probs(f"opp_scratch{sfx}", h3_opp, keys)}
    by_q = game_rows(d["player_id"])
    rows_j = np.flatnonzero(lab["fold"].values == j)
    qids = lab["question_id"].values[rows_j]
    return Half(sfx, val, test, vd["y"], vd["player"],
                np.stack([fins[0][k] for k in vkeys]), np.stack([fins[1][k] for k in keys]),
                [np.array(by_q[q]) for q in qids],
                np.array([lib.LEVELS.index(r) for r in lab["rank"].values[rows_j]]), qids), rows_j


def offsets_probs(name: str, half: Half, logs: tuple[np.ndarray, np.ndarray],
                  w: np.ndarray) -> np.ndarray:
    """Question probabilities of the default pipeline (fusion + validation offsets)."""
    vlogs, tlogs = logs
    b = offsets_like_predict_test(blend_eval.fuse(vlogs, w), half.vplayer)
    tfused = blend_eval.fuse(tlogs, w)
    qprob = lib.apply_class_offsets(np.stack([lib.pool(tfused[g], "logmean") for g in half.tq]), b)
    if name == "baseline":   # must reproduce predict_test.py's own decisions exactly
        sub = pd.read_csv(OUT / f"sub_all{half.sfx}.csv").set_index("question_id")["pred_rank"]
        mine = np.array(lib.LEVELS)[(qprob @ W).argmax(1)]
        agree = (mine == sub.loc[half.test_qids].values).mean()
        assert agree == 1.0, \
            f"baseline differs from predict_test.py on {1 - agree:.1%} of half {half.sfx[2:]}"
    return qprob


def residual_probs(half: Half, opp_key: str, logs: tuple[np.ndarray, np.ndarray],
                   w: np.ndarray) -> np.ndarray:
    """Question probabilities of the shift residual fitted on the validation questions
    (same as opponent_meta: 20 per player)."""
    vlogs, tlogs = logs
    vq, vqy = residual_shift.validation_questions(half.vplayer, half.vy)
    z, x = residual_shift.question_inputs(vq, blend_eval.fuse(vlogs, w), half.val["d96"],
                                          half.val[opp_key], half.vfin)
    model = residual_shift.ShiftResidual().fit(z, x, vqy, lam=0.01)
    tz, tx = residual_shift.question_inputs(half.tq, blend_eval.fuse(tlogs, w), half.test["d96"],
                                            half.test[opp_key], half.tfin)
    return model.predict_proba(tz, tx)


def evaluate_half(half: Half, per_q: dict[str, np.ndarray], rows_j: np.ndarray,
                  weights: dict[str, list]) -> None:
    """Scores the four variants on one half, filling per_q[...][rows_j] and weights."""
    for name, opp_key, residual in VARIANTS:
        members = ["d96", "res", opp_key]
        vlogs = np.log(np.stack([half.val[m] for m in members]))
        tlogs = np.log(np.stack([half.test[m] for m in members]))
        w = blend_eval.fit_weights(vlogs, half.vy)
        weights[f"{name} {half.sfx[1:]}"] = np.round(w, 3).tolist()
        if not residual:
            qprob = offsets_probs(name, half, (vlogs, tlogs), w)
        else:
            qprob = residual_probs(half, opp_key, (vlogs, tlogs), w)
        per_q[name][rows_j] = score(qprob, half.ty)


def build_report(per_q: dict[str, np.ndarray], lab: pd.DataFrame,
                 weights: dict[str, list]) -> dict[str, Any]:
    """Paired bootstrap over questions (seed 7) and Holm step-down over {C1, C2}."""
    rng = np.random.default_rng(7)
    boot = np.stack([rng.integers(0, len(lab), len(lab)) for _ in range(20000)])
    report: dict[str, Any] = {"n_questions": int(len(lab)), "weights": weights,
                              "scores": {k: float(v.mean()) for k, v in per_q.items()}}
    pvals: dict[str, float] = {}
    for name in ("C1", "C2", "C1+C2"):
        diff = per_q[name] - per_q["baseline"]
        bs = diff[boot].mean(1)
        p_one = float((bs <= 0).mean())
        report[name] = {"diff": float(diff.mean()),
                        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "p_one_sided": p_one,
                        "by_rank": {r: float(diff[lab["rank"].values == r].mean())
                                    for r in lib.LEVELS}}
        if name != "C1+C2":
            pvals[name] = p_one
    holm(report, pvals)
    return report


def holm(report: dict[str, Any], pvals: dict[str, float]) -> None:
    """Holm step-down, alpha 0.05: adds holm_threshold / adopt to report[name]."""
    order = sorted(pvals, key=pvals.__getitem__)
    alive = True
    for i, name in enumerate(order):
        thr = 0.05 / (len(order) - i)
        passed = alive and pvals[name] <= thr
        alive = passed
        report[name]["holm_threshold"], report[name]["adopt"] = thr, bool(passed)


def print_report(report: dict[str, Any], n_questions: int) -> None:
    """Console summary."""
    print(f"\nholdout3 cross-fitted, {n_questions} questions")
    print(f"baseline (current default)  {report['scores']['baseline']:.4f}")
    for name in ("C1", "C2", "C1+C2"):
        r = report[name]
        if name == "C1+C2":
            tag = "exploratory"
        else:
            verdict = "ADOPT" if r["adopt"] else "not adopted"
            tag = f"Holm thr {r['holm_threshold']:.3f} -> {verdict}"
        print(f"{name:6s} {report['scores'][name]:.4f}   diff {r['diff']:+.4f}  "
              f"95% [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"one-sided p {r['p_one_sided']:.4f}   {tag}")


def main() -> None:
    """Scores both halves, then writes results/report.json."""
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)   # as when the opponent_meta helpers were exec'd into this script
    lab = pd.read_csv(ROOT / "features/holdout3/question_folds.csv")
    h3_opp = lib.load_features(ROOT / "features/feat96opp/holdout3")
    fins = (residual_shift.load_final(ROOT / "features/final_result/val"),
            residual_shift.load_final(ROOT / "features/final_result/holdout3"))
    per_q = {name: np.full(len(lab), np.nan) for name in ("baseline", "C1", "C2", "C1+C2")}
    weights: dict[str, list] = {}
    for j in (0, 1):
        half, rows_j = load_half(j, lab, h3_opp, fins)
        evaluate_half(half, per_q, rows_j, weights)
        print(f"half {j}: " + "  ".join(f"{k} {v[rows_j].mean():.4f}" for k, v in per_q.items()),
              flush=True)
    report = build_report(per_q, lab, weights)
    print_report(report, len(lab))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
