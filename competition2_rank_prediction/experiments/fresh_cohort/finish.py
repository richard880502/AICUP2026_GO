"""Fresh-cohort confirmation of the opponent-side member (run after run.sh).

1. Validation probabilities of resnet8c_h2 as production uses them (mean of snapshot epochs 5-8,
   like val_probs_resnet8c_h.npz); the best-epoch file written by resnet_rank.py is kept as
   val_probs_resnet8c_h2_bestepoch.npz.
2. predict_test.py --suffix _h2 on features/holdout2 (400 new players, never trained on by any *_h2
   model) with --opp-feat and --dump, to get every member's per-game probabilities.
3. Pre-registered comparison (decided before looking): best2 vs best2 + opponent, and d96 vs d96 +
   opponent; fusion weights and per-rank offsets fitted on the validation cohort only; log-mean
   pooling; expected-score decode; paired bootstrap over players.
Run from the project root: c1/.venv/bin/python experiments/fresh_cohort/finish.py
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

HERE = ROOT / "experiments/fresh_cohort"
RES = ROOT / "results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VAL_RESNET = RES / "val_probs_resnet8c_h2.npz"
VAL_FILES = {"d96_h2": (RES / "val_probs_d96_h2.npz", "transformer"),
             "resnet8c_h2": (VAL_RESNET, "resnet8c_h2"),
             "d96_h2_opp": (RES / "val_probs_d96_h2_opp.npz", "transformer")}
W = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
CONFIGS = {"best2 (d96_h2 + resnet8c_h2)": ["d96_h2", "resnet8c_h2"],
           "best2 + opponent": ["d96_h2", "resnet8c_h2", "d96_h2_opp"],
           "d96_h2 alone": ["d96_h2"],
           "d96_h2 + opponent": ["d96_h2", "d96_h2_opp"],
           "opponent alone": ["d96_h2_opp"]}


class Cohort(NamedTuple):
    """Fresh-cohort member probabilities per game, question ids, game rows per question, truth."""
    members: dict[str, np.ndarray]
    qids: list[str]
    by_q: dict[str, list[int]]
    truth: np.ndarray


def snapshot_val_probs() -> None:
    """Step 1: snapshot-mean validation probabilities for the new ResNet."""
    best = RES / "val_probs_resnet8c_h2_bestepoch.npz"
    if best.exists():
        return
    shutil.copy(VAL_RESNET, best)
    v = np.load(best, allow_pickle=True)
    tokens, lengths, colour = pt.aligned_moves(ROOT / "features/moves/all_moves.npz",
                                               v["game_key"])
    p = pt.resnet_probs(tokens, lengths, colour, DEV, "resnet8c_h2")
    np.savez(VAL_RESNET, y=v["y"], player=v["player"], game_key=v["game_key"], resnet8c_h2=p)
    print("val_probs_resnet8c_h2 = snapshot mean of epochs", pt.SNAPSHOT_EPOCHS["resnet8c"])


def dump_members() -> pathlib.Path:
    """Step 2: production inference on the fresh cohort, dumping per-game member probabilities."""
    dump = HERE / "results/probs_h2.npz"
    dump.parent.mkdir(exist_ok=True)
    if not dump.exists():
        cmd = [sys.executable, "predict_test.py", "--tests-dir", "features/holdout2/tests",
               "--feat", "features/feat96/holdout2", "--moves", "features/moves/holdout2_moves.npz",
               "--suffix", "_h2", "--expect", "400",
               "--labels", "features/holdout2/tests/labels.csv",
               "--out", str(HERE / "results/sub_h2_opp.csv"),
               "--opp-feat", "experiments/opponent_side/features/feat96opp/holdout2",
               "--dump", str(dump)]
        print(subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout)
    return dump


def load_validation(models: list[str]) -> tuple[np.ndarray, Any, Any]:
    """Validation log-probabilities [M, G, 10] of the models, and the labels / players per game."""
    vl = []
    vy: Any = None
    vp: Any = None
    for m in models:
        f, key = VAL_FILES[m]
        v = np.load(f, allow_pickle=True)
        vl.append(np.log(np.clip(v[key], 1e-6, 1)))
        vy, vp = v["y"], v["player"]
    return np.stack(vl), vy, vp


def fit_on_validation(models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fusion weights and per-rank offsets (10 simulated questions per player), validation only."""
    val_logs, vy, vp = load_validation(models)
    w = blend_eval.fit_weights(val_logs, vy)
    vf = blend_eval.fuse(val_logs, w)
    rng = np.random.default_rng(0)
    vq = []
    for _ in range(10):
        for pid in np.unique(vp):
            r = np.flatnonzero(vp == pid)
            pick = rng.choice(r, size=min(int(rng.integers(5, 21)), len(r)), replace=False)
            vq.append(lib.pool(vf[pick], "logmean"))
    return w, lib.fit_class_offsets(np.stack(vq))


def per_question(models: list[str], cohort: Cohort) -> tuple[np.ndarray, dict[str, float]]:
    """Per-question competition score of the fused models on the fresh cohort, and the weights."""
    w, b = fit_on_validation(models)
    hf = blend_eval.fuse(np.stack([np.log(np.clip(cohort.members[m], 1e-6, 1)) for m in models]),
                         w)
    hq = lib.apply_class_offsets(
        np.stack([lib.pool(hf[cohort.by_q[q]], "logmean") for q in cohort.qids]), b)
    gap = np.abs((hq @ W).argmax(1) - cohort.truth)
    scores = np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))
    return scores, dict(zip(models, np.round(w, 3).tolist()))


def load_cohort(dump: pathlib.Path) -> tuple[Cohort, int]:
    """Fresh-cohort members from the dump and its labels; also returns the number of games."""
    d = np.load(dump, allow_pickle=True)
    members = {m: d[m] for m in ("d96_h2", "resnet8c_h2", "d96_h2_opp")}
    lab = pd.read_csv(ROOT / "features/holdout2/tests/labels.csv")
    by_q: dict[str, list[int]] = {}
    for i, q in enumerate(d["player_id"]):
        by_q.setdefault(str(q), []).append(i)
    truth = np.array([lib.LEVELS.index(r) for r in lab["rank"]])
    return Cohort(members, lab["question_id"].tolist(), by_q, truth), len(d["game_key"])


def main() -> None:
    """Runs the three steps and writes results/report.json."""
    snapshot_val_probs()
    cohort, n_games = load_cohort(dump_members())
    # ---- 3. pre-registered comparison
    res: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    print(f"fresh cohort: {len(cohort.qids)} questions / players, {n_games} games")
    for name, models in CONFIGS.items():
        s, w = per_question(models, cohort)
        res[name] = s
        report[name] = {"score": float(s.mean()), "se": float(s.std(ddof=1) / np.sqrt(len(s))),
                        "weights": w}
        print(f"{name:32s} {s.mean():.4f} +-{report[name]['se']:.3f}   weights {w}")
    rng = np.random.default_rng(1)
    for a, b in (("best2 + opponent", "best2 (d96_h2 + resnet8c_h2)"),
                 ("d96_h2 + opponent", "d96_h2 alone")):
        diff = res[a] - res[b]
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        report[f"[{a}] minus [{b}]"] = [float(diff.mean()), float(lo), float(hi)]
        print(f"[{a}] minus [{b}]: {diff.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]")
    (HERE / "results/report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
