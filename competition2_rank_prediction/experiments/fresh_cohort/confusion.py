"""Question-level confusion matrices: best2 vs the new default (best2 + opponent).

Covers the fresh cohort and the old holdout, using the predict_test.py outputs in results/
(sub_h2_best2.csv, sub_h2_opp.csv, sub_h_best2.csv, sub_h_opp.csv).
Run from the project root: c1/.venv/bin/python experiments/fresh_cohort/confusion.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/visualization"))
# same style as the earlier confusion figure; make_figures is a script directory, not a package
from make_figures import (  # type: ignore[import-not-found]  # noqa: E402  # pylint: disable=import-error,wrong-import-position
    confusion, draw_confusion)

HERE = ROOT / "experiments/fresh_cohort"
L = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
CASES = {"fresh cohort": ("features/holdout2/tests/labels.csv", "h2"),
         "old holdout": ("features/holdout/tests/labels.csv", "h")}
VARIANTS = (("best2 (previous default)", "best2"), ("best2 + opponent (new default)", "opp"))


def entry(p: np.ndarray, t: np.ndarray, m: np.ndarray, exact: float, adj: float) -> dict:
    """Summary of one confusion matrix m of predictions p vs truth t."""
    gap = np.abs(p - t)
    score = float(np.where(gap == 0, 1, np.where(gap == 1, np.exp(-1), 0)).mean())
    return {"score": score, "exact": exact, "off_by_one": adj, "worse": 1 - exact - adj,
            "per_rank_exact": {L[i]: float(m[i, i] / m[i].sum()) for i in range(10)},
            "confusion": m.tolist()}


def cohort_summary(axes_row: np.ndarray, cohort: str, labels: str, tag: str) -> dict[str, dict]:
    """Draws the two confusion matrices of one cohort and returns their summaries."""
    lab = pd.read_csv(ROOT / labels).set_index("question_id")["rank"]
    summary = {}
    for col, (name, variant) in enumerate(VARIANTS):
        sub = pd.read_csv(HERE / "results" / f"sub_{tag}_{variant}.csv")
        t = np.array([L.index(lab[q]) for q in sub["question_id"]])
        p = np.array([L.index(r) for r in sub["pred_rank"]])
        m = confusion(p, t)
        exact, adj = draw_confusion(axes_row[col], m,
                                    f"{cohort} (400 players): {name}\nrow % of each true rank")
        summary[f"{cohort} | {name}"] = entry(p, t, m, exact, adj)
    return summary


def main() -> None:
    """Draws the 2 x 2 figure and writes results/confusion_summary.json."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.4))
    summary = {}
    for row, (cohort, (labels, tag)) in enumerate(CASES.items()):
        summary.update(cohort_summary(axes[row], cohort, labels, tag))
    fig.tight_layout()
    fig.savefig(HERE / "figures/confusion_best2_vs_opponent.png", dpi=150)
    (HERE / "results/confusion_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k:55s} score {v['score']:.4f}  exact {v['exact']:.3f}  "
              f"off-by-one {v['off_by_one']:.3f}  worse {v['worse']:.3f}")
    print("\nper-rank exact (fresh cohort):  rank  best2  ->  +opponent")
    a = summary["fresh cohort | best2 (previous default)"]
    b = summary["fresh cohort | best2 + opponent (new default)"]
    for r in L:
        print(f"   {r:3s} {a['per_rank_exact'][r]:.2f} -> {b['per_rank_exact'][r]:.2f}")


if __name__ == "__main__":
    main()
