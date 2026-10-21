"""Experiment A: how much of the 0.94 ranking ability survives with d96_h, three seeds and real
5-20 game questions.

Uses results/questions.npz written by run.py (same questions as the method comparison, so
experiment B is read off the same rows).
Writes figures/strength_by_rank.png, figures/adjacent_overlap.png and results/ranking_ability.json.
Run from the project root: c1/.venv/bin/python experiments/strength_axis/ranking_ability.py
"""
import json
import pathlib
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr  # type: ignore[import-untyped]

HERE = pathlib.Path(__file__).resolve().parent
L = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#1f1f1d", "#5f5e5a", "#d6d5d0"
# subset of the validated ordinal blue ramp
K_RAMP = {"5-9": "#86b6ef", "10-14": "#256abf", "15-20": "#0d366b"}
GROUPS = {"5-9": (5, 9), "10-14": (10, 14), "15-20": (15, 20)}
METHODS = ("A_best2", "A2_dense_only", "C_axis_calibrated", "D_128d")


def adjacent_auc(s: np.ndarray, y: np.ndarray) -> list[float]:
    """P(score of a random rank r+1 question > score of a random rank r question), per adjacent
    pair."""
    out = []
    for r in range(9):
        a, b = s[y == r], s[y == r + 1]
        out.append(float(mannwhitneyu(b, a, alternative="two-sided").statistic / (len(a) * len(b)))
                   if len(a) and len(b) else np.nan)
    return out


def block(s: np.ndarray, y: np.ndarray, k: np.ndarray,
          scores: dict[str, np.ndarray]) -> dict[str, Any]:
    """Spearman, adjacent-pair AUCs, the same by question size (with each method's score / exact
    rate) and per-rank strength quantiles of one set of questions."""
    res: dict[str, Any] = {"n": int(len(s)), "spearman": float(spearmanr(s, y).correlation),
                           "adjacent_auc": adjacent_auc(s, y), "by_games": {}}
    for g, (lo, hi) in GROUPS.items():
        m = (k >= lo) & (k <= hi)
        res["by_games"][g] = {
            "n": int(m.sum()), "spearman": float(spearmanr(s[m], y[m]).correlation),
            "adjacent_auc": adjacent_auc(s[m], y[m]),
            **{name: {"score": float(v[m].mean()), "exact": float((v[m] == 1).mean())}
               for name, v in scores.items()}}
    res["per_rank_quantiles"] = {
        L[r]: [float(q) for q in np.quantile(s[y == r], [0.1, 0.25, 0.5, 0.75, 0.9])]
        for r in range(10)}
    return res


def print_report(report: dict[str, Any]) -> None:
    """Prints both question sets' summaries."""
    for name, r in report.items():
        print(f"\n== {name} (n={r['n']})   Spearman(strength, rank) = {r['spearman']:.3f}")
        print("   adjacent-pair AUC: " + "  ".join(f"{L[i]}/{L[i + 1]} {a:.2f}"
                                                   for i, a in enumerate(r["adjacent_auc"])))
        print(f"   {'games':6s} {'n':>5s} {'Spearman':>9s}   score / exact:  {'A best2':>14s} "
              f"{'A2 dense':>14s} {'C axis':>14s} {'D 128d':>14s}")
        for g, b in r["by_games"].items():
            cells = "".join(f"  {b[m]['score']:.3f} / {b[m]['exact']:.3f}" for m in METHODS)
            print(f"   {g:6s} {b['n']:5d} {b['spearman']:9.3f}  {cells}")


def tidy_axis(ax: Any) -> None:
    """Horizontal grid behind the data, no top / right spines."""
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_strength_by_rank(questions: Any, by_games: dict[str, Any]) -> None:
    """Figure 1: strength score distribution per true rank, small multiples by question size
    (validation questions)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    s, y, k = questions["q_s"], questions["q_y"], questions["q_k"]
    for ax, (g, (lo, hi)) in zip(axes, GROUPS.items()):
        m = (k >= lo) & (k <= hi)
        data = [s[m & (y == r)] for r in range(10)]
        bp = ax.boxplot(data, whis=(10, 90), showfliers=False, widths=0.55, patch_artist=True,
                        medianprops={"color": "#ffffff", "linewidth": 2},
                        whiskerprops={"color": INK2}, capprops={"color": INK2})
        for patch in bp["boxes"]:
            patch.set(facecolor=K_RAMP[g], edgecolor=K_RAMP[g])
        ax.set_xticks(range(1, 11), L)
        ax.set_title(f"{g} games per question  (Spearman {by_games[g]['spearman']:.2f})",
                     fontsize=11, color=INK, loc="left")
        tidy_axis(ax)
    axes[0].set_ylabel("question strength score (standardised)")
    fig.suptitle("Strength score by true rank, validation questions "
                 "(box = 25-75%, whiskers = 10-90%)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "figures/strength_by_rank.png", dpi=150)


def plot_adjacent_overlap(by_games: dict[str, Any]) -> None:
    """Figure 2: adjacent-rank separability (AUC) by question size."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(9)
    for g in GROUPS:
        auc = by_games[g]["adjacent_auc"]
        ax.plot(x, auc, color=K_RAMP[g], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        dy = {"5-9": 0, "10-14": -7, "15-20": 7}[g]   # keep the three end labels apart
        ax.annotate(f"{g} games", (x[-1], auc[-1]), xytext=(8, dy), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("0.5 = complete overlap", (0, 0.5), xytext=(0, -12), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.set_xticks(x, [f"{L[i]}/{L[i + 1]}" for i in range(9)])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("P(higher rank scores higher)")
    ax.set_title("How well the strength score separates neighbouring ranks (validation questions)",
                 fontsize=11, color=INK, loc="left")
    tidy_axis(ax)
    ax.set_xlim(-0.3, 9.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures/adjacent_overlap.png", dpi=150)


def main() -> None:
    """Writes results/ranking_ability.json and the two figures."""
    matplotlib.use("Agg")
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED,
                         "font.size": 10})
    z = np.load(HERE / "results/questions.npz", allow_pickle=True)
    val_scores = {"A_best2": z["cv_A"], "A2_dense_only": z["cv_A2"], "C_axis_calibrated": z["cv_C"],
                  "D_128d": z["cv_D"]}
    hold_scores = {"A_best2": z["hold_A"], "A2_dense_only": z["hold_A2"],
                   "C_axis_calibrated": z["hold_C"], "D_128d": z["hold_D"]}
    report = {"validation_questions": block(z["q_s"], z["q_y"], z["q_k"], val_scores),
              "holdout_questions": block(z["h_s"], z["h_y"], z["h_k"], hold_scores)}
    (HERE / "results/ranking_ability.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    plot_strength_by_rank(z, report["validation_questions"]["by_games"])
    plot_adjacent_overlap(report["validation_questions"]["by_games"])
    print("\nfigures written")


if __name__ == "__main__":
    main()
