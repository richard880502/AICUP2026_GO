"""Analyze and visualize which features separate hard adjacent-rank boundaries.

Consumes feature_table.npz from build_features.py and writes:
  * raw adjacent-boundary AUC heatmap;
  * v47-conditioned residual AUC heatmap;
  * family-level conditional AUC heatmap;
  * feature redundancy vs conditional-separability scatter;
  * matched-hard-pair effect heatmap;
  * detailed plots for A/1D, 1D/2D and 4D/5D;
  * CSV/JSON metrics and a lightweight HTML index.

No model selected here is used for submission. This is a diagnostic atlas.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
BOUNDARIES = tuple(f"{RANKS[i]}/{RANKS[i + 1]}" for i in range(9))
TARGET_BOUNDARIES = (3, 4, 7)
EPS = 1e-9


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC; label must be 0/1."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    pos = label == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(score, method="average")
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def separability(value: float) -> float:
    """Direction-free AUC, 0.5 random -> 1 perfect."""
    return max(value, 1.0 - value) if np.isfinite(value) else float("nan")


def expected_rank(probability: np.ndarray) -> np.ndarray:
    """Expected class index."""
    return probability @ np.arange(10, dtype=np.float64)


def decode(probability: np.ndarray) -> np.ndarray:
    """Official expected-utility decode."""
    gain = np.eye(10)
    gain += np.exp(-1.0) * (np.eye(10, k=1) + np.eye(10, k=-1))
    return (probability @ gain).argmax(1)


def conditional_residual(
    feature: np.ndarray,
    baseline_score: np.ndarray,
    boundary_mask: np.ndarray,
    fold: np.ndarray,
) -> np.ndarray:
    """Fold-safe residual of a feature after linear dependence on v47 pair log-odds."""
    result = np.full(len(feature), np.nan, np.float64)
    for target_fold in (0, 1):
        train = boundary_mask & (fold != target_fold)
        test = boundary_mask & (fold == target_fold)
        if train.sum() < 3 or test.sum() == 0:
            continue
        design = np.stack(
            [np.ones(train.sum()), baseline_score[train]], axis=1
        )
        beta, *_ = np.linalg.lstsq(design, feature[train], rcond=None)
        test_design = np.stack(
            [np.ones(test.sum()), baseline_score[test]], axis=1
        )
        result[test] = feature[test] - test_design @ beta
    return result


def family(name: str) -> str:
    """Feature family prefix."""
    return name.split("__", 1)[0] if "__" in name else "other"


def baseline_boundary_score(probability: np.ndarray, low: int) -> np.ndarray:
    """v47 high-vs-low log odds for one adjacent boundary."""
    return np.log(np.clip(probability[:, low + 1], EPS, 1)) - np.log(
        np.clip(probability[:, low], EPS, 1)
    )


def matched_pairs(
    score: np.ndarray,
    game_count: np.ndarray,
    truth: np.ndarray,
    fold: np.ndarray,
    low: int,
    keep_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one low/high matching on v47 score and question size, fold by fold."""
    lows, highs, costs = [], [], []
    for which_fold in (0, 1):
        left = np.flatnonzero((truth == low) & (fold == which_fold))
        right = np.flatnonzero((truth == low + 1) & (fold == which_fold))
        if not len(left) or not len(right):
            continue
        score_scale = max(float(np.std(score[np.r_[left, right]])), 1e-6)
        count_scale = max(float(np.std(game_count[np.r_[left, right]])), 1.0)
        cost = (
            np.abs(score[left, None] - score[right][None, :]) / score_scale
            + 0.15
            * np.abs(game_count[left, None] - game_count[right][None, :])
            / count_scale
        )
        row, col = linear_sum_assignment(cost)
        lows.extend(left[row].tolist())
        highs.extend(right[col].tolist())
        costs.extend(cost[row, col].tolist())
    if not lows:
        return np.zeros(0, int), np.zeros(0, int)
    lows_arr = np.asarray(lows)
    highs_arr = np.asarray(highs)
    costs_arr = np.asarray(costs)
    cutoff = np.quantile(costs_arr, keep_quantile)
    keep = costs_arr <= cutoff
    return lows_arr[keep], highs_arr[keep]


def bootstrap_effect(
    low_values: np.ndarray,
    high_values: np.ndarray,
    scale: float,
    seed: int,
    trials: int,
) -> tuple[float, float, float]:
    """Matched-pair standardized effect and paired bootstrap interval."""
    diff = (high_values - low_values) / max(scale, 1e-9)
    effect = float(np.mean(diff))
    if len(diff) < 2:
        return effect, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(diff), size=(trials, len(diff)))
    values = diff[sampled].mean(1)
    return effect, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def heatmap(
    matrix: np.ndarray,
    rows: list[str],
    columns: tuple[str, ...],
    title: str,
    output: pathlib.Path,
    vmin: float,
    vmax: float,
    fmt: str = ".2f",
) -> None:
    """Readable annotated heatmap."""
    height = max(5.0, 0.34 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, height))
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title(title, loc="left")
    for y in range(len(rows)):
        for x in range(len(columns)):
            value = matrix[y, x]
            if np.isfinite(value):
                ax.text(x, y, format(value, fmt), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def hard_boundary_plot(
    low: int,
    names: list[str],
    features: np.ndarray,
    truth: np.ndarray,
    fold: np.ndarray,
    v47: np.ndarray,
    conditional: np.ndarray,
    output: pathlib.Path,
    top_n: int = 4,
) -> None:
    """Top conditional features inside the genuinely difficult half of a boundary."""
    boundary = (truth == low) | (truth == low + 1)
    score = baseline_boundary_score(v47, low)
    difficulty = np.abs(score[boundary])
    threshold = float(np.median(difficulty))
    hard = boundary & (np.abs(score) <= threshold)
    pred = decode(v47)
    swapped = boundary & np.isin(pred, [low, low + 1]) & (pred != truth)
    hard |= swapped

    order = np.argsort(-conditional[:, low])
    chosen = [
        index for index in order
        if np.isfinite(conditional[index, low])
        and not names[index].startswith(("base__", "v47__"))
    ][:top_n]
    if not chosen:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(9.5, 2.6 * len(chosen)))
    axes = np.atleast_1d(axes)
    for ax, index in zip(axes, chosen):
        low_values = features[hard & (truth == low), index]
        high_values = features[hard & (truth == low + 1), index]
        ax.boxplot(
            [low_values, high_values],
            labels=[RANKS[low], RANKS[low + 1]],
            showfliers=False,
            whis=(10, 90),
        )
        ax.set_title(
            f"{names[index]}  |  conditional AUC {conditional[index, low]:.3f}",
            loc="left",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"{RANKS[low]} vs {RANKS[low + 1]} — top features inside hard/confused questions",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def redundancy_scatter(
    names: list[str],
    max_conditional: np.ndarray,
    corr: np.ndarray,
    output: pathlib.Path,
) -> None:
    """Independent-signal map: low v47 correlation and high conditional AUC is desirable."""
    keep = np.isfinite(max_conditional) & np.isfinite(corr)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(np.abs(corr[keep]), max_conditional[keep], alpha=0.65)
    candidates = np.flatnonzero(keep)
    score = max_conditional[candidates] - 0.15 * np.abs(corr[candidates])
    for index in candidates[np.argsort(-score)[:15]]:
        ax.annotate(
            names[index],
            (abs(corr[index]), max_conditional[index]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0.58, linestyle="--", linewidth=1)
    ax.axvline(0.8, linestyle="--", linewidth=1)
    ax.set_xlabel("|Spearman correlation with v47 expected rank|")
    ax.set_ylabel("Best v47-conditioned adjacent-boundary AUC")
    ax.set_title("Where independent discriminative signal may remain", loc="left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def boundary_bar(
    low: int,
    names: list[str],
    raw: np.ndarray,
    conditional: np.ndarray,
    output: pathlib.Path,
    top_n: int = 12,
) -> None:
    """Raw vs conditional separability for one boundary."""
    candidate = np.arange(len(names))
    candidate = candidate[
        ~np.asarray([name.startswith(("base__", "v47__")) for name in names])
    ]
    chosen = candidate[np.argsort(-conditional[candidate, low])[:top_n]]
    y = np.arange(len(chosen))
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.42 * len(chosen) + 1.5)))
    ax.barh(y - 0.18, raw[chosen, low], height=0.34, label="Raw AUC")
    ax.barh(y + 0.18, conditional[chosen, low], height=0.34, label="After v47 residualization")
    ax.axvline(0.5, linewidth=1, linestyle="--")
    ax.set_yticks(y, [names[index] for index in chosen])
    ax.invert_yaxis()
    ax.set_xlim(0.48, 1.0)
    ax.set_xlabel("Direction-free AUC")
    ax.set_title(f"{RANKS[low]} vs {RANKS[low + 1]} — strongest non-baseline features", loc="left")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_html(
    output: pathlib.Path,
    summary: dict,
    figure_names: list[str],
    top_rows: list[dict],
) -> None:
    """Small local HTML index for browsing the atlas."""
    rows = "".join(
        "<tr>"
        + f"<td>{html.escape(row['feature'])}</td>"
        + f"<td>{html.escape(row['boundary'])}</td>"
        + f"<td>{row['conditional_auc']:.3f}</td>"
        + f"<td>{row['raw_auc']:.3f}</td>"
        + f"<td>{row['corr_v47']:.3f}</td>"
        + "</tr>"
        for row in top_rows
    )
    figures = "".join(
        f"<section><h2>{html.escape(name)}</h2>"
        f"<img src='figures/{html.escape(name)}' alt='{html.escape(name)}'></section>"
        for name in figure_names
    )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Hard Boundary Feature Atlas</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#222}}
code{{background:#f2f2f2;padding:2px 5px;border-radius:4px}}
table{{border-collapse:collapse;width:100%;margin:18px 0 32px}}
th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #ddd}}
section{{margin:36px 0}}
</style>
<h1>Hard Boundary Feature Atlas</h1>
<p>Questions: {summary['questions']} · features: {summary['features']} · v47 score: {summary['v47_score']:.4f}</p>
<h2>Highest conditional-separability candidates</h2>
<table><thead><tr><th>Feature</th><th>Boundary</th><th>Conditional AUC</th><th>Raw AUC</th><th>corr(v47)</th></tr></thead>
<tbody>{rows}</tbody></table>
{figures}
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Writes the complete atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/hard_boundary_atlas/"
            "feature_table.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/hard_boundary_atlas"
        ),
    )
    parser.add_argument("--top-features", type=int, default=24)
    parser.add_argument("--hard-pair-quantile", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    with np.load(args.table, allow_pickle=True) as saved:
        truth = saved["truth"].astype(np.int64)
        fold = saved["fold"].astype(np.int64)
        v47 = saved["v47_prob"].astype(np.float64)
        names = saved["feature_names"].astype(str).tolist()
        features = saved["features"].astype(np.float64)

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    n_features = features.shape[1]
    raw = np.full((n_features, 9), np.nan)
    conditional = np.full((n_features, 9), np.nan)
    direction = np.full((n_features, 9), np.nan)
    baseline_auc = np.full(9, np.nan)
    hard_effect = np.full((n_features, 9), np.nan)
    hard_lo = np.full((n_features, 9), np.nan)
    hard_hi = np.full((n_features, 9), np.nan)
    pair_counts = np.zeros(9, dtype=np.int64)

    game_count_index = names.index("structure__game_count")
    game_count = features[:, game_count_index]

    for low in range(9):
        boundary = (truth == low) | (truth == low + 1)
        y = (truth[boundary] == low + 1).astype(np.int64)
        score = baseline_boundary_score(v47, low)
        baseline_auc[low] = auc(score[boundary], y)
        low_pair, high_pair = matched_pairs(
            score, game_count, truth, fold, low, args.hard_pair_quantile
        )
        pair_counts[low] = len(low_pair)

        for feature_index in range(n_features):
            value = features[:, feature_index]
            raw_signed = auc(value[boundary], y)
            raw[feature_index, low] = separability(raw_signed)
            direction[feature_index, low] = 1.0 if raw_signed >= 0.5 else -1.0

            residual = conditional_residual(value, score, boundary, fold)
            cond_signed = auc(residual[boundary], y)
            conditional[feature_index, low] = separability(cond_signed)

            if len(low_pair):
                scale = float(np.std(value[boundary]))
                effect, lo, hi = bootstrap_effect(
                    value[low_pair],
                    value[high_pair],
                    scale,
                    1000 + low * 100 + feature_index,
                    args.bootstrap,
                )
                hard_effect[feature_index, low] = effect
                hard_lo[feature_index, low] = lo
                hard_hi[feature_index, low] = hi

    v47_expected = expected_rank(v47)
    corr = np.asarray(
        [
            float(spearmanr(features[:, index], v47_expected).statistic)
            if np.std(features[:, index]) > 1e-12 else 0.0
            for index in range(n_features)
        ]
    )
    max_conditional = np.nanmax(conditional, axis=1)

    # Feature-level heatmaps: keep the most useful diagnostics readable.
    selected = np.argsort(-max_conditional)[: min(args.top_features, n_features)]
    heatmap(
        raw[selected],
        [names[index] for index in selected],
        BOUNDARIES,
        "Raw adjacent-rank separability",
        figures / "01_raw_auc_heatmap.png",
        0.5,
        1.0,
    )
    heatmap(
        conditional[selected],
        [names[index] for index in selected],
        BOUNDARIES,
        "Separability remaining after fold-safe v47 residualization",
        figures / "02_conditional_auc_heatmap.png",
        0.5,
        1.0,
    )
    heatmap(
        np.abs(hard_effect[selected]),
        [names[index] for index in selected],
        BOUNDARIES,
        "Matched hard-pair standardized effect |higher - lower rank|",
        figures / "03_hard_pair_effect_heatmap.png",
        0.0,
        max(0.5, float(np.nanpercentile(np.abs(hard_effect[selected]), 95))),
    )

    # Family-level view.
    families = sorted(set(family(name) for name in names))
    family_matrix = np.full((len(families), 9), np.nan)
    for row, group in enumerate(families):
        members = [index for index, name in enumerate(names) if family(name) == group]
        family_matrix[row] = np.nanmax(conditional[members], axis=0)
    heatmap(
        family_matrix,
        families,
        BOUNDARIES,
        "Best v47-conditioned AUC within each feature family",
        figures / "04_family_conditional_auc_heatmap.png",
        0.5,
        1.0,
    )

    redundancy_scatter(
        names,
        max_conditional,
        corr,
        figures / "05_redundancy_vs_conditional_auc.png",
    )

    for low in TARGET_BOUNDARIES:
        safe = BOUNDARIES[low].replace("/", "_")
        boundary_bar(
            low,
            names,
            raw,
            conditional,
            figures / f"06_{safe}_top_features.png",
        )
        hard_boundary_plot(
            low,
            names,
            features,
            truth,
            fold,
            v47,
            conditional,
            figures / f"07_{safe}_hard_distributions.png",
        )

    # Baseline boundary difficulty.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(9), [separability(value) for value in baseline_auc])
    ax.set_xticks(np.arange(9), BOUNDARIES, rotation=35, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("v47 pair-odds AUC")
    ax.set_title("Which adjacent boundaries remain intrinsically difficult for v47", loc="left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "00_v47_boundary_difficulty.png", dpi=170)
    plt.close(fig)

    # Machine-readable long-form metrics.
    metrics_path = args.output / "feature_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "feature", "family", "boundary", "raw_auc", "conditional_auc",
                "raw_direction", "hard_pair_effect", "hard_pair_ci_low",
                "hard_pair_ci_high", "hard_pair_count", "spearman_v47_expected_rank",
            ),
        )
        writer.writeheader()
        for feature_index, name in enumerate(names):
            for low, boundary_name in enumerate(BOUNDARIES):
                writer.writerow(
                    {
                        "feature": name,
                        "family": family(name),
                        "boundary": boundary_name,
                        "raw_auc": raw[feature_index, low],
                        "conditional_auc": conditional[feature_index, low],
                        "raw_direction": direction[feature_index, low],
                        "hard_pair_effect": hard_effect[feature_index, low],
                        "hard_pair_ci_low": hard_lo[feature_index, low],
                        "hard_pair_ci_high": hard_hi[feature_index, low],
                        "hard_pair_count": int(pair_counts[low]),
                        "spearman_v47_expected_rank": corr[feature_index],
                    }
                )

    top_candidates = []
    for feature_index, name in enumerate(names):
        low = int(np.nanargmax(conditional[feature_index]))
        top_candidates.append(
            {
                "feature": name,
                "boundary": BOUNDARIES[low],
                "conditional_auc": float(conditional[feature_index, low]),
                "raw_auc": float(raw[feature_index, low]),
                "corr_v47": float(corr[feature_index]),
                "hard_effect": float(hard_effect[feature_index, low]),
            }
        )
    top_candidates.sort(
        key=lambda row: row["conditional_auc"] - 0.10 * abs(row["corr_v47"]),
        reverse=True,
    )

    v47_pred = decode(v47)
    gap = np.abs(v47_pred - truth)
    summary = {
        "questions": int(len(truth)),
        "features": int(n_features),
        "v47_score": float(
            np.mean(np.where(gap == 0, 1.0, np.where(gap == 1, np.exp(-1.0), 0.0)))
        ),
        "v47_exact": float((gap == 0).mean()),
        "v47_adjacent_error": float((gap == 1).mean()),
        "v47_far_error": float((gap >= 2).mean()),
        "baseline_boundary_auc": {
            BOUNDARIES[index]: float(separability(value))
            for index, value in enumerate(baseline_auc)
        },
        "hard_pair_counts": {
            BOUNDARIES[index]: int(pair_counts[index]) for index in range(9)
        },
        "top_candidates": top_candidates[:30],
        "interpretation_gate": {
            "interesting_conditional_auc": 0.58,
            "strong_conditional_auc": 0.60,
            "note": (
                "Prefer features with conditional AUC >=0.58-0.60, consistent "
                "matched-pair effect and modest correlation with v47. This is a "
                "diagnostic screen, not a production adoption rule."
            ),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure_names = sorted(path.name for path in figures.glob("*.png"))
    write_html(
        args.output / "report.html",
        summary,
        figure_names,
        top_candidates[:20],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
