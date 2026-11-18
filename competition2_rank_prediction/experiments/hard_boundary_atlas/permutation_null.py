"""Label-permutation null for the hard-boundary atlas.

The atlas reports a direction-free conditional AUC, ``max(AUC, 1 - AUC)``, per
feature and boundary, then takes the best over boundaries and over a family.
With about 200 questions per rank on holdout3, each of those maxima is biased
upward even when a feature carries no information, so a raw cut such as 0.58
cannot be read without a null.

The residuals are built without the labels (fold-safe linear residual against
the v47 boundary log-odds), so permuting the labels within each fold while
holding the residuals fixed is a valid null for "this residual carries no
registered-rank information". The same permutation is applied to every feature
of a boundary, so the family-wise maximum keeps the correlation between
features.

Reported per boundary:

* each cell's permutation p-value and Benjamini-Hochberg q-value over every
  feature x boundary cell;
* each family's observed best conditional AUC against the 95th percentile of
  that family's permuted best (a family-wise check);
* the nonparametric matched-pair effect from the atlas, for comparison.

CPU only; reads ``feature_table.npz`` and trains nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.hard_boundary_atlas.analyze import (
    RANKS,
    baseline_boundary_score,
    conditional_residual,
    family,
)

SEED = 20261118


def binned_residual(
    feature: np.ndarray,
    baseline_score: np.ndarray,
    boundary_mask: np.ndarray,
    fold: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Fold-safe nonparametric residual: subtract the feature's mean within
    quantile bins of the v47 boundary score, with bin edges and means fitted on
    the opposite fold only. Removes any monotone or non-monotone dependence on
    v47 that a straight line cannot."""
    result = np.full(len(feature), np.nan, np.float64)
    for target_fold in (0, 1):
        train = np.flatnonzero(boundary_mask & (fold != target_fold))
        test = np.flatnonzero(boundary_mask & (fold == target_fold))
        if len(train) < bins or test.size == 0:
            continue
        edges = np.quantile(baseline_score[train], np.linspace(0, 1, bins + 1)[1:-1])
        train_bin = np.searchsorted(edges, baseline_score[train])
        test_bin = np.searchsorted(edges, baseline_score[test])
        means = np.array(
            [
                feature[train][train_bin == b].mean()
                if np.any(train_bin == b) else feature[train].mean()
                for b in range(bins)
            ]
        )
        result[test] = feature[test] - means[test_bin]
    return result


def permuted_labels(
    y: np.ndarray, fold: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """[draws, n] label matrices, each shuffled within fold."""
    out = np.empty((draws, len(y)), dtype=np.float64)
    for draw in range(draws):
        shuffled = y.copy()
        for which in np.unique(fold):
            members = np.flatnonzero(fold == which)
            shuffled[members] = rng.permutation(y[members])
        out[draw] = shuffled
    return out


def separability_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """max(AUC, 1 - AUC) for every label row and every feature column."""
    positives = labels.sum(1, keepdims=True)
    negatives = labels.shape[1] - positives
    rank_sum = labels @ ranks
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return np.maximum(auc, 1.0 - auc)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH q-values for a flat array of p-values."""
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(q_sorted)
    q_values[order] = np.minimum(q_sorted, 1.0)
    return q_values


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Writes the permutation-null report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=pathlib.Path, required=True)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument(
        "--residual", choices=("linear", "binned"), default="linear",
        help="linear reproduces the atlas; binned removes nonlinear v47 dependence",
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    # NpzFile values are arrays; pylint infers the np.array function.
    # pylint: disable=no-member
    with np.load(args.table, allow_pickle=True) as saved:
        truth = saved["truth"].astype(np.int64)
        fold = saved["fold"].astype(np.int64)
        v47 = saved["v47_prob"].astype(np.float64)
        names = saved["feature_names"].astype(str).tolist()
        features = saved["features"].astype(np.float64)
    # pylint: enable=no-member
    families = [family(name) for name in names]
    family_index: dict[str, list[int]] = collections.defaultdict(list)
    for index, name in enumerate(families):
        family_index[name].append(index)

    hard_effect: dict[tuple[str, str], tuple[float, float, float]] = {}
    with args.metrics.open(encoding="utf-8") as source:
        header = source.readline().strip().split(",")
        for line in source:
            row = dict(zip(header, line.strip().split(",")))
            hard_effect[(row["feature"], row["boundary"])] = (
                float(row["hard_pair_effect"] or "nan"),
                float(row["hard_pair_ci_low"] or "nan"),
                float(row["hard_pair_ci_high"] or "nan"),
            )

    rng = np.random.default_rng(SEED)
    cells = []
    family_rows = []
    for low in range(9):
        boundary_name = f"{RANKS[low]}/{RANKS[low + 1]}"
        boundary = (truth == low) | (truth == low + 1)
        y = (truth[boundary] == low + 1).astype(np.float64)
        score = baseline_boundary_score(v47, low)
        residual = np.stack(
            [
                (
                    binned_residual(features[:, index], score, boundary, fold, args.bins)
                    if args.residual == "binned"
                    else conditional_residual(features[:, index], score, boundary, fold)
                )[boundary]
                for index in range(len(names))
            ],
            axis=1,
        )
        usable = np.isfinite(residual).all(0) & (residual.std(0) > 1e-12)
        ranks = np.zeros_like(residual)
        ranks[:, usable] = np.apply_along_axis(rankdata, 0, residual[:, usable])

        observed = separability_from_ranks(ranks, y[None])[0]
        null = separability_from_ranks(
            ranks, permuted_labels(y, fold[boundary], args.draws, rng)
        )
        observed[~usable] = np.nan
        null[:, ~usable] = np.nan

        p_values = (1 + (null >= observed[None]).sum(0)) / (1 + args.draws)
        for index, name in enumerate(names):
            if not usable[index]:
                continue
            effect, lo, hi = hard_effect.get((name, boundary_name), (np.nan,) * 3)
            cells.append(
                {
                    "feature": name,
                    "family": families[index],
                    "boundary": boundary_name,
                    "conditional_auc": round(float(observed[index]), 4),
                    "null_q95": round(float(np.nanquantile(null[:, index], 0.95)), 4),
                    "p": float(p_values[index]),
                    "hard_pair_effect": round(effect, 4),
                    "hard_pair_ci": [round(lo, 4), round(hi, 4)],
                }
            )
        for name, members in family_index.items():
            members_usable = [m for m in members if usable[m]]
            if not members_usable:
                continue
            best = float(np.nanmax(observed[members_usable]))
            best_null = np.nanmax(null[:, members_usable], axis=1)
            family_rows.append(
                {
                    "family": name,
                    "boundary": boundary_name,
                    "features": len(members_usable),
                    "best_conditional_auc": round(best, 4),
                    "family_null_q95": round(float(np.quantile(best_null, 0.95)), 4),
                    "family_null_mean": round(float(best_null.mean()), 4),
                    "family_p": float((1 + (best_null >= best).sum()) / (1 + args.draws)),
                }
            )

    q_values = benjamini_hochberg(np.asarray([cell["p"] for cell in cells]))
    for cell, q_value in zip(cells, q_values):
        cell["q_bh"] = round(float(q_value), 4)
        cell["p"] = round(cell["p"], 4)

    survivors = sorted(
        (cell for cell in cells if cell["q_bh"] < 0.10),
        key=lambda cell: cell["q_bh"],
    )
    report = {
        "draws": args.draws,
        "residual": args.residual,
        "cell_count": len(cells),
        "single_cell_null_mean_q95": round(
            float(np.mean([cell["null_q95"] for cell in cells])), 4
        ),
        "family_wise": family_rows,
        "cells_with_bh_q_below_0_10": survivors,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cells"}, indent=1))


if __name__ == "__main__":
    main()
