# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Exploratory follow-up: does soft rank pooling over-weight populous ranks?

v47 pools identity evidence as ``softmax(cosines / T) @ one_hot(class_rank)``,
which sums probability over every gallery class of a rank. The v4 gallery is
unbalanced (A has 1,434 classes, 1D has 358), so a rank with many classes
collects mass for being populous. This scores

    posterior_r  proportional to  n_r ** (-beta) * sum_{c in r} exp(cos_c / T)

where ``beta = 0`` is v47-style pooling and ``beta = 1`` averages within a rank,
i.e. a uniform prior over ranks. T and beta are chosen on the opposite
player-disjoint fold by rank NLL from fixed grids, then evaluated once.

Caveat built into the reading: the control games' true ranks are close to
balanced, so ``beta = 1`` partly matches this sample's prior. Whether it helps
v47 must be tested inside v47, where learned transition and bias terms may
already compensate.

Reads the frozen control scores only (``present_rank_scores.npz``); no training.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
TEMPERATURES = (0.02, 0.03, 0.04, 0.06)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 20261120


def posterior(cosines: np.ndarray, onehot: np.ndarray, counts: np.ndarray,
              temperature: float, beta: float) -> np.ndarray:
    """Rank posterior with an optional class-count correction."""
    logits = cosines / temperature
    logits -= logits.max(1, keepdims=True)
    mass = (np.exp(logits) @ onehot) * counts[None, :] ** (-beta)
    return mass / mass.sum(1, keepdims=True)


def true_nll(probabilities: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean negative log probability of the true rank over masked rows."""
    rows = probabilities[mask][np.arange(int(mask.sum())), truth[mask]]
    return float(-np.log(np.clip(rows, 1e-12, 1.0)).mean())


def cross_fit(cache: dict, truth: np.ndarray, fold: np.ndarray, betas: tuple) -> dict:
    """Select (T, beta) on the opposite fold, evaluate on the held-out fold."""
    prediction = np.zeros(len(truth), dtype=int)
    probability = np.zeros(len(truth))
    chosen = {}
    for held in (0, 1):
        train, test = fold != held, fold == held
        candidates = [(t, b) for t in TEMPERATURES for b in betas]
        losses = [true_nll(cache[k], truth, train) for k in candidates]
        key = candidates[int(np.argmin(losses))]
        chosen[str(held)] = list(key)
        prediction[test] = cache[key][test].argmax(1)
        probability[test] = cache[key][test][np.arange(int(test.sum())), truth[test]]
    correct = prediction == truth
    return {
        "chosen_T_beta_by_fold": chosen,
        "rank_top1": round(float(correct.mean()), 4),
        "rank_nll": round(float(-np.log(np.clip(probability, 1e-12, 1.0)).mean()), 4),
        "within_one": round(float((np.abs(prediction - truth) <= 1).mean()), 4),
        "per_rank_top1": {r: round(float(correct[truth == i].mean()), 3)
                          for i, r in enumerate(RANKS)},
        "true_1D_predicted_A": int(((truth == 4) & (prediction == 3)).sum()),
        "_correct": correct,
    }


def load_scores(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosines, class ranks, true ranks and player ids from the frozen control scores."""
    # NpzFile values are arrays; pylint infers the np.array function.
    with np.load(path) as saved:  # pylint: disable=no-member
        return (
            saved["cosines"].astype(np.float64),  # pylint: disable=no-member
            saved["class_rank"].astype(int),  # pylint: disable=no-member
            saved["true_rank"].astype(int),  # pylint: disable=no-member
            saved["player_id"].astype(str),  # pylint: disable=no-member
        )


def build_cache(cosines: np.ndarray, class_rank: np.ndarray) -> dict:
    """Rank posteriors for every (temperature, beta) pair on the fixed grids."""
    counts = np.bincount(class_rank, minlength=10).astype(float)
    onehot = np.eye(10)[class_rank]
    return {(t, b): posterior(cosines, onehot, counts, t, b)
            for t in TEMPERATURES for b in BETAS}


def cluster_bootstrap(diff: np.ndarray, inverse: np.ndarray, rng: np.random.Generator) -> list:
    """95% interval of the game-weighted mean, resampling player clusters."""
    sums = np.bincount(inverse, weights=diff)
    sizes = np.bincount(inverse).astype(float)
    pick = rng.integers(0, len(sums), size=(5000, len(sums)))
    boots = sums[pick].sum(1) / sizes[pick].sum(1)
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def main() -> None:  # pylint: disable=too-many-locals
    """Prints the uncorrected-vs-corrected comparison as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    cosines, class_rank, truth, players = load_scores(args.scores)
    unique, inverse = np.unique(players, return_inverse=True)
    rng = np.random.default_rng(SEED)
    fold = (rng.permutation(len(unique)) % 2)[inverse]
    cache = build_cache(cosines, class_rank)
    counts = np.bincount(class_rank, minlength=10)

    uncorrected = cross_fit(cache, truth, fold, (0.0,))
    corrected = cross_fit(cache, truth, fold, BETAS)
    diff = corrected.pop("_correct").astype(float) - uncorrected.pop("_correct").astype(float)
    report = {
        "gallery_classes_per_rank": dict(zip(RANKS, counts.tolist())),
        "control_true_rank_counts": dict(zip(RANKS, np.bincount(truth, minlength=10).tolist())),
        "uncorrected_temperature_only": uncorrected,
        "count_corrected": corrected,
        "corrected_minus_uncorrected_rank_top1": {
            "delta": round(float(diff.mean()), 4),
            "ci95": cluster_bootstrap(diff, inverse, rng),
            "players": int(len(unique)),
            "bootstrap": "player clusters, game-weighted",
        },
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
