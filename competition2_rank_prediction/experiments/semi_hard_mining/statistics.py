"""Shares player-level reward, bootstrap and screening calculations.

For multi-seed analyses aggregate seed differences before bootstrapping.
"""

import pathlib
import sys

import numpy as np


# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


def player_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Returns one realized competition reward per player."""
    gap = np.abs(rank_lib.decode(probabilities) - labels)
    return np.where(gap == 0, 1, np.where(gap == 1, np.exp(-1), 0))


def bootstrap_players(
    differences: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Resamples a one-dimensional vector of paired player differences."""
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("Expected one nonempty difference vector per player")
    if not 0 < confidence < 1:
        raise ValueError(
            "Bootstrap confidence must be strictly between zero and one"
        )
    generator = np.random.default_rng(20261108)
    means = []
    for _ in range(5000):
        rows = generator.integers(0, len(differences), len(differences))
        means.append(float(differences[rows].mean()))
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, (tail, 1 - tail))
    return {
        "delta": float(differences.mean()),
        "low": float(low),
        "high": float(high),
    }


def screening_conditions(
    baseline: dict[str, float],
    candidate: dict[str, float],
    worst_rank_delta: float,
) -> dict[str, bool]:
    """Returns the fixed utility/AUC and error safety criteria."""
    return {
        "utility_positive": candidate["utility"] > baseline["utility"],
        "pair_auc_positive": candidate["pair_auc"] > baseline["pair_auc"],
        "large_errors_safe": candidate["large_error"] - baseline["large_error"]
        <= 0.01,
        "rank_losses_safe": worst_rank_delta >= -0.05,
    }
