"""Training-free score corrections and the label-free repeated-target check.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition1_player_identification.experiments.trunk_style.score_corrections \
      structure
    ... score_corrections repeats

structure: counts repeated top-1 candidates among the official questions and
compares with two simulations on the final cohort (targets distinct, or
drawn with replacement). No labels are read.
repeats: CSLS and Sinkhorn balancing on cohort episodes (400 questions of
600 candidates, 100 for each of 1/2/3/5 query games) when some targets
repeat.
"""

import argparse

import numpy as np

from competition1_player_identification.experiments.trunk_style import rerank
from competition1_player_identification.experiments.trunk_style import train

COUNTS = rerank.COUNTS


def unit(values: np.ndarray) -> np.ndarray:
    """Scales vectors to unit length."""
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-9)


def standardize(scores: np.ndarray) -> np.ndarray:
    """Standardizes every row (question) of a score matrix."""
    return (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True)


def csls(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Row-standardizes, then subtracts each row's and column's top-k mean."""
    scores = standardize(scores)
    row = np.sort(scores, 1)[:, -k:].mean(1, keepdims=True)
    return 2 * scores - row - np.sort(scores, 0)[-k:].mean(0, keepdims=True)


def sinkhorn(scores: np.ndarray, temperature: float, rounds: int) -> np.ndarray:
    """Alternating row / column normalization of standardized scores."""
    logits = standardize(scores) / temperature
    for _ in range(rounds):
        logits = logits - np.logaddexp.reduce(logits, 1, keepdims=True)
        logits = logits - np.logaddexp.reduce(logits, 0, keepdims=True)
    return logits


METHODS = {
    "base": lambda s: s,
    "csls5": csls,
    "sink_t0.3_n30": lambda s: sinkhorn(s, 0.3, 30),
    "sink_t0.3_n1": lambda s: sinkhorn(s, 0.3, 1),
    "sink_t0.5_n30": lambda s: sinkhorn(s, 0.5, 30),
    "sink_t0.5_n1": lambda s: sinkhorn(s, 0.5, 1),
}


def episode_scores(embeddings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Centroid cosine [400,600]; 100 targets for each of 1/2/3/5 query games."""
    centroids = unit(embeddings[:, :10].mean(1))
    blocks = [
        unit(embeddings[rows, 10 : 10 + count].mean(1)) @ centroids.T
        for count, rows in zip(COUNTS, targets.reshape(4, 100))
    ]
    return np.concatenate(blocks)


def repeat_share(scores: np.ndarray) -> tuple[float, float]:
    """Repeated top-1 share among all questions and the confident half."""
    order = np.sort(scores, 1)
    margin = order[:, -1] - order[:, -2]
    top = scores.argmax(1)
    confident = top[margin >= np.median(margin)]
    return (
        1 - len(np.unique(top)) / len(top),
        1 - len(np.unique(confident)) / len(confident),
    )


def structure(embeddings: np.ndarray) -> None:
    """Prints repeat shares of the official pools and of two simulations."""

    def rounded(scores: np.ndarray) -> tuple:
        return tuple(round(share, 4) for share in repeat_share(scores))

    for name in ("pub2", "priv2"):
        path = train.ROOT / "submission_style_v4" / name / "scores.npz"
        with np.load(path) as arrays:
            style = np.asarray(arrays["style"])
            print(name, style.shape, (len(style),) + rounded(style))
    for label, replace in (("distinct targets ", False), ("with replacement ", True)):
        shares = [
            rounded(
                episode_scores(
                    embeddings,
                    np.random.default_rng(seed).choice(600, 400, replace=replace),
                )
            )
            for seed in range(4)
        ]
        print(label, shares)


def repeats(embeddings: np.ndarray, count: int) -> tuple[np.ndarray, dict]:
    """Mean repeat shares and utility of every method with `count` repeats."""
    totals: dict = {name: [] for name in METHODS}
    shares = []
    for seed in range(20):
        targets = rerank.repeated_targets(seed, count)
        scores = episode_scores(embeddings, targets)
        shares.append(repeat_share(scores))
        for name, method in METHODS.items():
            corrected = method(scores)
            own = corrected[np.arange(400), targets]
            ranks = (corrected > own[:, None]).sum(1)
            totals[name].append(np.where(ranks < 5, np.exp(-ranks), 0.0).mean())
    return np.mean(shares, 0), {k: float(np.mean(v)) for k, v in totals.items()}


def main() -> None:
    """Prints the selected check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("structure", "repeats"))
    arguments = parser.parse_args()
    cohorts = np.load(rerank.RERANK / "cohorts_style_v4.npz")
    if arguments.check == "structure":
        structure(cohorts["final"])
        return
    for split in ("validation", "final"):
        print(split)
        for count in (0, 10, 20, 40, 80):
            shares, result = repeats(cohorts[split], count)
            print(
                f" repeats {count:3d} share_all {shares[0]:.3f} "
                f"share_conf {shares[1]:.3f} | "
                + " ".join(f"{k} {v:.4f}" for k, v in result.items())
            )


if __name__ == "__main__":
    main()
