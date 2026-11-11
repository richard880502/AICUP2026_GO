"""Computes v2 retrieval scores and player-bootstrap confidence intervals.

Both mean-centroid and game-to-game scores use exactly ten reference games.
Final-holdout scores must only be generated after model selection is locked.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional

COUNTS = (
    1,
    2,
    3,
    5,
)


def save_scores(path: pathlib.Path, scores: dict) -> None:
    """Stores fixed-aggregator matrices without changing their query order."""
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            centroid=scores["centroid"],
            pair_mean=scores["pair_mean"],
            pair_top3=scores["pair_top3"],
            pair_lme=scores["pair_lme"],
        )


def similarity_scores(embeddings: np.ndarray, device: str) -> dict:
    """Scores aligned [600, 15, D] game embeddings with fixed aggregators."""
    values = torch.from_numpy(embeddings).to(device)
    values = torch.nn.functional.normalize(values, dim=-1)
    references = values[:, :10]
    queries = values[:, 10:]
    centroids = torch.nn.functional.normalize(references.mean(1), dim=-1)
    pairs = (
        queries.reshape(-1, values.shape[-1])
        @ references.reshape(
            -1,
            values.shape[-1],
        ).T
    ).reshape(
        len(values),
        5,
        len(values),
        10,
    )
    results: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "centroid",
            "pair_mean",
            "pair_top3",
            "pair_lme",
        )
    }
    for count in COUNTS:
        query_centroids = torch.nn.functional.normalize(
            queries[:, :count].mean(1),
            dim=-1,
        )
        results["centroid"].append(
            (query_centroids @ centroids.T).cpu().numpy()
        )
        subset = pairs[:, :count]
        results["pair_mean"].append(subset.mean((1, 3)).cpu().numpy())
        results["pair_top3"].append(
            subset.topk(3, dim=-1).values.mean((1, 3)).cpu().numpy(),
        )
        results["pair_lme"].append(
            (torch.logsumexp(subset * 10, dim=-1).mean(1) / 10).cpu().numpy()
        )
    return {name: np.stack(scores) for name, scores in results.items()}


def correct_ranks(scores: np.ndarray) -> np.ndarray:
    """Finds true ranks with stable index tie-breaking for every query size."""
    order = np.argsort(
        -scores,
        axis=-1,
        kind="stable",
    )
    labels = np.arange(scores.shape[-1])[None, :, None]
    return (np.argmax(order == labels, axis=-1) + 1).astype(np.int32)


def summarize(scores: np.ndarray) -> dict:
    """Reports the exponential Top-5 metric, recall, and clustered uncertainty."""
    ranks = correct_ranks(scores)
    rewards = np.where(
        ranks <= 5,
        np.exp(-(ranks - 1)),
        0.0,
    )
    generator = np.random.default_rng(20261111)
    samples = generator.integers(
        0,
        ranks.shape[1],
        (2000, ranks.shape[1]),
    )
    interval = np.quantile(rewards.mean(0)[samples].mean(1), (0.025, 0.975))
    return {
        "score": float(rewards.mean()),
        "player_bootstrap_95ci": interval.tolist(),
        "by_query_count": {
            str(count): {
                "score": float(rewards[index].mean()),
                "top1": float((ranks[index] == 1).mean()),
                "top5": float((ranks[index] <= 5).mean()),
                "top20": float((ranks[index] <= 20).mean()),
            }
            for index, count in enumerate(COUNTS)
        },
    }


def paired_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict:
    """Bootstraps paired player-level differences, preserving nested queries."""
    ranks = correct_ranks(candidate)
    old_ranks = correct_ranks(baseline)
    differences = (
        np.where(
            ranks <= 5,
            np.exp(-(ranks - 1)),
            0.0,
        )
        - np.where(
            old_ranks <= 5,
            np.exp(-(old_ranks - 1)),
            0.0,
        )
    ).mean(0)
    generator = np.random.default_rng(20261111)
    indices = generator.integers(
        0,
        len(differences),
        (2000, len(differences)),
    )
    return {
        "delta": float(differences.mean()),
        "player_bootstrap_95ci": np.quantile(
            differences[indices].mean(1),
            (0.025, 0.975),
        ).tolist(),
    }


def color_subgroups(scores: np.ndarray, colors: np.ndarray) -> dict:
    """Reports query-color subgroups without refitting or altering any scores."""
    ranks = correct_ranks(scores)
    report: dict[str, dict] = {}
    for position, count in enumerate(COUNTS):
        black = (colors[:, :count] == 0).all(1)
        white = (colors[:, :count] == 1).all(1)
        groups = {
            "all_black": black,
            "all_white": white,
            "mixed": ~(black | white),
        }
        report[str(count)] = {}
        for name, mask in groups.items():
            chosen = ranks[position, mask]
            report[str(count)][name] = {
                "questions": len(chosen),
                "score": (
                    float(
                        np.where(
                            chosen <= 5,
                            np.exp(-(chosen - 1)),
                            0.0,
                        ).mean()
                    )
                    if len(chosen)
                    else None
                ),
                "top1": float((chosen == 1).mean()) if len(chosen) else None,
                "top5": float((chosen <= 5).mean()) if len(chosen) else None,
            }
    return report
