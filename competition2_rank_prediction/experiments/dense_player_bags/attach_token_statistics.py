"""Appends lossless decision-distribution summaries to dense embeddings.

The frozen game embedding is useful but intentionally compresses 96 decisions
into 384 values. This utility preserves inexpensive distributional signals:
mean, standard deviation, quartiles, four game phases, and relative HumanSL
policy likelihoods. It performs no KataGo inference.

Example:
    c1/.venv/bin/python -m \
        experiments.dense_player_bags.attach_token_statistics
"""

import argparse
import pathlib

import numpy as np

import rank_lib
from experiments.dense_player_bags import extract_game_embeddings

SPLITS = ("train", "validation", "holdout")
QUANTILES = (0.25, 0.50, 0.75)
PHASES = 4


def masked_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Computes a decision-axis mean for every game and feature."""
    weights = mask[..., None].astype(np.float32)
    count = np.maximum(weights.sum(axis=1), 1.0)
    return (values * weights).sum(axis=1) / count


def phase_means(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Returns four equally sized phase means using each game's valid length."""
    positions = np.cumsum(mask, axis=1) - 1
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    parts = []
    for phase in range(PHASES):
        lower = phase * counts / PHASES
        upper = (phase + 1) * counts / PHASES
        selected = mask & (positions >= lower) & (positions < upper)
        parts.append(masked_mean(values, selected))
    return np.concatenate(parts, axis=1)


def masked_quantiles(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Returns per-game feature quartiles, ignoring padded decisions."""
    missing = np.where(mask[..., None], values, np.nan)
    statistics = np.nanquantile(missing, QUANTILES, axis=1)
    return statistics.transpose(1, 0, 2).reshape(len(values), -1)


def human_log_likelihood(
    tokens: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Summarizes each HumanSL profile's relative policy likelihood."""
    surprisals = tokens[:, :, 1:, 0]
    shifted = -surprisals - (-surprisals).max(axis=2, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    log_posterior = shifted - log_normalizer
    mean = masked_mean(log_posterior, mask)
    centered = log_posterior - mean[:, None, :]
    variance = masked_mean(centered * centered, mask)
    return np.concatenate((mean, np.sqrt(variance + 1e-8)), axis=1)


def make_statistics(features: dict[str, np.ndarray]) -> np.ndarray:
    """Builds fixed-length raw summaries from one dense feature split."""
    tokens = features["tokens"].astype(np.float32)
    mask = features["masks"].astype(bool)
    flattened = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
    mean = masked_mean(flattened, mask)
    centered = flattened - mean[:, None, :]
    variance = masked_mean(centered * centered, mask)
    result = np.concatenate(
        (
            mean,
            np.sqrt(variance + 1e-8),
            masked_quantiles(flattened, mask),
            phase_means(flattened, mask),
            human_log_likelihood(tokens, mask),
        ),
        axis=1,
    )
    if not np.isfinite(result).all():
        raise ValueError("Decision statistics contain non-finite values")
    return result.astype(np.float16)


def split_features(
    name: str,
    arguments: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Loads raw decision features in the same order as one embedding split."""
    if name == "train":
        return extract_game_embeddings.merge_training_features(
            arguments.base_train,
            arguments.extra_train,
            arguments.players,
        )
    directory = arguments.validation if name == "validation" else arguments.holdout
    return rank_lib.load_features(directory)


def write_split(
    name: str,
    arguments: argparse.Namespace,
) -> None:
    """Writes one source embedding split with appended raw statistics."""
    features = split_features(name, arguments)
    source = arguments.source / f"{name}.npz"
    with np.load(source, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    keys = payload["game_key"].astype(str)
    if not np.array_equal(keys, features["game_key"].astype(str)):
        raise ValueError(f"Raw feature and embedding game keys differ for {name}")
    statistics = make_statistics(features)
    payload["embeddings"] = np.concatenate(
        (payload["embeddings"].astype(np.float16), statistics),
        axis=1,
    )
    np.savez_compressed(arguments.output / f"{name}.npz", **payload)
    print(f"saved {name}: {payload['embeddings'].shape}", flush=True)


def main() -> None:
    """Creates style, rank-logit, and raw-decision-statistic embeddings."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=experiment / "embeddings_logits",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=experiment / "embeddings_logits_stats",
    )
    parser.add_argument(
        "--base-train",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "train",
    )
    parser.add_argument(
        "--extra-train",
        type=pathlib.Path,
        default=experiment / "features",
    )
    parser.add_argument(
        "--players",
        type=pathlib.Path,
        default=experiment / "shards" / "players.txt",
    )
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "val",
    )
    parser.add_argument(
        "--holdout",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "holdout",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_split(name, arguments)


if __name__ == "__main__":
    main()
