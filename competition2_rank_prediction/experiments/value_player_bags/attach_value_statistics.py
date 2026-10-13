"""Appends robust normal-KataGo value summaries to dense bag embeddings.

The dense policy embeddings omit direct value loss and ownership-change
features. This module joins the two original value-feature games per selected
player with the new 18 extra games, verifies exact game-key alignment, and
adds per-game distribution summaries to the 606D frozen representation.

Example:
    c1/.venv/bin/python -m \
      experiments.value_player_bags.attach_value_statistics
"""

import argparse
import pathlib

import numpy as np

from experiments.dense_player_bags import extract_game_embeddings

import rank_lib


PHASES = 4
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SPLITS = ("train", "validation", "holdout")


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns one mean per game and feature, ignoring padded decisions."""
    weights = mask[..., None].astype(np.float32)
    counts = np.maximum(weights.sum(axis=1), 1.0)
    return (values * weights).sum(axis=1) / counts


def _masked_standard_deviation(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Returns one standard deviation per game and feature."""
    centered = values - mean[:, None, :]
    return np.sqrt(_masked_mean(centered * centered, mask) + 1e-8)


def _masked_quantiles(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns requested decision quantiles per game and feature."""
    missing = np.where(mask[..., None], values, np.nan)
    quantiles = np.nanquantile(missing, QUANTILES, axis=1)
    return quantiles.transpose(1, 0, 2).reshape(len(values), -1)


def _phase_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns means for four equal-length phases of each game."""
    positions = np.cumsum(mask, axis=1) - 1
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    parts = []
    for phase in range(PHASES):
        lower = phase * counts / PHASES
        upper = (phase + 1) * counts / PHASES
        phase_mask = mask & (positions >= lower) & (positions < upper)
        parts.append(_masked_mean(values, phase_mask))
    return np.concatenate(parts, axis=1)


def _correlations(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Summarizes decision-quality and ownership-change relationships."""
    pairs = ((0, 8), (0, 9), (0, 10), (8, 10), (9, 10))
    parts = []
    for left, right in pairs:
        left_values = values[:, :, left]
        right_values = values[:, :, right]
        left_mean = _masked_mean(left_values[..., None], mask).squeeze(1)
        right_mean = _masked_mean(right_values[..., None], mask).squeeze(1)
        left_centered = left_values - left_mean[:, None]
        right_centered = right_values - right_mean[:, None]
        covariance = _masked_mean(
            (left_centered * right_centered)[..., None],
            mask,
        ).squeeze(1)
        left_scale = _masked_mean(
            (left_centered * left_centered)[..., None],
            mask,
        ).squeeze(1)
        right_scale = _masked_mean(
            (right_centered * right_centered)[..., None],
            mask,
        ).squeeze(1)
        parts.append(covariance / np.sqrt(left_scale * right_scale + 1e-8))
    return np.stack(parts, axis=1)


def make_statistics(features: dict[str, np.ndarray]) -> np.ndarray:
    """Builds a robust fixed vector from each game's value tokens."""
    values = features["tokens"].astype(np.float32)
    mask = features["masks"].astype(bool)
    mean = _masked_mean(values, mask)
    parts = (
        mean,
        _masked_standard_deviation(values, mask, mean),
        _masked_quantiles(values, mask),
        _phase_means(values, mask),
        _correlations(values, mask),
        mask.sum(axis=1, keepdims=True).astype(np.float32),
    )
    statistics = np.concatenate(parts, axis=1)
    if not np.isfinite(statistics).all():
        raise ValueError("Value statistics contain non-finite values")
    return statistics.astype(np.float16)


def _split_features(
    split: str,
    arguments: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Loads value features aligned with one source embedding split."""
    if split == "train":
        return extract_game_embeddings.merge_training_features(
            arguments.base_train,
            arguments.extra_train,
            arguments.players,
        )
    directory = arguments.validation if split == "validation" else arguments.holdout
    return rank_lib.load_features(directory)


def _write_split(split: str, arguments: argparse.Namespace) -> None:
    """Writes one source embedding split with appended value statistics."""
    features = _split_features(split, arguments)
    source_path = arguments.source / f"{split}.npz"
    with np.load(source_path, allow_pickle=True) as source:
        payload = {key: source[key] for key in source.files}
    game_keys = payload["game_key"].astype(str)
    if not np.array_equal(game_keys, features["game_key"].astype(str)):
        raise ValueError(f"Value and embedding keys differ for {split}")
    payload["embeddings"] = np.concatenate(
        (payload["embeddings"].astype(np.float16), make_statistics(features)),
        axis=1,
    )
    np.savez_compressed(arguments.output / f"{split}.npz", **payload)
    print(f"saved {split}: {payload['embeddings'].shape}", flush=True)


def main() -> None:
    """Creates value-augmented embeddings after extraction completes."""
    parser = argparse.ArgumentParser()
    root = rank_lib.ROOT
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=root / "experiments" / "dense_player_bags" / "embeddings_logits_moves",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "experiments" / "value_player_bags" / "embeddings_logits_moves_value",
    )
    parser.add_argument(
        "--base-train",
        type=pathlib.Path,
        default=root / "features" / "feat_value" / "train",
    )
    parser.add_argument(
        "--extra-train",
        type=pathlib.Path,
        default=experiment / "features",
    )
    parser.add_argument(
        "--players",
        type=pathlib.Path,
        default=root / "experiments" / "dense_player_bags" / "shards" / "players.txt",
    )
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=root / "features" / "feat_value" / "val",
    )
    parser.add_argument(
        "--holdout",
        type=pathlib.Path,
        default=root / "features" / "feat_value" / "holdout",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
        help="Embedding splits to create; omit holdout during model selection.",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for split in arguments.splits:
        _write_split(split, arguments)


if __name__ == "__main__":
    main()
