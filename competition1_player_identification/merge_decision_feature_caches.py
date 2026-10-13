"""Merges stable NBT tokens with supplemental decision-token feature columns.

Example:
    .venv/bin/python merge_decision_feature_caches.py \
      --base features/main/ownership_400x100B_q20B_s32 \
      --supplement features/ablation/humansl_rank_5k_1d_5d \
      --feature-prefix human_ --output features/ablation/humansl_merged
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np


CACHE_PARTS = ("candidates", "queries")


@dataclasses.dataclass(frozen=True)
class FeatureCacheArrays:
    """Token arrays and metadata loaded from one compressed cache part."""

    tokens: np.ndarray
    masks: np.ndarray
    player_indices: np.ndarray
    feature_names: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    """Parses two compatible feature caches and an output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--supplement", type=pathlib.Path, required=True)
    parser.add_argument("--feature-prefix", required=True)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("candidates", "queries", "train"),
        default=CACHE_PARTS,
        help="Cache files to merge; use train for a training-only cache.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_feature_cache(path: pathlib.Path) -> FeatureCacheArrays:
    """Loads one compressed cache file into a typed in-memory container."""
    with np.load(path) as raw_cache:
        return FeatureCacheArrays(
            raw_cache["tokens"],
            raw_cache["masks"],
            raw_cache["player_indices"],
            tuple(
                str(name)
                for name in np.asarray(raw_cache["feature_names"]).reshape(-1)
            ),
        )


def merge_feature_arrays(
    base: FeatureCacheArrays,
    supplement: FeatureCacheArrays,
    feature_prefix: str,
) -> FeatureCacheArrays:
    """Appends prefixed columns after verifying game identity and alignment."""
    if not feature_prefix:
        raise ValueError("feature_prefix must be non-empty")
    _validate_aligned_cache_arrays(base, supplement)
    supplement_indices = [
        index
        for index, name in enumerate(supplement.feature_names)
        if name.startswith(feature_prefix)
    ]
    if not supplement_indices:
        raise ValueError(f"Supplement cache has no {feature_prefix!r} features")
    supplement_names = tuple(
        supplement.feature_names[index]
        for index in supplement_indices
    )
    duplicated_names = set(base.feature_names) & set(supplement_names)
    if duplicated_names:
        raise ValueError(
            f"Supplement features duplicate base names: {sorted(duplicated_names)!r}",
        )
    merged_tokens = np.concatenate(
        (base.tokens, supplement.tokens[:, :, supplement_indices]),
        axis=2,
    )
    return FeatureCacheArrays(
        merged_tokens,
        base.masks,
        base.player_indices,
        base.feature_names + supplement_names,
    )


def _validate_aligned_cache_arrays(
    base: FeatureCacheArrays,
    supplement: FeatureCacheArrays,
) -> None:
    """Ensures no decision data can be paired with a different game."""
    if base.tokens.shape[:2] != supplement.tokens.shape[:2]:
        raise ValueError(
            "Base and supplement token game/turn shapes differ: "
            f"{base.tokens.shape!r} versus {supplement.tokens.shape!r}",
        )
    if not np.array_equal(base.masks, supplement.masks):
        raise ValueError("Base and supplement masks differ")
    if not np.array_equal(base.player_indices, supplement.player_indices):
        raise ValueError("Base and supplement player labels differ")


def write_feature_cache(path: pathlib.Path, cache: FeatureCacheArrays) -> None:
    """Writes one merged candidate or query cache file."""
    np.savez_compressed(
        path,
        tokens=cache.tokens,
        masks=cache.masks,
        player_indices=cache.player_indices,
        feature_names=np.asarray(cache.feature_names),
    )


def make_metadata(
    base_path: pathlib.Path,
    supplement_path: pathlib.Path,
    feature_prefix: str,
    feature_names: tuple[str, ...],
) -> dict[str, object]:
    """Records the provenance of the fixed-base and supplemental features."""
    with (base_path / "metadata.json").open(encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    metadata["base_feature_cache"] = str(base_path)
    metadata["supplement_feature_cache"] = str(supplement_path)
    metadata["supplement_feature_prefix"] = feature_prefix
    metadata["features"] = feature_names
    return metadata


def main() -> None:
    """Creates a provenance-preserving cache with appended feature columns."""
    arguments = parse_arguments()
    if arguments.output.exists():
        raise FileExistsError(f"Output directory already exists: {arguments.output!s}")
    arguments.output.mkdir(parents=True)
    merged_feature_names: tuple[str, ...] | None = None
    for part in arguments.parts:
        base = read_feature_cache(arguments.base / f"{part}.npz")
        supplement = read_feature_cache(arguments.supplement / f"{part}.npz")
        merged = merge_feature_arrays(base, supplement, arguments.feature_prefix)
        if merged_feature_names is None:
            merged_feature_names = merged.feature_names
        elif merged.feature_names != merged_feature_names:
            raise ValueError("Selected feature-cache parts have different schemas")
        write_feature_cache(arguments.output / f"{part}.npz", merged)
    if merged_feature_names is None:
        raise RuntimeError("No feature-cache parts were merged")
    metadata = make_metadata(
        arguments.base,
        arguments.supplement,
        arguments.feature_prefix,
        merged_feature_names,
    )
    with (arguments.output / "metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)


if __name__ == "__main__":
    main()
