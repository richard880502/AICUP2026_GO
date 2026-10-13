"""Materializes reusable rich game profiles from a decision-token cache.

The decision cache contains KataGo features per sampled move.  Building the
multi-scale robust profile is relatively expensive, so downstream experiments
such as supervised UMAP and discriminative classifiers should consume this
artifact rather than repeat the aggregation.

Example:
    .venv/bin/python cache_rich_profiles.py \
      --features features/independent_s64_rank5k_merged \
      --output features/independent_s64_rank5k_profiles.npz
"""

import argparse
import concurrent.futures
import json
import pathlib

import numpy as np

import evaluate_decision_features as decision_features


DROP_FEATURES = ("winrate_drop", "score_lead_drop_scaled")


def parse_arguments() -> argparse.Namespace:
    """Parses the source decision-feature directory and cache destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel profile chunks; use 1 for the reference implementation.",
    )
    parser.add_argument(
        "--phase-scales",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32),
        help="Temporal phase counts concatenated into each game profile.",
    )
    return parser.parse_args()


def make_profile_config(
    phase_scales: tuple[int, ...],
) -> decision_features.ProfileConfig:
    """Builds one rich profile configuration with requested temporal scales."""
    return decision_features.ProfileConfig(
        representation="multi_scale_stats",
        phase_scales=phase_scales,
        include_phase_std=True,
        include_robust_stats=True,
        include_interactions=True,
    )


def make_profile_chunk(
    tokens: np.ndarray,
    masks: np.ndarray,
    feature_names: tuple[str, ...],
    profile_config: decision_features.ProfileConfig,
) -> np.ndarray:
    """Builds rich profiles for one independent game chunk."""
    return decision_features.make_feature_stat_profiles(
        tokens,
        masks,
        feature_names,
        profile_config,
    )


def make_profiles(
    tokens: np.ndarray,
    masks: np.ndarray,
    feature_names: tuple[str, ...],
    workers: int,
    profile_config: decision_features.ProfileConfig,
) -> np.ndarray:
    """Builds independent game profiles in bounded parallel chunks."""
    if workers == 1:
        return make_profile_chunk(
            tokens,
            masks,
            feature_names,
            profile_config,
        )
    indices = np.array_split(np.arange(len(tokens)), workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk_indices in indices:
            futures.append(executor.submit(
                make_profile_chunk,
                tokens[chunk_indices],
                masks[chunk_indices],
                feature_names,
                profile_config,
            ))
    return np.concatenate([future.result() for future in futures])


def load_profile_file(
    path: pathlib.Path,
    workers: int,
    profile_config: decision_features.ProfileConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads one feature cache and builds the selected rich game profiles."""
    tokens, masks, labels, names = decision_features.read_feature_cache(path)
    selected_indices = decision_features.select_feature_indices(names, DROP_FEATURES)
    selected_names = tuple(names[index] for index in selected_indices)
    profiles = make_profiles(
        tokens[:, :, selected_indices],
        masks,
        selected_names,
        workers,
        profile_config,
    )
    return np.nan_to_num(profiles).astype(np.float32), labels


def load_profiles(
    feature_root: pathlib.Path,
    workers: int,
    profile_config: decision_features.ProfileConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads candidate and query profiles with the selected feature schema."""
    candidates, candidate_labels = load_profile_file(
        feature_root / "candidates.npz",
        workers,
        profile_config,
    )
    queries, query_labels = load_profile_file(
        feature_root / "queries.npz",
        workers,
        profile_config,
    )
    return (
        candidates,
        candidate_labels,
        queries,
        query_labels,
    )


def write_profiles(
    output_path: pathlib.Path,
    candidates: np.ndarray,
    candidate_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
) -> None:
    """Writes one compressed, self-contained profile artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_profiles=candidates,
        candidate_labels=candidate_labels,
        query_profiles=queries,
        query_labels=query_labels,
    )


def write_metadata(
    output_path: pathlib.Path,
    input_root: pathlib.Path,
    profile_config: decision_features.ProfileConfig,
) -> None:
    """Writes the profile schema needed to validate downstream experiments."""
    metadata = {
        "source_features": str(input_root),
        "dropped_features": list(DROP_FEATURES),
        "profile_config": {
            "representation": "multi_scale_stats",
            "phase_scales": list(profile_config.phase_scales),
            "include_phase_std": True,
            "include_robust_stats": True,
            "include_interactions": True,
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Builds the reusable profile cache and reports its dimensions."""
    arguments = parse_arguments()
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")
    phase_scales = tuple(arguments.phase_scales)
    if not phase_scales:
        raise ValueError("phase-scales must not be empty")
    profile_config = make_profile_config(phase_scales)
    candidates, candidate_labels, queries, query_labels = load_profiles(
        arguments.features,
        arguments.workers,
        profile_config,
    )
    write_profiles(
        arguments.output,
        candidates,
        candidate_labels,
        queries,
        query_labels,
    )
    write_metadata(arguments.output, arguments.features, profile_config)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "candidate_shape": list(candidates.shape),
                "query_shape": list(queries.shape),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
