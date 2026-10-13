"""Scores cached KataGo decision features with centroid cosine retrieval.

Example:
    .venv/bin/python evaluate_decision_features.py \
      --features experiments/10_katago_decision/features/pilot_32x10_q5_s4 \
      --output experiments/10_katago_decision/results/pilot_metrics.json
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from go_retrieval import result_io


ROBUST_FEATURE_NAMES = (
    "policy_surprisal",
    "log_policy_rank",
    "ownership_change",
)
INTERACTION_FEATURE_PAIRS = (
    ("policy_surprisal", "ownership_change"),
    ("log_policy_rank", "player_score_lead_scaled"),
    ("policy_entropy", "policy_surprisal"),
)


@dataclasses.dataclass(frozen=True)
class ProfileConfig:
    """Controls construction of a per-game decision profile."""

    representation: str = "flatten"
    phase_bins: int = 4
    include_phase_std: bool = False
    phase_scales: tuple[int, ...] = (1, 2, 4, 8)
    include_robust_stats: bool = False
    include_interactions: bool = False


@dataclasses.dataclass(frozen=True)
class FeatureWeightingConfig:
    """Controls global feature reliability weighting for cosine retrieval."""

    method: str = "uniform"
    fisher_exponent: float = 0.5
    fisher_minimum: float = 0.01
    fisher_maximum: float = 20.0


@dataclasses.dataclass(frozen=True)
class GaussianScoreConfig:
    """Controls candidate-distribution matching with diagonal Gaussians."""

    shrinkage: float = 0.01
    score_weight: float = 1.0
    query_variance_weight: float = 1.0
    log_variance_weight: float = 1.0


@dataclasses.dataclass(frozen=True)
class GaussianInputs:
    """Raw candidate/query profiles used by diagonal Gaussian matching."""

    candidate_game_profiles: np.ndarray
    candidate_labels: np.ndarray
    query_profiles: np.ndarray
    query_labels: np.ndarray
    candidate_players: list[int]
    query_count: int
    feature_weights: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class CandidateDistribution:
    """Per-player candidate means, variances, and game counts."""

    means: np.ndarray
    variances: np.ndarray
    counts: np.ndarray
    global_variance: np.ndarray


@dataclasses.dataclass(frozen=True)
class QueryDistribution:
    """Per-query-player means and within-query variances."""

    means: np.ndarray
    variances: np.ndarray


@dataclasses.dataclass(frozen=True)
class EvaluationInputs:
    """Feature profiles and labels needed by either retrieval scorer."""

    candidate_game_profiles: np.ndarray
    candidate_labels: np.ndarray
    candidate_profiles: np.ndarray
    query_profiles: np.ndarray
    query_labels: np.ndarray
    candidate_players: list[int]
    feature_weights: np.ndarray | None


def parse_arguments() -> argparse.Namespace:
    """Parses a feature-cache directory and metric output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--drop-features", nargs="*", default=())
    parser.add_argument(
        "--representation",
        choices=("flatten", "phase_stats", "multi_scale_stats"),
        default="flatten",
        help="Per-game representation before player-level averaging.",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=4,
        help="Number of temporal bins used by phase_stats.",
    )
    parser.add_argument(
        "--include-phase-std",
        action="store_true",
        help="Append within-bin standard deviations for phase_stats.",
    )
    parser.add_argument(
        "--phase-scales",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8),
        help="Temporal scales used by multi_scale_stats.",
    )
    parser.add_argument(
        "--include-robust-stats",
        action="store_true",
        help="Add quantiles, IQR, and skewness for fingerprint features.",
    )
    parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Add phase-level correlations and conditional decision features.",
    )
    parser.add_argument(
        "--feature-weighting",
        choices=("uniform", "fisher"),
        default="uniform",
        help="Weights dimensions uniformly or by candidate-derived reliability.",
    )
    parser.add_argument(
        "--fisher-exponent",
        type=float,
        default=0.5,
        help="Reliability exponent used when --feature-weighting=fisher.",
    )
    parser.add_argument(
        "--fisher-minimum",
        type=float,
        default=0.01,
        help="Minimum clipped Fisher reliability ratio.",
    )
    parser.add_argument(
        "--fisher-maximum",
        type=float,
        default=20.0,
        help="Maximum clipped Fisher reliability ratio.",
    )
    parser.add_argument(
        "--matching-method",
        choices=("cosine", "shrinkage_gaussian", "cosine_gaussian"),
        default="cosine",
        help="Scores player centroids or their game-level distributions.",
    )
    parser.add_argument(
        "--gaussian-shrinkage",
        type=float,
        default=0.01,
        help="Pooled within-player variance multiplier for Gaussian matching.",
    )
    parser.add_argument(
        "--gaussian-score-weight",
        type=float,
        default=1.0,
        help="Gaussian score share after row-wise cosine/Gaussian standardization.",
    )
    parser.add_argument(
        "--gaussian-query-variance-weight",
        type=float,
        default=1.0,
        help="Multiplier for empirical query-mean variance in Gaussian matching.",
    )
    parser.add_argument(
        "--gaussian-log-variance-weight",
        type=float,
        default=1.0,
        help="Multiplier for the Gaussian log-variance penalty.",
    )
    return parser.parse_args()


def load_feature_file(
    path: pathlib.Path,
    dropped_features: tuple[str, ...],
    profile_config: ProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads per-game decision profiles and their player labels.

    Args:
        path: Compressed feature cache to load.
        dropped_features: Names of cache dimensions to exclude.
        profile_config: Aggregation method and included feature groups.

    Returns:
        Per-game profiles and their player labels.

    Raises:
        ValueError: The representation name or phase-bin count is invalid.
    """
    if profile_config is None:
        profile_config = ProfileConfig()
    profile_config = resolve_profile_config(profile_config)
    tokens, masks, player_indices, feature_names = read_feature_cache(path)
    selected_features = select_feature_indices(feature_names, dropped_features)
    tokens = tokens[:, :, selected_features]
    selected_names = tuple(feature_names[index] for index in selected_features)
    if profile_config.representation == "flatten":
        profiles = tokens * masks[:, :, None]
        return profiles.reshape(len(profiles), -1), player_indices
    if profile_config.representation == "phase_stats":
        profiles = make_feature_stat_profiles(
            tokens,
            masks,
            selected_names,
            profile_config,
        )
        return profiles, player_indices
    if profile_config.representation == "multi_scale_stats":
        profiles = make_feature_stat_profiles(
            tokens,
            masks,
            selected_names,
            profile_config,
        )
        return profiles, player_indices
    raise ValueError(f"Unknown representation: {profile_config.representation!r}")


def resolve_profile_config(profile_config: ProfileConfig) -> ProfileConfig:
    """Makes the effective temporal-scale configuration explicit in outputs."""
    if profile_config.representation == "phase_stats":
        return dataclasses.replace(
            profile_config,
            phase_scales=(profile_config.phase_bins,),
        )
    return profile_config


def read_feature_cache(
    path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads one compressed decision-token cache without feature selection."""
    with np.load(path) as payload:
        tokens = np.asarray(payload["tokens"])
        masks = np.asarray(payload["masks"])
        player_indices = np.asarray(payload["player_indices"])
        feature_names = [
            str(feature_name)
            for feature_name in np.asarray(payload["feature_names"]).reshape(-1)
        ]
    return tokens, masks, player_indices, feature_names


def make_phase_stat_profiles(
    tokens: np.ndarray,
    masks: np.ndarray,
    phase_bins: int,
    include_phase_std: bool,
) -> np.ndarray:
    """Summarizes uniformly aligned decision phases by mean and deviation.

    The compact representation preserves opening-to-endgame variation while
    reducing query-side noise from sparse 5--20-game query sets.

    Args:
        tokens: Decision token tensor shaped games by turns by features.
        masks: Valid-turn mask shaped games by turns.
        phase_bins: Number of temporal groups to summarize.
        include_phase_std: Whether to append standard deviations for each bin.

    Returns:
        One phase-statistics profile per game.

    Raises:
        ValueError: ``phase_bins`` is outside the available turn range.
    """
    turn_count = tokens.shape[1]
    if phase_bins <= 0 or phase_bins > turn_count:
        raise ValueError(
            f"phase_bins must be in [1, {turn_count}], got {phase_bins}",
        )
    profile_parts: list[np.ndarray] = []
    for turn_indices in np.array_split(np.arange(turn_count), phase_bins):
        phase_tokens = tokens[:, turn_indices, :]
        phase_masks = masks[:, turn_indices, None].astype(np.float32)
        phase_counts = phase_masks.sum(axis=1).clip(min=1.0)
        phase_means = (phase_tokens * phase_masks).sum(axis=1) / phase_counts
        profile_parts.append(phase_means)
        if include_phase_std:
            deviations = phase_tokens - phase_means[:, None, :]
            variances = (deviations * deviations * phase_masks).sum(axis=1)
            variances /= phase_counts
            profile_parts.append(np.sqrt(variances))
    return np.concatenate(profile_parts, axis=1)


def make_feature_stat_profiles(
    tokens: np.ndarray,
    masks: np.ndarray,
    feature_names: tuple[str, ...],
    profile_config: ProfileConfig,
) -> np.ndarray:
    """Builds concatenated phase statistics for one or more temporal scales.

    Robust moments focus on policy surprise, policy rank, and ownership change.
    Interactions describe decision coupling and conditional behaviour within a
    phase, rather than treating every decision feature independently.

    Args:
        tokens: Decision token tensor shaped games by turns by features.
        masks: Valid-turn mask shaped games by turns.
        feature_names: Ordered names corresponding to the final token axis.
        profile_config: Temporal scales and included statistic feature groups.

    Returns:
        One multi-scale statistics profile per game.

    Raises:
        ValueError: A scale or required feature name is invalid.
    """
    validate_phase_scales(profile_config.phase_scales, tokens.shape[1])
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    validate_requested_features(
        feature_indices,
        profile_config.include_robust_stats,
        profile_config.include_interactions,
    )
    profile_parts: list[np.ndarray] = []
    for phase_bins in profile_config.phase_scales:
        for turn_indices in np.array_split(np.arange(tokens.shape[1]), phase_bins):
            phase_tokens = tokens[:, turn_indices, :]
            phase_masks = masks[:, turn_indices, None].astype(np.float32)
            phase_means, phase_stds = make_phase_moments(
                phase_tokens,
                phase_masks,
            )
            profile_parts.append(phase_means)
            if profile_config.include_phase_std:
                profile_parts.append(phase_stds)
            if profile_config.include_robust_stats:
                profile_parts.append(
                    make_robust_statistics(
                        phase_tokens,
                        phase_masks,
                        phase_means,
                        phase_stds,
                        feature_indices,
                    ),
                )
            if profile_config.include_interactions:
                profile_parts.append(
                    make_interaction_statistics(
                        phase_tokens,
                        phase_masks,
                        phase_means,
                        feature_indices,
                    ),
                )
    return np.concatenate(profile_parts, axis=1)


def validate_phase_scales(phase_scales: tuple[int, ...], turn_count: int) -> None:
    """Validates non-duplicate temporal scales against the token sequence."""
    if not phase_scales:
        raise ValueError("phase_scales must contain at least one scale")
    if len(set(phase_scales)) != len(phase_scales):
        raise ValueError(f"phase_scales must not repeat values: {phase_scales!r}")
    invalid_scales = [scale for scale in phase_scales if scale <= 0 or scale > turn_count]
    if invalid_scales:
        raise ValueError(
            f"phase_scales must be in [1, {turn_count}], got {invalid_scales!r}",
        )


def validate_requested_features(
    feature_indices: dict[str, int],
    include_robust_stats: bool,
    include_interactions: bool,
) -> None:
    """Fails early when requested feature groups were excluded from a cache."""
    required_names: set[str] = set()
    if include_robust_stats:
        required_names.update(ROBUST_FEATURE_NAMES)
    if include_interactions:
        for left_name, right_name in INTERACTION_FEATURE_PAIRS:
            required_names.update((left_name, right_name))
    if include_interactions:
        required_names.update(
            (
                "policy_entropy",
                "policy_surprisal",
                "player_score_lead_scaled",
                "ownership_change",
            ),
        )
    missing_names = sorted(required_names - set(feature_indices))
    if missing_names:
        raise ValueError(f"Profile statistics require missing features: {missing_names!r}")


def make_phase_moments(
    tokens: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes masked per-feature means and standard deviations for a phase."""
    counts = masks.sum(axis=1).clip(min=1.0)
    means = (tokens * masks).sum(axis=1) / counts
    deviations = tokens - means[:, None, :]
    variances = (deviations * deviations * masks).sum(axis=1) / counts
    return means, np.sqrt(variances)


def make_robust_statistics(
    tokens: np.ndarray,
    masks: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    feature_indices: dict[str, int],
) -> np.ndarray:
    """Computes quantile, IQR, and skewness fingerprints for key dimensions."""
    indices = [
        index
        for name, index in feature_indices.items()
        if is_robust_feature(name)
    ]
    values = tokens[:, :, indices]
    quantile_statistics = make_quantile_statistics(values, masks)
    skewness = make_skewness(
        values,
        masks,
        means[:, indices],
        standard_deviations[:, indices],
    )
    return np.concatenate((quantile_statistics, skewness), axis=1)


def is_robust_feature(name: str) -> bool:
    """Selects features whose within-game distribution identifies a player."""
    if name in ROBUST_FEATURE_NAMES:
        return True
    return name.startswith("human_") and name.endswith(
        ("_surprisal", "_log_policy_rank"),
    )


def make_quantile_statistics(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Returns median, tail quantiles, and IQR for every selected feature."""
    quantile_values = np.where(masks.astype(bool), values, np.nan)
    quantiles = np.nanquantile(
        quantile_values,
        (0.1, 0.25, 0.5, 0.75, 0.9),
        axis=1,
    )
    q10, q25, median, q75, q90 = (
        np.nan_to_num(quantile) for quantile in quantiles
    )
    return np.concatenate((median, q10, q25, q75, q90, q75 - q25), axis=1)


def make_skewness(
    values: np.ndarray,
    masks: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    """Returns standardized masked third moments with stable constant cases."""
    centered_values = values - means[:, None, :]
    third_moments = (centered_values ** 3 * masks).sum(axis=1)
    third_moments /= masks.sum(axis=1).clip(min=1.0)
    return np.divide(
        third_moments,
        standard_deviations ** 3,
        out=np.zeros_like(third_moments),
        where=standard_deviations > 1e-6,
    )


def make_interaction_statistics(
    tokens: np.ndarray,
    masks: np.ndarray,
    means: np.ndarray,
    feature_indices: dict[str, int],
) -> np.ndarray:
    """Computes phase-level correlations and conditional decision behaviour."""
    correlations = make_correlation_statistics(tokens, masks, feature_indices)
    high_entropy_delta = make_high_entropy_surprisal_delta(
        tokens,
        masks,
        means,
        feature_indices,
    )
    behind_ownership_delta = make_behind_ownership_delta(
        tokens,
        masks,
        means,
        feature_indices,
    )
    return np.stack(correlations + [high_entropy_delta, behind_ownership_delta], axis=1)


def make_correlation_statistics(
    tokens: np.ndarray,
    masks: np.ndarray,
    feature_indices: dict[str, int],
) -> list[np.ndarray]:
    """Computes the configured relationship coefficients for a phase."""
    return [
        make_correlation(
            tokens[:, :, feature_indices[left_name]],
            tokens[:, :, feature_indices[right_name]],
            masks,
        )
        for left_name, right_name in INTERACTION_FEATURE_PAIRS
    ]


def make_high_entropy_surprisal_delta(
    tokens: np.ndarray,
    masks: np.ndarray,
    means: np.ndarray,
    feature_indices: dict[str, int],
) -> np.ndarray:
    """Measures surprise elevation among the phase's highest-entropy decisions."""
    policy_entropy = tokens[:, :, feature_indices["policy_entropy"]]
    policy_surprisal = tokens[:, :, feature_indices["policy_surprisal"]]
    entropy_quantiles = np.where(masks[:, :, 0].astype(bool), policy_entropy, np.nan)
    high_entropy_threshold = np.nanquantile(entropy_quantiles, 0.75, axis=1)
    high_entropy_mask = masks[:, :, 0].astype(bool)
    high_entropy_mask &= policy_entropy >= high_entropy_threshold[:, None]
    high_entropy_surprisal = make_conditional_mean(
        policy_surprisal,
        high_entropy_mask,
    )
    return high_entropy_surprisal - means[:, feature_indices["policy_surprisal"]]


def make_behind_ownership_delta(
    tokens: np.ndarray,
    masks: np.ndarray,
    means: np.ndarray,
    feature_indices: dict[str, int],
) -> np.ndarray:
    """Measures ownership volatility elevation while the player is behind."""
    score_lead = tokens[:, :, feature_indices["player_score_lead_scaled"]]
    ownership_change = tokens[:, :, feature_indices["ownership_change"]]
    behind_mask = masks[:, :, 0].astype(bool) & (score_lead < 0.0)
    behind_ownership = make_conditional_mean(ownership_change, behind_mask)
    return behind_ownership - means[:, feature_indices["ownership_change"]]


def make_correlation(
    left_values: np.ndarray,
    right_values: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Computes a masked per-game Pearson correlation with stable zero cases."""
    weights = masks[:, :, 0]
    counts = weights.sum(axis=1).clip(min=1.0)
    left_means = (left_values * weights).sum(axis=1) / counts
    right_means = (right_values * weights).sum(axis=1) / counts
    left_centered = left_values - left_means[:, None]
    right_centered = right_values - right_means[:, None]
    covariance = (left_centered * right_centered * weights).sum(axis=1) / counts
    left_std = np.sqrt((left_centered ** 2 * weights).sum(axis=1) / counts)
    right_std = np.sqrt((right_centered ** 2 * weights).sum(axis=1) / counts)
    return np.divide(
        covariance,
        left_std * right_std,
        out=np.zeros_like(covariance),
        where=(left_std * right_std) > 1e-6,
    )


def make_conditional_mean(values: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Returns a stable zero-filled per-game mean over conditional decisions."""
    weights = condition.astype(np.float32)
    return (values * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1.0)


def select_feature_indices(
    feature_names: list[str],
    dropped_features: tuple[str, ...],
) -> list[int]:
    """Selects cache dimensions after validating requested feature exclusions."""
    unknown_features = set(dropped_features) - set(feature_names)
    if unknown_features:
        raise ValueError(f"Unknown dropped features: {sorted(unknown_features)!r}")
    indices = [
        index for index, feature_name in enumerate(feature_names)
        if feature_name not in dropped_features
    ]
    if not indices:
        raise ValueError("At least one decision feature must remain")
    return indices


def grouped_average(
    profiles: np.ndarray,
    player_indices: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Averages game profiles for each player in a stable label order."""
    players = sorted(np.unique(player_indices).tolist())
    averages = [profiles[player_indices == player].mean(axis=0) for player in players]
    return players, np.stack(averages)


def normalize_profiles(
    candidate_profiles: np.ndarray,
    query_profiles: np.ndarray,
    feature_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardizes dimensions using candidates then applies cosine weighting.

    Args:
        candidate_profiles: One centroid profile per candidate player.
        query_profiles: Query profiles aligned to candidate dimensions.
        feature_weights: Optional non-negative global dimension weights.

    Returns:
        Candidate and query vectors normalized for dot-product cosine scoring.

    Raises:
        ValueError: The feature-weight shape or values are invalid.
    """
    means = candidate_profiles.mean(axis=0)
    scales = candidate_profiles.std(axis=0).clip(min=1e-6)
    candidates = (candidate_profiles - means) / scales
    queries = (query_profiles - means) / scales
    if feature_weights is not None:
        validate_feature_weights(feature_weights, candidates.shape[1])
        candidates *= feature_weights
        queries *= feature_weights
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True).clip(min=1e-6)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-6)
    return candidates, queries


def standardize_score_rows(scores: np.ndarray) -> np.ndarray:
    """Z-scores each query's candidate scores before score fusion."""
    means = scores.mean(axis=1, keepdims=True)
    scales = scores.std(axis=1, keepdims=True).clip(min=1e-8)
    return (scores - means) / scales


def validate_feature_weights(feature_weights: np.ndarray, dimension_count: int) -> None:
    """Rejects malformed retrieval weights before broadcasting them to profiles."""
    if feature_weights.ndim != 1 or len(feature_weights) != dimension_count:
        raise ValueError(
            "feature_weights must be one-dimensional with length "
            f"{dimension_count}, got shape {feature_weights.shape!r}",
        )
    if not np.isfinite(feature_weights).all() or np.any(feature_weights < 0.0):
        raise ValueError("feature_weights must be finite and non-negative")


def make_feature_weights(
    game_profiles: np.ndarray,
    player_indices: np.ndarray,
    candidate_players: list[int],
    candidate_profiles: np.ndarray,
    weighting_config: FeatureWeightingConfig,
) -> np.ndarray | None:
    """Builds candidate-derived global dimension weights for retrieval.

    Fisher reliability is the candidate-player between variance divided by mean
    within-player game variance. It rewards dimensions that distinguish players
    while remaining consistent over a player's candidate games.

    Args:
        game_profiles: Candidate-game profiles before player averaging.
        player_indices: Player label for every candidate game profile.
        candidate_players: Stable label order of candidate centroids.
        candidate_profiles: Candidate centroids in ``candidate_players`` order.
        weighting_config: Selected weighting method and Fisher hyperparameters.

    Returns:
        Per-dimension multipliers, or ``None`` for uniform weighting.

    Raises:
        ValueError: The weighting method or Fisher configuration is invalid.
    """
    if weighting_config.method == "uniform":
        return None
    if weighting_config.method != "fisher":
        raise ValueError(f"Unknown feature weighting: {weighting_config.method!r}")
    validate_fisher_config(weighting_config)
    reliability = make_fisher_reliability(
        game_profiles,
        player_indices,
        candidate_players,
        candidate_profiles,
        weighting_config,
    )
    return reliability ** weighting_config.fisher_exponent


def validate_fisher_config(weighting_config: FeatureWeightingConfig) -> None:
    """Checks Fisher reliability clipping and exponent configuration."""
    if weighting_config.fisher_exponent < 0.0:
        raise ValueError("fisher_exponent must be non-negative")
    if weighting_config.fisher_minimum <= 0.0:
        raise ValueError("fisher_minimum must be positive")
    if weighting_config.fisher_maximum < weighting_config.fisher_minimum:
        raise ValueError("fisher_maximum must be at least fisher_minimum")


def make_fisher_reliability(
    game_profiles: np.ndarray,
    player_indices: np.ndarray,
    candidate_players: list[int],
    candidate_profiles: np.ndarray,
    weighting_config: FeatureWeightingConfig,
) -> np.ndarray:
    """Calculates clipped global between-player to within-player variance ratios."""
    within_variances = []
    for player, centroid in zip(candidate_players, candidate_profiles, strict=True):
        residuals = game_profiles[player_indices == player] - centroid
        within_variances.append(np.mean(residuals * residuals, axis=0))
    within_variance = np.mean(np.stack(within_variances), axis=0)
    between_variance = np.var(candidate_profiles, axis=0)
    reliability = between_variance / np.maximum(within_variance, 1e-8)
    return np.clip(
        reliability,
        weighting_config.fisher_minimum,
        weighting_config.fisher_maximum,
    )


def make_candidate_distribution(
    candidate_game_profiles: np.ndarray,
    candidate_labels: np.ndarray,
    candidate_players: list[int],
) -> CandidateDistribution:
    """Builds one candidate mean and variance vector per player."""
    means = []
    variances = []
    counts = []
    for player in candidate_players:
        player_rows = candidate_game_profiles[candidate_labels == player]
        if player_rows.size == 0:
            raise ValueError(f"Candidate player {player!r} has no games")
        means.append(player_rows.mean(axis=0))
        variances.append(player_rows.var(axis=0))
        counts.append(len(player_rows))
    variance_array = np.stack(variances)
    return CandidateDistribution(
        np.stack(means),
        variance_array,
        np.asarray(counts, dtype=np.float64),
        variance_array.mean(axis=0),
    )


def make_query_distribution(
    query_profiles: np.ndarray,
    query_labels: np.ndarray,
    candidate_players: list[int],
    query_count: int,
) -> QueryDistribution:
    """Builds aligned query-player mean and variance vectors."""
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    means = []
    variances = []
    for player in candidate_players:
        player_rows = query_profiles[query_labels == player][:query_count]
        if len(player_rows) != query_count:
            raise ValueError(
                f"Query player {player!r} has fewer than {query_count} games",
            )
        means.append(player_rows.mean(axis=0))
        variances.append(player_rows.var(axis=0))
    return QueryDistribution(np.stack(means), np.stack(variances))


def make_gaussian_scores(
    gaussian_inputs: GaussianInputs,
    gaussian_config: GaussianScoreConfig,
) -> np.ndarray:
    """Scores query means against candidate game distributions."""
    if gaussian_config.shrinkage < 0.0:
        raise ValueError("Gaussian shrinkage must be non-negative")
    if not 0.0 <= gaussian_config.score_weight <= 1.0:
        raise ValueError("Gaussian score weight must be between zero and one")
    if gaussian_config.query_variance_weight < 0.0:
        raise ValueError("Gaussian query variance weight must be non-negative")
    if gaussian_config.log_variance_weight < 0.0:
        raise ValueError("Gaussian log variance weight must be non-negative")
    dimension_count = gaussian_inputs.candidate_game_profiles.shape[1]
    if gaussian_inputs.query_profiles.shape[1] != dimension_count:
        raise ValueError("Candidate and query profile dimensions differ")
    if gaussian_inputs.feature_weights is None:
        weights = np.ones(dimension_count, dtype=np.float64)
    else:
        validate_feature_weights(gaussian_inputs.feature_weights, dimension_count)
        weights = np.square(
            gaussian_inputs.feature_weights.astype(np.float64, copy=False),
        )
    candidates = make_candidate_distribution(
        gaussian_inputs.candidate_game_profiles,
        gaussian_inputs.candidate_labels,
        gaussian_inputs.candidate_players,
    )
    queries = make_query_distribution(
        gaussian_inputs.query_profiles,
        gaussian_inputs.query_labels,
        gaussian_inputs.candidate_players,
        gaussian_inputs.query_count,
    )
    scores = np.empty(
        (len(gaussian_inputs.candidate_players), len(gaussian_inputs.candidate_players)),
        dtype=np.float64,
    )
    candidate_mean_variance = candidates.variances / candidates.counts[:, None]
    for query_index, query_mean in enumerate(queries.means):
        mean_delta = candidates.means - query_mean
        mean_variance = (
            candidate_mean_variance
            + gaussian_config.query_variance_weight
            * queries.variances[query_index] / gaussian_inputs.query_count
            + gaussian_config.shrinkage * candidates.global_variance
        ).clip(min=1e-8)
        terms = mean_delta * mean_delta / mean_variance
        terms += gaussian_config.log_variance_weight * np.log(mean_variance)
        scores[query_index] = -0.5 * np.sum(weights * terms, axis=1)
    return scores


def compute_metrics(
    evaluation_inputs: EvaluationInputs,
    matching_method: str = "cosine",
    gaussian_config: GaussianScoreConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Computes official-score retrieval metrics for every available query count."""
    if gaussian_config is None:
        gaussian_config = GaussianScoreConfig()
    max_queries = min(
        int(np.sum(evaluation_inputs.query_labels == player))
        for player in evaluation_inputs.candidate_players
    )
    max_queries = min(max_queries, 20)
    metrics: dict[str, dict[str, float]] = {}
    for query_count in (1, 5, 10, 15, 20):
        if query_count > max_queries:
            continue
        if matching_method in ("cosine", "cosine_gaussian"):
            query_rows = [
                evaluation_inputs.query_profiles[
                    evaluation_inputs.query_labels == player
                ][:query_count].mean(axis=0)
                for player in evaluation_inputs.candidate_players
            ]
            normalized_candidates, normalized_queries = normalize_profiles(
                evaluation_inputs.candidate_profiles,
                np.stack(query_rows),
                evaluation_inputs.feature_weights,
            )
            scores = normalized_queries @ normalized_candidates.T
        elif matching_method != "shrinkage_gaussian":
            raise ValueError(f"Unknown matching method: {matching_method!r}")
        if matching_method in ("shrinkage_gaussian", "cosine_gaussian"):
            gaussian_scores = make_gaussian_scores(
                GaussianInputs(
                    evaluation_inputs.candidate_game_profiles,
                    evaluation_inputs.candidate_labels,
                    evaluation_inputs.query_profiles,
                    evaluation_inputs.query_labels,
                    evaluation_inputs.candidate_players,
                    query_count,
                    evaluation_inputs.feature_weights,
                ),
                gaussian_config,
            )
            if matching_method == "shrinkage_gaussian":
                scores = gaussian_scores
            else:
                scores = (
                    (1.0 - gaussian_config.score_weight)
                    * standardize_score_rows(scores)
                    + gaussian_config.score_weight
                    * standardize_score_rows(gaussian_scores)
                )
        ranks = (scores > np.diag(scores)[:, None]).sum(axis=1) + 1
        official_scores = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0)
        metrics[str(query_count)] = {
            "top1": float(np.mean(ranks == 1)),
            "top5": float(np.mean(ranks <= 5)),
            "top5_exponential_score": float(np.mean(official_scores)),
            "mrr": float(np.mean(1 / ranks)),
        }
    return metrics


def main() -> None:
    """Evaluates cached decision tokens with an engine-feature-only baseline."""
    arguments = parse_arguments()
    profile_config = ProfileConfig(
        arguments.representation,
        arguments.phase_bins,
        arguments.include_phase_std,
        tuple(arguments.phase_scales),
        arguments.include_robust_stats,
        arguments.include_interactions,
    )
    profile_config = resolve_profile_config(profile_config)
    weighting_config = FeatureWeightingConfig(
        arguments.feature_weighting,
        arguments.fisher_exponent,
        arguments.fisher_minimum,
        arguments.fisher_maximum,
    )
    gaussian_config = GaussianScoreConfig(
        arguments.gaussian_shrinkage,
        arguments.gaussian_score_weight,
        arguments.gaussian_query_variance_weight,
        arguments.gaussian_log_variance_weight,
    )
    candidate_profiles, candidate_labels = load_feature_file(
        arguments.features / "candidates.npz",
        tuple(arguments.drop_features),
        profile_config,
    )
    query_profiles, query_labels = load_feature_file(
        arguments.features / "queries.npz",
        tuple(arguments.drop_features),
        profile_config,
    )
    candidate_game_profiles = candidate_profiles
    candidate_players, candidate_profiles = grouped_average(
        candidate_profiles,
        candidate_labels,
    )
    feature_weights = make_feature_weights(
        candidate_game_profiles,
        candidate_labels,
        candidate_players,
        candidate_profiles,
        weighting_config,
    )
    results = {
        "feature_source": str(arguments.features),
        "candidate_players": len(candidate_players),
        "dropped_features": arguments.drop_features,
        "representation": profile_config.representation,
        "phase_bins": profile_config.phase_bins,
        "include_phase_std": profile_config.include_phase_std,
        "phase_scales": profile_config.phase_scales,
        "include_robust_stats": profile_config.include_robust_stats,
        "include_interactions": profile_config.include_interactions,
        "feature_weighting": weighting_config.method,
        "fisher_exponent": weighting_config.fisher_exponent,
        "fisher_minimum": weighting_config.fisher_minimum,
        "fisher_maximum": weighting_config.fisher_maximum,
        "matching_method": arguments.matching_method,
        "gaussian_shrinkage": gaussian_config.shrinkage,
        "gaussian_score_weight": gaussian_config.score_weight,
        "gaussian_query_variance_weight": gaussian_config.query_variance_weight,
        "gaussian_log_variance_weight": gaussian_config.log_variance_weight,
        "metrics": compute_metrics(
            EvaluationInputs(
                candidate_game_profiles,
                candidate_labels,
                candidate_profiles,
                query_profiles,
                query_labels,
                candidate_players,
                feature_weights,
            ),
            arguments.matching_method,
            gaussian_config,
        ),
    }
    result_io.write_json(arguments.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
