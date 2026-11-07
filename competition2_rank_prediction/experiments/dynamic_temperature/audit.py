"""Builds fixed uncertainty diagnostics from cached player-held-out games.

Used by the dynamic-temperature pilot. Subset and leave-one-out predictions
are unweighted geometric-pooling proxies, not recomputed full C1+C2 outputs.
"""

import dataclasses
import pathlib

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import diagnose
from competition2_rank_prediction.experiments.rank_experts import run


SIGNAL_NAMES = (
    "consensus_rank_sd", "consensus_rank_se", "centered_logprob_variance",
    "member_rank_disagreement", "own_opponent_rank_gap", "game_count",
    "baseline_entropy", "baseline_utility_margin", "loo_pooled_rank_variance",
)
PRIMARY_SIGNAL = SIGNAL_NAMES.index("consensus_rank_se")
SUBSET_SIZES = (5, 8, 12, 16, 20)


@dataclasses.dataclass(frozen=True)
class AuditData:
    """Stores question-aligned signals, instability and subset records."""

    signals: np.ndarray
    loo_flip_fraction: np.ndarray
    subset_records: np.ndarray
    full_proxy_predictions: np.ndarray


def softmax(logits: np.ndarray) -> np.ndarray:
    """Returns stable last-axis probabilities."""
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def decode(probabilities: np.ndarray) -> np.ndarray:
    """Returns the rank maximizing expected official utility."""
    return (probabilities @ diagnose.REWARDS).argmax(axis=-1)


def _proxy_statistics(
    game_logs: np.ndarray,
    generator: np.random.Generator,
) -> tuple[float, float, np.ndarray, int]:
    total = game_logs.sum(axis=0)
    full = int(decode(softmax(total / len(game_logs))))
    loo_probabilities = softmax((total[None, :] - game_logs) / (len(game_logs) - 1))
    loo_variance = float((loo_probabilities @ np.arange(10)).var())
    flip_fraction = float(np.mean(decode(loo_probabilities) != full))
    subsets = np.full((len(SUBSET_SIZES), 4), np.nan)
    for index, size in enumerate(SUBSET_SIZES):
        if size > len(game_logs):
            continue
        probabilities = []
        for _ in range(20):
            indices = generator.choice(len(game_logs), size, replace=False)
            probabilities.append(softmax(game_logs[indices].mean(axis=0)))
        stacked = np.stack(probabilities)
        decisions = decode(stacked)
        counts = np.bincount(decisions, minlength=10)
        subsets[index] = (
            np.mean(decisions != full), (stacked @ np.arange(10)).std(),
            np.mean(diagnose.REWARDS[full, decisions]), 1 - counts.max() / 20,
        )
    return loo_variance, flip_fraction, subsets, full


def build_audit(root: pathlib.Path, data: run.PilotData) -> AuditData:
    """Builds audit signals without fitting models or using labels for routing."""
    games, question_ids = _cached_arrays(root)
    signals = []
    flips = []
    subsets = []
    full_predictions = []
    generator = np.random.default_rng(20261107)
    for index, question in enumerate(question_ids):
        selected = np.flatnonzero(games["question_id"] == question)
        if len(selected) != data.counts[index] or len(selected) < 5:
            raise ValueError(f"Invalid question game count for {question!r}")
        members = np.stack((
            games["d96"][selected], games["resnet"][selected],
            games["opp"][selected],
        ), axis=1)
        values = _question_audit(members, data.probabilities[index], generator)
        signals.append(values[0])
        flips.append(values[1])
        subsets.append(values[2])
        full_predictions.append(values[3])
    return AuditData(
        np.asarray(signals), np.asarray(flips), np.stack(subsets),
        np.asarray(full_predictions),
    )


def _cached_arrays(root: pathlib.Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path = root / "experiments/sgf_structure/results/game_members_holdout3.npz"
    with np.load(path) as payload:
        games = {name: np.asarray(payload[name]).copy() for name in payload.files}
    baseline_path = root / "experiments/sgf_structure/results/base_holdout3.npz"
    with np.load(baseline_path) as payload:
        question_ids = np.asarray(payload["question_id"]).copy()
    return games, question_ids


def _question_audit(
    members: np.ndarray,
    baseline: np.ndarray,
    generator: np.random.Generator,
) -> tuple[list[float], float, np.ndarray, int]:
    consensus = members.mean(axis=1) @ np.arange(10)
    member_ranks = members @ np.arange(10)
    game_logs = np.log(members.clip(1e-6)).mean(axis=1)
    centered_logs = game_logs - game_logs.mean(axis=1, keepdims=True)
    variance, flip, subsets, full = _proxy_statistics(game_logs, generator)
    ordered = np.sort(baseline @ diagnose.REWARDS)
    signals = [
        consensus.std(), consensus.std() / np.sqrt(len(members)),
        centered_logs.var(axis=0).mean(), member_ranks.std(axis=1).mean(),
        np.abs(member_ranks[:, 0] - member_ranks[:, 2]).mean(), len(members),
        -np.sum(baseline * np.log(baseline.clip(1e-9))),
        ordered[-1] - ordered[-2], variance,
    ]
    return signals, flip, subsets, full


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() < 1e-10 or second.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def association_report(audit: AuditData, data: run.PilotData) -> dict[str, object]:
    """Reports descriptive feature-error and feature-instability associations."""
    predictions = decode(data.probabilities)
    realized = diagnose.REWARDS[data.labels, predictions]
    groups = {
        "all": np.ones(len(data.labels), dtype=bool),
        "half_0": data.halves == 0, "half_1": data.halves == 1,
    }
    results: dict[str, object] = {}
    for name, selected in groups.items():
        correlations = {}
        for index, feature in enumerate(SIGNAL_NAMES):
            values = audit.signals[selected, index]
            correlations[feature] = {
                "r_with_realized_utility": _correlation(values, realized[selected]),
                "r_with_loo_proxy_instability": _correlation(
                    values, audit.loo_flip_fraction[selected],
                ),
            }
        results[name] = {
            "players": int(selected.sum()), "associations": correlations,
        }
    primary = audit.signals[:, PRIMARY_SIGNAL]
    quartiles = np.quantile(primary, (0.25, 0.5, 0.75))
    bins = np.digitize(primary, quartiles)
    results["primary_uncertainty_quartiles"] = [
        {
            "quartile": index + 1, "players": int((bins == index).sum()),
            "baseline_utility": float(realized[bins == index].mean()),
            "loo_proxy_flip_fraction": float(audit.loo_flip_fraction[bins == index].mean()),
        }
        for index in range(4)
    ]
    return results


def subset_report(audit: AuditData, data: run.PilotData) -> dict[str, object]:
    """Aggregates resamples within player and distinguishes matched cohorts."""
    results: dict[str, object] = {}
    for scope in ("all_eligible", "same_twenty_game_players"):
        rows = []
        for index, size in enumerate(SUBSET_SIZES):
            selected = data.counts >= size
            if scope == "same_twenty_game_players":
                selected &= data.counts >= 20
            values = audit.subset_records[selected, index]
            rows.append({
                "games": size, "players": int(selected.sum()),
                "flip_vs_full_proxy": float(values[:, 0].mean()),
                "expected_rank_std_across_resamples": float(values[:, 1].mean()),
                "utility_agreement_with_full_proxy": float(values[:, 2].mean()),
                "one_minus_modal_decision_share": float(values[:, 3].mean()),
            })
        results[scope] = rows
    results["interpretation"] = (
        "Unweighted geometric member ensemble proxy, not full C1+C2; "
        "each player contributes one mean over 20 resamples. Full-set "
        "sampling is necessarily stable when subset size equals game count."
    )
    return results


def temperature_grid_audit(data: run.PilotData) -> dict[str, object]:
    """Reports an unrealizable true-label oracle over a fixed diagnostic grid."""
    grid = np.geomspace(0.5, 2.0, 9)
    decisions = np.stack([
        decode(softmax(np.log(data.probabilities.clip(1e-9)) / temperature))
        for temperature in grid
    ])
    utilities = diagnose.REWARDS[data.labels[None, :], decisions]
    return {
        "temperature_grid": grid.tolist(),
        "all_grid_decisions_identical_fraction": float(np.mean(
            np.all(decisions == decisions[0], axis=0),
        )),
        "true_label_grid_oracle_utility_not_a_model": float(utilities.max(axis=0).mean()),
        "warning": "Oracle uses labels and cannot be fitted, selected or submitted",
    }
