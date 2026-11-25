"""Tests unlabeled uniform-prior adaptation on fixed cross-fitted outputs."""

import json
import pathlib
import sys

import numpy as np

from competition2_rank_prediction.experiments.semi_hard_mining import statistics
from competition2_rank_prediction.experiments.reward_calibration import run as reward_run

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


def adapt(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fits uniform class offsets from probabilities only and applies them."""
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    offsets = np.zeros(probabilities.shape[1])
    for _ in range(200):
        logits = log_probabilities + offsets
        logits -= logits.max(1, keepdims=True)
        adjusted = np.exp(logits)
        adjusted /= adjusted.sum(1, keepdims=True)
        marginal = np.clip(adjusted.mean(0), 1e-12, 1.0)
        offsets -= 0.5 * np.log(marginal * probabilities.shape[1])
        offsets -= offsets.mean()
    logits = log_probabilities + offsets
    logits -= logits.max(1, keepdims=True)
    adjusted = np.exp(logits)
    adjusted /= adjusted.sum(1, keepdims=True)
    return adjusted, offsets


def summarize(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Reports utility after no-label adaptation, with scores kept separate."""
    adapted = np.empty_like(arrays["nll"])
    offsets = np.zeros((2, arrays["nll"].shape[1]))
    marginals_before = np.zeros_like(offsets)
    marginals_after = np.zeros_like(offsets)
    for fold in (0, 1):
        mask = arrays["fold"] == fold
        adapted[mask], offsets[fold] = adapt(arrays["nll"][mask])
        marginals_before[fold] = arrays["nll"][mask].mean(0)
        marginals_after[fold] = adapted[mask].mean(0)
    labels = arrays["y"]
    baseline = reward_run.official_scores(arrays["nll"], labels)
    candidate = reward_run.official_scores(adapted, labels)
    difference = candidate - baseline
    per_player = np.asarray(
        [
            difference[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    interval = statistics.bootstrap_players(per_player, confidence=0.95)
    half_deltas = {
        str(fold): float(difference[arrays["fold"] == fold].mean())
        for fold in (0, 1)
    }
    report = {
        "scope": "historically reused 2000-player cross-fit; no-label prior adaptation, not Public",
        "utility": {
            "nll_c1": float(baseline.mean()),
            "uniform_prior_adaptation": float(candidate.mean()),
        },
        "utility_delta": float(difference.mean()),
        "paired_player_95ci": interval,
        "half_deltas": half_deltas,
        "mean_probability_before": marginals_before.tolist(),
        "mean_probability_after": marginals_after.tolist(),
        "fitted_offsets": offsets.tolist(),
        "label_counts_after_fit_for_audit_only": [
            np.bincount(labels[arrays["fold"] == fold], minlength=10).tolist()
            for fold in (0, 1)
        ],
        "changed_decisions": int(
            (
                rank_lib.decode(arrays["nll"])
                != rank_lib.decode(adapted)
            ).sum()
        ),
        "screen_pass": bool(
            difference.mean() >= 0.003
            and interval["low"] > 0
            and all(value > 0 for value in half_deltas.values())
        ),
    }
    return report, {"adapted": adapted, "difference": difference}


def save_result(
    output: pathlib.Path,
    arrays: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    report: dict[str, object],
) -> None:
    """Stores source predictions, adapted outputs and the evaluation report."""
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz",
        allow_pickle=False,
        **arrays,
        **prediction,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    """Adapts each unlabeled half once and stores its evaluation report."""
    root = pathlib.Path.cwd()
    output = root / "production/label_shift_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    source = root / "production/decision_alignment_holdout3_20261109"
    with np.load(source / "predictions.npz") as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    report, prediction = summarize(arrays)
    save_result(output, arrays, prediction, report)


if __name__ == "__main__":
    main()
