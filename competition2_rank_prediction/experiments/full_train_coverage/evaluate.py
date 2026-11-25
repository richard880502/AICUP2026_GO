# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Evaluates the expanded encoder inside the locked C1+C2 cross-fit.

The existing validation cohort fits fusion weights and the C1 residual. The
2,000-question cross-fit is then scored once, with its held-out players
excluded from each fold's expanded training data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import typing

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch


def _load_crossfit() -> typing.Any:
    """The locked C1+C2 cross-fit module, loaded by path (it puts the project root on sys.path)."""
    path = pathlib.Path(__file__).resolve().parents[1] / "crossfit_cohort" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("crossfit_cohort_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crossfit = _load_crossfit()

# pylint: disable=wrong-import-position
import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
# pylint: enable=wrong-import-position

VARIANTS = {
    "C1+C2": ("d96", "res", "oppC2"),
    "C1+C2+fullcov": ("d96", "res", "oppC2", "fullcov"),
}


def project_root() -> pathlib.Path:
    """Returns the repository root for this experiment module."""
    return pathlib.Path(__file__).resolve().parents[3]


def aligned_probabilities(
    prediction_path: pathlib.Path,
    game_keys: np.ndarray,
) -> np.ndarray:
    """Loads probabilities in the requested game-key order."""
    with np.load(prediction_path, allow_pickle=True) as data:
        keys = data.get("game_key")
        probabilities = data.get("probabilities")
        row_by_key = {
            str(game_key): row
            for row, game_key in enumerate(keys)
        }
        missing = [key for key in game_keys if str(key) not in row_by_key]
        if missing:
            raise ValueError(
                f"{len(missing)} requested game keys are missing from "
                f"{prediction_path!r}",
            )
        rows = [row_by_key[str(key)] for key in game_keys]
        return np.clip(probabilities[rows], 1e-6, 1.0)


def attach_fold_predictions(
    half: typing.Any,
    fold: int,
    probability_dir: pathlib.Path,
    candidate_name: str,
) -> None:
    """Adds the fold-specific expanded model to existing validation/test data."""
    validation_file = crossfit.RES / f"val_probs_d96_c{fold}.npz"
    with np.load(validation_file, allow_pickle=True) as validation_data:
        validation_keys = validation_data["game_key"]
    test_file = crossfit.member_dump(half.sfx)
    with np.load(test_file, allow_pickle=True) as test_data:
        test_keys = test_data["game_key"]
    half.val[candidate_name] = aligned_probabilities(
        probability_dir / f"c{fold}_validation.npz",
        validation_keys,
    )
    half.test[candidate_name] = aligned_probabilities(
        probability_dir / f"c{fold}_holdout3.npz",
        test_keys,
    )


def score_variant(
    half: typing.Any,
    members: tuple[str, ...],
) -> tuple[np.ndarray, list[float]]:
    """Fits validation fusion and C1, then returns per-question scores."""
    validation_logs = np.log(
        np.stack([half.val[member] for member in members]),
    )
    test_logs = np.log(
        np.stack([half.test[member] for member in members]),
    )
    weights = blend_eval.fit_weights(validation_logs, half.vy)
    probabilities = crossfit.residual_probs(
        half,
        "oppC2",
        (validation_logs, test_logs),
        weights,
    )
    return crossfit.score(probabilities, half.ty), weights.tolist()


def paired_report(
    baseline: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    ranks: np.ndarray,
) -> dict[str, object]:
    """Summarizes paired player-level score changes and the fixed advancement gate."""
    difference = candidate - baseline
    random_generator = np.random.default_rng(20261109)
    bootstrap_rows = random_generator.integers(
        0,
        len(difference),
        size=(10000, len(difference)),
    )
    bootstrap_means = difference[bootstrap_rows].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
    fold_differences = {
        str(fold): float(difference[folds == fold].mean())
        for fold in (0, 1)
    }
    rank_differences = {
        str(rank): float(difference[ranks == rank].mean())
        for rank in rank_lib.LEVELS
    }
    delta = float(difference.mean())
    passed = (
        delta >= 0.003
        and lower > 0.0
        and all(value >= 0.0 for value in fold_differences.values())
    )
    return {
        "baseline_score": float(baseline.mean()),
        "candidate_score": float(candidate.mean()),
        "delta": delta,
        "paired_player_bootstrap_95_ci": [float(lower), float(upper)],
        "by_crossfit_fold_delta": fold_differences,
        "by_rank_delta": rank_differences,
        "advancement_gate_passed": passed,
    }


def verify_baseline(
    report: dict[str, object],
    reference_path: pathlib.Path,
) -> None:
    """Checks that the rerun C1+C2 result matches the recorded cross-fit."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = float(reference["scores"]["C1+C2"])
    observed = float(typing.cast(float, report["baseline_score"]))
    if abs(expected - observed) > 1e-6:
        raise ValueError(
            f"Baseline replay mismatch: expected {expected:.9f}, "
            f"observed {observed:.9f}",
        )


def evaluate_crossfit(
    research: pathlib.Path,
    probability_dir: pathlib.Path,
    candidate_name: str,
) -> dict[str, object]:
    """Scores both cross-fit folds using validation-fitted calibration."""
    # The legacy evaluator defaults to GPU 0, which hosts another service.
    crossfit.DEV = torch.device("cpu")
    labels = pd.read_csv(research / "features/holdout3/question_folds.csv")
    opponent_features = rank_lib.load_features(
        research / "features/feat96opp/holdout3",
    )
    final_results = (
        residual_shift.load_final(
            research / "features/final_result/val",
        ),
        residual_shift.load_final(
            research / "features/final_result/holdout3",
        ),
    )
    baseline_scores = np.full(len(labels), np.nan)
    candidate_scores = np.full(len(labels), np.nan)
    fold_weights: dict[str, list[float]] = {}
    for fold in (0, 1):
        half, rows = crossfit.load_half(
            fold,
            labels,
            opponent_features,
            final_results,
        )
        attach_fold_predictions(
            half,
            fold,
            probability_dir,
            candidate_name,
        )
        baseline_scores[rows], fold_weights[f"fold{fold}_C1+C2"] = score_variant(
            half,
            VARIANTS["C1+C2"],
        )
        candidate_scores[rows], fold_weights[f"fold{fold}_candidate"] = score_variant(
            half,
            ("d96", "res", "oppC2", candidate_name),
        )
        print(
            f"fold {fold}: baseline={baseline_scores[rows].mean():.6f}, "
            f"{candidate_name}={candidate_scores[rows].mean():.6f}",
            flush=True,
        )
    if not np.isfinite(baseline_scores).all() or not np.isfinite(
        candidate_scores,
    ).all():
        raise ValueError("Cross-fit evaluation left unscored questions")
    report = paired_report(
        baseline_scores,
        candidate_scores,
        labels["fold"].to_numpy(),
        labels["rank"].to_numpy(),
    )
    report["questions"] = int(len(labels))
    report["fusion_weights"] = fold_weights
    report["candidate"] = candidate_name
    report["cohort_note"] = (
        "Historical 2,000-player cross-fit; labels were used only for final "
        "screening. This is not a new Public result."
    )
    verify_baseline(
        report,
        research / "experiments/crossfit_cohort/results/report.json",
    )
    return report


def main() -> None:
    """Scores the expanded model against the original C1+C2 pipeline."""
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", default="fullcov")
    parser.add_argument("--probability-dir", type=pathlib.Path, default=None)
    parser.add_argument("--output-report", type=pathlib.Path, default=None)
    args = parser.parse_args()
    output = root / "production/full_train_coverage_20261109"
    probability_dir = args.probability_dir
    if probability_dir is None:
        probability_dir = output / "probabilities"
    torch.set_num_threads(8)
    report = evaluate_crossfit(
        root / "runtime/competition2_research",
        probability_dir,
        args.candidate_name,
    )
    report_path = args.output_report
    if report_path is None:
        report_path = output / "crossfit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
