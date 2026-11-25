"""Evaluates three fixed leave-one-seed-out opponent ensembles."""

import json
import pathlib

import numpy as np
import torch
from competition2_rank_prediction.experiments.decision_alignment import (
    holdout3,
)
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")
SEED_SETS = {
    "drop0": (1, 2),
    "drop1": (0, 2),
    "drop2": (0, 1),
}


def aligned_seed_probabilities(
    path: pathlib.Path, keys: np.ndarray
) -> np.ndarray:
    """Loads seed probabilities in the requested immutable game order."""
    with np.load(path) as cache:
        lookup = {
            str(key): row for row, key in enumerate(cache["game_key"])
        }
        if set(keys) != set(lookup):
            raise ValueError(f"Seed cache keys differ from requested keys: {path}")
        rows = np.asarray([lookup[str(key)] for key in keys])
        result = np.asarray(cache["probabilities"])
    if result.shape[0] != 3:
        raise ValueError("Expected exactly three trained seeds")
    return result[:, rows]


def seed_mean(
    probabilities: np.ndarray, selected: tuple[int, ...]
) -> np.ndarray:
    """Returns the uniform mean over a fixed subset of seed predictions."""
    return probabilities[np.asarray(selected)].mean(0)


def audit_seed_mean(
    root: pathlib.Path,
    fold: int,
    split: str,
    data: dict[str, np.ndarray],
    seed_probabilities: np.ndarray,
) -> None:
    """Requires three-seed means to replay established cached predictions."""
    if split == "validation":
        path = (
            root
            / "runtime/competition2_research/results"
            / f"val_probs_opp_scratch_c{fold}.npz"
        )
        field = "transformer"
    elif split == "holdout3":
        path = (
            root
            / "production/decision_alignment_holdout3_20261109"
            / f"opp_scratch_c{fold}.npz"
        )
        field = "opp_scratch"
    else:
        raise ValueError(f"Unknown seed-audit split: {split}")
    with np.load(path, allow_pickle=True) as cache:
        cached_keys = np.asarray(cache["game_key"]).astype(str)
        lookup = {
            str(key): row for row, key in enumerate(cached_keys)
        }
        rows = np.asarray(
            [lookup[str(key)] for key in data["game_key"]]
        )
        cached = np.asarray(cache[field])[rows]
    np.testing.assert_allclose(
        seed_mean(seed_probabilities, (0, 1, 2)),
        cached,
        atol=1e-6,
        rtol=1e-6,
    )


def fold_predictions(
    root: pathlib.Path, fold: int, question_rows: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    """Fits validation-only calibration for the full and three reduced means."""
    output = root / "production/seed_ensemble_20261109"
    training = holdout3.validation_data(root, fold)
    testing, questions, labels = holdout3.test_data(root, fold, question_rows)
    validation_seeds = aligned_seed_probabilities(
        output / f"val_c{fold}.npz", training["game_key"]
    )
    test_seeds = aligned_seed_probabilities(
        output / f"holdout3_c{fold}.npz", testing["game_key"]
    )
    audit_seed_mean(root, fold, "validation", training, validation_seeds)
    audit_seed_mean(root, fold, "holdout3", testing, test_seeds)
    results: dict[str, np.ndarray] = {}
    for name, selected in (("all3", (0, 1, 2)), *SEED_SETS.items()):
        training["opp_scratch"] = seed_mean(validation_seeds, selected)
        testing["opp_scratch"] = seed_mean(test_seeds, selected)
        model = calibration.fit(training, MEMBERS)
        results[name] = calibration.predict(model, testing, questions)
        print(f"seed ensemble fold={fold} condition={name}", flush=True)
    results["y"] = labels
    results["player"] = np.asarray(
        [row["player"] for row in question_rows if int(row["fold"]) == fold]
    )
    results["fold"] = np.full(len(labels), fold, dtype=np.int64)
    return results


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Reports adjusted paired-player intervals for all leave-one-out means."""
    baseline = statistics.player_scores(arrays["all3"], arrays["y"])
    conditions: dict[str, object] = {}
    report: dict[str, object] = {
        "scope": "historical 2000-player cross-fit; seed robustness diagnostic",
        "baseline_all_three": float(baseline.mean()),
        "n_questions": int(len(arrays["y"])),
        "players": int(len(np.unique(arrays["player"]))),
        "conditions": conditions,
    }
    for name, selected in SEED_SETS.items():
        candidate = statistics.player_scores(arrays[name], arrays["y"])
        difference = candidate - baseline
        per_player = np.asarray(
            [
                difference[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        )
        interval = statistics.bootstrap_players(
            per_player, confidence=1 - 0.05 / len(SEED_SETS)
        )
        half_deltas = {
            str(fold): float(difference[arrays["fold"] == fold].mean())
            for fold in (0, 1)
        }
        conditions[name] = {
            "seeds_kept": list(selected),
            "utility": float(candidate.mean()),
            "delta": float(difference.mean()),
            "bonferroni_98_333ci": interval,
            "half_deltas": half_deltas,
            "screen_pass": bool(
                difference.mean() >= 0.003
                and interval["low"] > 0
                and all(value > 0 for value in half_deltas.values())
            ),
        }
    return report


def main() -> None:
    """Evaluates all fixed seed subsets and saves the results."""
    root = pathlib.Path.cwd()
    output = root / "production/seed_ensemble_20261109"
    report_path = output / "report.json"
    prediction_path = output / "evaluation_predictions.npz"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("Refusing existing seed ensemble evaluation")
    question_rows = holdout3.read_labels(
        root
        / "runtime/competition2_research/features/holdout3/question_folds.csv"
    )
    holdout3.validate_exclusion(root, question_rows)
    torch.set_num_threads(4)
    folds = [fold_predictions(root, fold, question_rows) for fold in (0, 1)]
    arrays = {
        key: np.concatenate([fold[key] for fold in folds])
        for key in folds[0]
    }
    report = summarize(arrays)
    np.savez_compressed(
        prediction_path, allow_pickle=False, **arrays
    )
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
