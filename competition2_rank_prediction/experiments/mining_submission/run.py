# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Validates a mining-augmented candidate and writes an upload-ready CSV.

Run prepare first; the stable submission is read-only throughout.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.mining_submission import prepare

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


def official_questions(
    tests: pathlib.Path,
    game_questions: np.ndarray,
) -> tuple[list[str], list[np.ndarray]]:
    """Requires all official IDs and exactly the expected games for each."""
    identifiers = []
    questions = []
    for split in ("public", "private"):
        with (tests / f"rank_prediction_test_{split}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                identifier = row["question_id"]
                positions = np.flatnonzero(game_questions == identifier)
                if len(positions) != int(row["num_games"]):
                    raise ValueError(
                        f"Test game count mismatch for {identifier}"
                    )
                identifiers.append(identifier)
                questions.append(positions)
    if len(identifiers) != 800 or len(set(identifiers)) != 800:
        raise ValueError(
            "Official questions must contain exactly 800 unique IDs"
        )
    if set(identifiers) != set(game_questions):
        raise ValueError("Unknown or missing question IDs in game cache")
    return identifiers, questions


def write_submission(
    path: pathlib.Path, identifiers: list[str], probabilities: np.ndarray
) -> None:
    """Writes official rank labels in original Public-then-Private order."""
    if probabilities.shape != (len(identifiers), 10):
        raise ValueError(
            "Submission needs one ten-rank probability row per question"
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1
    ):
        raise ValueError("Submission probabilities are invalid")
    predictions = rank_lib.decode(probabilities)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("question_id", "pred_rank"))
        for identifier, prediction in zip(identifiers, predictions):
            writer.writerow((identifier, rank_lib.LEVELS[int(prediction)]))


def _baseline_parity(
    output: pathlib.Path,
    baseline: pathlib.Path,
) -> None:
    if output.read_bytes() != baseline.read_bytes():
        raise ValueError(
            "Baseline replay does not reproduce the stable CSV exactly"
        )


def _cohorts(
    output: pathlib.Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(output / "prepared.npz") as saved:
        validation = {
            name[4:]: np.asarray(saved[name])
            for name in saved.files
            if name.startswith("val_")
        }
        test = {
            name[5:]: np.asarray(saved[name])
            for name in saved.files
            if name.startswith("test_")
        }
    return validation, test


def _replay(
    args: argparse.Namespace,
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    tests = (
        args.research / "data/aicup2026_rank_prediction_tests_20261030/tests"
    )
    identifiers, questions = official_questions(tests, test["player_id"])
    original = calibration.fit(validation, prepare.MEMBERS[:3])
    probabilities = calibration.predict(original, test, questions)
    replay = args.output / "baseline_reproduced.csv"
    write_submission(replay, identifiers, probabilities)
    _baseline_parity(replay, args.baseline / "submission_rank.csv")
    return identifiers, questions, probabilities


def _method_report(
    path: pathlib.Path,
    identifiers: list[str],
    questions: list[np.ndarray],
    changed: np.ndarray,
    member: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("question_id", "num_games", "method", "changed"))
        for identifier, rows, difference in zip(
            identifiers, questions, changed
        ):
            writer.writerow(
                (identifier, len(rows), f"C1+C2+{member}", int(difference))
            )


def main() -> None:
    """Completes player-safe validation, parity and alternate test inference."""
    args = prepare.arguments()
    if (args.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite submission report")
    torch.set_num_threads(4)
    validation, test = _cohorts(args.output)
    members = tuple(prepare.MEMBERS[:3]) + (args.member,)
    cv, oof = calibration.cross_validate(validation, members)
    np.savez_compressed(
        args.output / "validation_oof.npz", allow_pickle=False, **oof
    )
    (args.output / "validation_cv.json").write_text(
        json.dumps(cv, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(cv, indent=2), flush=True)
    if cv["utility"]["candidate"] < cv["utility"]["baseline"] - 0.005:
        raise RuntimeError("Candidate failed the validation delivery screen")
    identifiers, questions, old_probabilities = _replay(args, validation, test)
    model = calibration.fit(validation, members)
    if model.weights[-1] <= 1e-6:
        raise RuntimeError(
            "Candidate model has no contribution to validation-fitted fusion"
        )
    probabilities = calibration.predict(model, test, questions)
    changed = rank_lib.decode(probabilities) != rank_lib.decode(
        old_probabilities
    )
    if not changed.any():
        raise RuntimeError(
            "Alternate submission would be identical to the stable version"
        )
    submission = args.output / f"submission_rank_{args.member}_20261108.csv"
    write_submission(submission, identifiers, probabilities)
    np.savez_compressed(
        args.output / "question_predictions.npz",
        question_id=np.asarray(identifiers),
        baseline=old_probabilities,
        candidate=probabilities,
    )
    _method_report(
        args.output / "prediction_report.csv",
        identifiers,
        questions,
        changed,
        args.member,
    )
    report = {
        "method": f"C1+C2 + fourth {args.member} member; validation refit",
        "validation_cv": cv,
        "fusion_weights": dict(zip(members, model.weights.tolist())),
        "baseline_replay_exact": True,
        "questions": len(identifiers),
        "changed_questions": int(changed.sum()),
        "fallbacks": 0,
        "submission": str(submission),
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "scope": "experimental alternate; official Public/Private performance unknown",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
