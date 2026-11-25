"""Confirms the fixed C1 decision objective on historical cross-fitted players."""

import csv
import pathlib

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.decision_alignment import artifacts
from competition2_rank_prediction.experiments.decision_alignment import run
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")


def read_labels(path: pathlib.Path) -> list[dict[str, str]]:
    """Loads the fixed question, player, rank and fold metadata."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aligned_values(path: pathlib.Path, field: str, keys: np.ndarray) -> np.ndarray:
    """Aligns a saved probability field to immutable game keys."""
    with np.load(path, allow_pickle=True) as cache:
        lookup = {str(key): row for row, key in enumerate(cache["game_key"])}
        if set(keys) - set(lookup):
            raise ValueError(f"Missing game keys in {path}")
        return np.asarray(cache[field])[
            np.asarray([lookup[str(key)] for key in keys])
        ]


def validation_data(root: pathlib.Path, fold: int) -> dict[str, np.ndarray]:
    """Loads validation members for the selected cross-fitted model half."""
    results = root / "runtime/competition2_research/results"
    suffix = f"_c{fold}"
    path = results / f"val_probs_d96{suffix}.npz"
    with np.load(path, allow_pickle=True) as cache:
        keys = np.asarray(cache["game_key"]).astype(str)
        data = {
            "game_key": keys,
            "y": np.asarray(cache["y"]),
            "player": np.asarray(cache["player"]).astype(str),
            "d96": np.asarray(cache["transformer"]),
        }
    data["resnet8c"] = aligned_values(
        results / f"val_probs_resnet8c{suffix}.npz",
        f"resnet8c{suffix}",
        keys,
    )
    data["opp_scratch"] = aligned_values(
        results / f"val_probs_opp_scratch{suffix}.npz", "transformer", keys
    )
    final = residual_shift.load_final(
        root / "runtime/competition2_research/features/final_result/val"
    )
    data["final"] = np.stack([final[str(key)] for key in keys])
    return data


def test_data(
    root: pathlib.Path, fold: int, question_rows: list[dict[str, str]]
) -> tuple[dict[str, np.ndarray], list[np.ndarray], np.ndarray]:
    """Loads per-game cross-fit probabilities and groups each query's games."""
    path = (
        root
        / "runtime/competition2_research/experiments/crossfit_cohort/results"
        / f"probs_c{fold}.npz"
    )
    with np.load(path, allow_pickle=True) as cache:
        game_keys = np.asarray(cache["game_key"]).astype(str)
        query_ids = np.asarray(cache["player_id"]).astype(str)
        data = {
            "game_key": game_keys,
            "player": query_ids,
            "d96": np.asarray(cache[f"d96_c{fold}"]),
            "resnet8c": np.asarray(cache[f"resnet8c_c{fold}"]),
        }
    opponent_path = (
        root
        / "production/decision_alignment_holdout3_20261109"
        / f"opp_scratch_c{fold}.npz"
    )
    with np.load(opponent_path) as cache:
        opponent_keys = np.asarray(cache["game_key"]).astype(str)
        lookup = {str(key): row for row, key in enumerate(opponent_keys)}
        if set(game_keys) != set(lookup):
            raise ValueError(
                "opp_scratch prediction keys differ from base members"
            )
        data["opp_scratch"] = np.asarray(cache["opp_scratch"])[
            np.asarray([lookup[str(key)] for key in game_keys])
        ]
    final = residual_shift.load_final(
        root / "runtime/competition2_research/features/final_result/holdout3"
    )
    data["final"] = np.stack([final[str(key)] for key in game_keys])
    questions, labels = grouped_questions(
        query_ids, game_keys, question_rows, fold
    )
    return data, questions, labels


def grouped_questions(
    query_ids: np.ndarray,
    game_keys: np.ndarray,
    question_rows: list[dict[str, str]],
    fold: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Groups cached rows by question while preserving the SGF order."""
    lookup: dict[str, list[int]] = {}
    for row, query_id in enumerate(query_ids):
        lookup.setdefault(str(query_id), []).append(row)
    questions = []
    labels = []
    for question in question_rows:
        if int(question["fold"]) != fold:
            continue
        indices = lookup.get(question["question_id"], [])
        indices.sort(
            key=lambda item: int(str(game_keys[item]).rsplit("#", 1)[1])
        )
        if not indices:
            raise ValueError("Holdout3 question has no cached game predictions")
        questions.append(np.asarray(indices))
        labels.append(rank_lib.LEVELS.index(question["rank"]))
    return questions, np.asarray(labels, dtype=np.int64)


def validate_exclusion(root: pathlib.Path, rows: list[dict[str, str]]) -> None:
    """Requires each fold's exclusion list to equal its scored players."""
    directory = root / "runtime/competition2_research/features/holdout3"
    for fold in (0, 1):
        expected = {
            row["player"] for row in rows if int(row["fold"]) == fold
        }
        with (directory / f"exclude_c{fold}.txt").open(
            encoding="utf-8"
        ) as handle:
            observed = {line.strip() for line in handle if line.strip()}
        if observed != expected:
            raise ValueError(f"Cross-fit exclusion mismatch for fold {fold}")


def evaluate_fold(
    root: pathlib.Path, fold: int, rows: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    """Fits on validation and evaluates both fixed residuals on one test half."""
    training = validation_data(root, fold)
    testing, questions, labels = test_data(root, fold, rows)
    model = calibration.fit(training, MEMBERS)
    nll = calibration.predict(model, testing, questions)
    decision_residual = run.fit_decision_residual(training, model.weights)
    decision_model = calibration.FittedModel(
        MEMBERS, model.weights, decision_residual
    )
    return {
        "nll": nll,
        "decision": calibration.predict(decision_model, testing, questions),
        "y": labels,
        "player": np.asarray(
            [row["player"] for row in rows if int(row["fold"]) == fold]
        ),
        "fold": np.full(len(labels), fold, dtype=np.int64),
    }


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Reports paired utility and confirms both cross-fitted halves."""
    scores = {
        name: run.base.official_scores(arrays[name], arrays["y"])
        for name in ("nll", "decision")
    }
    differences = scores["decision"] - scores["nll"]
    per_player = np.asarray(
        [
            differences[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    interval = statistics.bootstrap_players(per_player, confidence=0.95)
    halves = {
        str(fold): float(differences[arrays["fold"] == fold].mean())
        for fold in (0, 1)
    }
    delta = float(differences.mean())
    return {
        "scope": (
            "historically reused 2000-player cross-fit; "
            "method-specific confirmation, not Public"
        ),
        "utility": {key: float(value.mean()) for key, value in scores.items()},
        "utility_delta": delta,
        "paired_player_95ci": interval,
        "half_deltas": halves,
        "screen_pass": delta >= 0.003
        and interval["low"] > 0
        and all(value > 0 for value in halves.values()),
        "exact": {
            name: float((rank_lib.decode(arrays[name]) == arrays["y"]).mean())
            for name in scores
        },
        "questions": len(arrays["y"]),
        "players": len(per_player),
    }


def main() -> None:
    """Runs the one-time fixed external confirmation and saves its evidence."""
    root = pathlib.Path.cwd()
    output = root / "production/decision_alignment_best_holdout3_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    rows = read_labels(
        root / "runtime/competition2_research/features/holdout3/question_folds.csv"
    )
    validate_exclusion(root, rows)
    torch.set_num_threads(4)
    results = [evaluate_fold(root, fold, rows) for fold in (0, 1)]
    arrays = {
        key: np.concatenate([result[key] for result in results])
        for key in results[0]
    }
    report = summarize(arrays)
    artifacts.save(output, arrays, report)


if __name__ == "__main__":
    main()
