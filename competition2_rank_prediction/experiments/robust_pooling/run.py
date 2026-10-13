"""Compares robust multi-game pooling rules with player-disjoint validation.

The production system geometrically averages all game probabilities in a
question.  This experiment tests whether reducing the influence of atypical
games improves rank prediction without retraining either frozen encoder.

Example:
    c1/.venv/bin/python -m experiments.robust_pooling.run
"""

import concurrent.futures
import dataclasses
import json
import pathlib
from typing import Callable

import numpy as np

import blend_eval
import rank_lib
from experiments.bag_distribution import run as bag_run
from experiments.model_selection import evaluate as model_evaluate

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAMES = ("d96_h", "resnet8c_h")
SEEDS = (20261005, 20261006, 20261007)
FOLDS = 5
TRAIN_BAGS_PER_PLAYER = 10
TEST_BAGS_PER_PLAYER = 20


def _normalize(logits: np.ndarray) -> np.ndarray:
    """Converts one vector of unnormalized log probabilities to probabilities."""
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _logmean(probabilities: np.ndarray) -> np.ndarray:
    return _normalize(np.log(np.clip(probabilities, 1e-6, 1.0)).mean(axis=0))


def _mean(probabilities: np.ndarray) -> np.ndarray:
    result = probabilities.mean(axis=0)
    return result / result.sum()


def _median_log(probabilities: np.ndarray) -> np.ndarray:
    logs = np.log(np.clip(probabilities, 1e-6, 1.0))
    return _normalize(np.median(logs, axis=0))


def _trimmed_log(
    probabilities: np.ndarray,
    fraction: float,
) -> np.ndarray:
    logs = np.sort(
        np.log(np.clip(probabilities, 1e-6, 1.0)),
        axis=0,
    )
    trim = int(np.floor(len(logs) * fraction))
    if trim:
        logs = logs[trim:-trim]
    return _normalize(logs.mean(axis=0))


def _central_log(
    probabilities: np.ndarray,
    keep_fraction: float,
) -> np.ndarray:
    logs = np.log(np.clip(probabilities, 1e-6, 1.0))
    centered = logs - logs.mean(axis=1, keepdims=True)
    center = np.median(centered, axis=0)
    distance = np.square(centered - center).mean(axis=1)
    keep = max(3, int(np.ceil(len(logs) * keep_fraction)))
    selected = np.argpartition(distance, keep - 1)[:keep]
    return _normalize(logs[selected].mean(axis=0))


def _central_weighted(probabilities: np.ndarray) -> np.ndarray:
    logs = np.log(np.clip(probabilities, 1e-6, 1.0))
    centered = logs - logs.mean(axis=1, keepdims=True)
    center = np.median(centered, axis=0)
    distance = np.sqrt(np.square(centered - center).mean(axis=1))
    scale = max(float(np.median(distance)), 1e-4)
    weights = np.exp(-distance / scale)
    pooled = np.average(logs, axis=0, weights=weights)
    return _normalize(pooled)


def _trim10(probabilities: np.ndarray) -> np.ndarray:
    return _trimmed_log(probabilities, 0.1)


def _trim20(probabilities: np.ndarray) -> np.ndarray:
    return _trimmed_log(probabilities, 0.2)


def _central80(probabilities: np.ndarray) -> np.ndarray:
    return _central_log(probabilities, 0.8)


def _central60(probabilities: np.ndarray) -> np.ndarray:
    return _central_log(probabilities, 0.6)


POOLERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "logmean": _logmean,
    "mean": _mean,
    "median_log": _median_log,
    "trim10_log": _trim10,
    "trim20_log": _trim20,
    "central80_log": _central80,
    "central60_log": _central60,
    "central_weighted_log": _central_weighted,
}


@dataclasses.dataclass(frozen=True)
class EvaluationData:
    """Contains arrays and player mappings shared by validation folds."""

    logs: np.ndarray
    labels: np.ndarray
    rows_by_player: dict[str, np.ndarray]
    label_by_player: dict[str, int]
    fold_by_player: dict[str, int]


def _pool_bags(
    probabilities: np.ndarray,
    bags: list[tuple[str, np.ndarray]],
    pooler: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    return np.stack([pooler(probabilities[rows]) for _, rows in bags])


def _make_folds(
    players: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    rows_by_player = bag_run.rows_by_group(players)
    label_by_player = {
        player: int(labels[rows[0]])
        for player, rows in rows_by_player.items()
    }
    fold_by_player = bag_run.stratified_folds(
        np.asarray(sorted(rows_by_player)),
        label_by_player,
        FOLDS,
        seed,
    )
    return rows_by_player, label_by_player, fold_by_player


def _evaluate_pooler(
    fused: np.ndarray,
    train_bags: list[tuple[str, np.ndarray]],
    test_bags: list[tuple[str, np.ndarray]],
    pooler: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    train_probabilities = _pool_bags(fused, train_bags, pooler)
    offsets = rank_lib.fit_class_offsets(train_probabilities)
    test_probabilities = _pool_bags(fused, test_bags, pooler)
    calibrated = rank_lib.apply_class_offsets(test_probabilities, offsets)
    return rank_lib.decode(calibrated)


def _evaluate_fold(
    data: EvaluationData,
    seed: int,
    fold: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    training = {
        player: rows
        for player, rows in data.rows_by_player.items()
        if data.fold_by_player[player] != fold
    }
    testing = {
        player: rows
        for player, rows in data.rows_by_player.items()
        if data.fold_by_player[player] == fold
    }
    train_bags = bag_run.sample_bags(
        training,
        TRAIN_BAGS_PER_PLAYER,
        seed + fold,
    )
    test_bags = bag_run.sample_bags(
        testing,
        TEST_BAGS_PER_PLAYER,
        seed + 1000 + fold,
    )
    truth = bag_run.bag_labels(test_bags, data.label_by_player)
    train_rows = np.concatenate(list(training.values()))
    weights = blend_eval.fit_weights(
        data.logs[:, train_rows],
        data.labels[train_rows],
    )
    fused = blend_eval.fuse(data.logs, weights)
    predictions = {
        name: _evaluate_pooler(fused, train_bags, test_bags, pooler)
        for name, pooler in POOLERS.items()
    }
    return predictions, truth


def evaluate_seed(
    logs: np.ndarray,
    labels: np.ndarray,
    players: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Evaluates every pooling rule for one question-sampling seed."""
    mappings = _make_folds(
        players,
        labels,
        seed,
    )
    data = EvaluationData(logs, labels, *mappings)
    predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in POOLERS
    }
    truths = []
    for fold in range(FOLDS):
        fold_predictions, truth = _evaluate_fold(
            data,
            seed,
            fold,
        )
        for name, prediction in fold_predictions.items():
            predictions[name].append(prediction)
        truths.append(truth)
    combined_truth = np.concatenate(truths)
    return {
        name: rank_lib.competition_score(
            np.concatenate(parts),
            combined_truth,
        )
        for name, parts in predictions.items()
    }


def main() -> None:
    """Runs repeated player-disjoint validation and saves its report."""
    logs, labels, players = model_evaluate.validation_data(
        MODEL_NAMES,
    )
    probability_sums = np.exp(logs).sum(axis=2)
    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise ValueError("Validation inputs must be normalized log-probabilities")
    by_seed = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(SEEDS),
    ) as executor:
        futures = {
            executor.submit(evaluate_seed, logs, labels, players, seed): seed
            for seed in SEEDS
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            by_seed[str(seed)] = result
            print(f"seed {seed}: {result}", flush=True)
    summary = {
        name: {
            "mean": float(np.mean([result[name] for result in by_seed.values()])),
            "minimum": float(
                np.min([result[name] for result in by_seed.values()]),
            ),
            "maximum": float(
                np.max([result[name] for result in by_seed.values()]),
            ),
        }
        for name in POOLERS
    }
    report = {
        "protocol": {
            "models": list(MODEL_NAMES),
            "folds": FOLDS,
            "seeds": list(SEEDS),
            "train_bags_per_player": TRAIN_BAGS_PER_PLAYER,
            "test_bags_per_player": TEST_BAGS_PER_PLAYER,
            "holdout_evaluated": False,
        },
        "by_seed": by_seed,
        "summary": summary,
    }
    output = pathlib.Path(__file__).resolve().parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
