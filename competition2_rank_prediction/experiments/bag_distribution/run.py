"""Trains a distribution-aware rank predictor on multi-game questions.

The script consumes cached per-game probabilities from the four frozen models.
It selects a bag classifier and a baseline blend weight using player-disjoint
cross-validation, then evaluates the locked choice once on the holdout cohort.

Example:
    ../../c1/.venv/bin/python run.py
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import blend_eval
import rank_lib

MODEL_SPECS = (
    ("d96_h", "transformer"),
    ("move_h", "move_h"),
    ("resnet5_h", "resnet5_h"),
    ("resnet8c_h", "resnet8c_h"),
)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
BLEND_GRID = tuple(np.linspace(0.0, 1.0, 11))


def load_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads aligned validation probabilities, labels, and player groups."""
    probabilities = []
    reference: dict[str, np.ndarray] | None = None
    for tag, key in MODEL_SPECS:
        path = ROOT / "results" / f"val_probs_{tag}.npz"
        with np.load(path, allow_pickle=True) as data:
            if reference is None:
                reference = {
                    name: data[name]
                    for name in ("y", "player", "game_key")
                }
            elif not np.array_equal(data["game_key"], reference["game_key"]):
                raise ValueError(f"Game ordering differs for {tag!r}")
            probabilities.append(np.clip(data[key], 1e-6, 1.0))
    if reference is None:
        raise ValueError("No validation models were loaded")
    return (
        np.stack(probabilities),
        reference["y"].astype(np.int64),
        reference["player"].astype(str),
    )


def load_holdout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads holdout probabilities, question IDs, and question labels."""
    path = ROOT / "features" / "holdout" / "probs_final.npz"
    with np.load(path, allow_pickle=True) as data:
        probabilities = np.stack(
            [np.clip(data[tag], 1e-6, 1.0) for tag, _ in MODEL_SPECS],
        )
        questions = data["player_id"].astype(str)
    labels_path = ROOT / "features" / "holdout" / "tests" / "labels.csv"
    labels = pd.read_csv(labels_path)
    label_by_question = {
        str(row.question_id): rank_lib.LEVELS.index(str(row.rank))
        for row in labels.itertuples(index=False)
    }
    return probabilities, questions, np.asarray(
        [label_by_question[question] for question in questions],
        dtype=np.int64,
    )


def rows_by_group(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Maps each player or question ID to its game rows."""
    return {
        group: np.flatnonzero(groups == group)
        for group in np.unique(groups)
    }


def stratified_folds(
    players: np.ndarray,
    labels: dict[str, int],
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assigns whole players to folds while balancing rank labels."""
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for rank_id in range(len(rank_lib.LEVELS)):
        rank_players = players[
            np.asarray([labels[player] == rank_id for player in players])
        ]
        for index, player in enumerate(rng.permutation(rank_players)):
            assignment[player] = index % folds
    return assignment


def sample_bags(
    group_rows: dict[str, np.ndarray],
    per_group: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Samples 5-20 game bags without replacement from each player."""
    rng = np.random.default_rng(seed)
    bags = []
    for group, rows in sorted(group_rows.items()):
        maximum = min(20, len(rows))
        if maximum < 5:
            continue
        for _ in range(per_group):
            size = int(rng.integers(5, maximum + 1))
            bags.append((group, rng.choice(rows, size=size, replace=False)))
    return bags


def _distribution_features(log_values: np.ndarray) -> np.ndarray:
    """Summarizes model, game, and class distributions for one bag."""
    probabilities = np.exp(log_values)
    sections = [
        log_values.mean(axis=1).reshape(-1),
        log_values.std(axis=1).reshape(-1),
        probabilities.mean(axis=1).reshape(-1),
        probabilities.std(axis=1).reshape(-1),
    ]
    sections.extend(
        np.quantile(log_values, quantile, axis=1).reshape(-1)
        for quantile in QUANTILES
    )
    rank_axis = np.arange(len(rank_lib.LEVELS), dtype=np.float64)
    expected_rank = probabilities @ rank_axis
    entropy = -(probabilities * log_values).sum(axis=2)
    for values in (expected_rank, entropy):
        sections.extend(
            (
                values.mean(axis=1),
                values.std(axis=1),
                np.quantile(values, 0.1, axis=1),
                np.quantile(values, 0.5, axis=1),
                np.quantile(values, 0.9, axis=1),
            ),
        )
    model_disagreement = probabilities.mean(axis=1).std(axis=0)
    sections.append(model_disagreement)
    return np.concatenate(sections)


def bag_features(
    logs: np.ndarray,
    bags: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Builds fixed-length distribution features for sampled questions."""
    features = []
    for _, rows in bags:
        summary = _distribution_features(logs[:, rows])
        size_features = np.asarray(
            [len(rows) / 20.0, 1.0 / np.sqrt(len(rows)), np.log(len(rows))],
        )
        features.append(np.concatenate((summary, size_features)))
    return np.asarray(features, dtype=np.float32)


def bag_labels(
    bags: list[tuple[str, np.ndarray]],
    labels: dict[str, int],
) -> np.ndarray:
    """Returns one rank label for every sampled player bag."""
    return np.asarray([labels[group] for group, _ in bags], dtype=np.int64)


def make_model(name: str, seed: int) -> Any:
    """Constructs one candidate bag classifier."""
    if name == "logistic_003":
        return sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(
                C=0.03,
                max_iter=800,
                random_state=seed,
            ),
        )
    if name == "logistic_01":
        return sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(
                C=0.1,
                max_iter=800,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=12,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=16,
            random_state=seed,
        )
    raise ValueError(f"Unknown model name {name!r}")


def utility_scores(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Returns per-question competition utilities."""
    difference = np.abs(prediction - truth)
    return np.where(
        difference == 0,
        1.0,
        np.where(difference == 1, rank_lib.ADJACENT, 0.0),
    )


def normalize_log_blend(
    first: np.ndarray,
    second: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Geometrically blends two probability distributions."""
    logits = (
        (1.0 - weight) * np.log(np.clip(first, 1e-7, 1.0))
        + weight * np.log(np.clip(second, 1e-7, 1.0))
    )
    result = np.exp(logits - logits.max(axis=1, keepdims=True))
    return result / result.sum(axis=1, keepdims=True)


def baseline_probabilities(
    logs: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    train_bags: list[tuple[str, np.ndarray]],
    test_bags: list[tuple[str, np.ndarray]],
    test_logs: np.ndarray | None = None,
) -> np.ndarray:
    """Fits fusion and rank offsets on training players, then pools test bags."""
    weights = blend_eval.fit_weights(logs[:, train_rows], labels[train_rows])
    fused = blend_eval.fuse(logs, weights)
    train_probabilities = np.stack(
        [rank_lib.pool(fused[rows], "logmean") for _, rows in train_bags],
    )
    offsets = rank_lib.fit_class_offsets(train_probabilities)
    evaluation_logs = logs if test_logs is None else test_logs
    evaluation_fused = blend_eval.fuse(evaluation_logs, weights)
    test_probabilities = np.stack([
        rank_lib.pool(evaluation_fused[rows], "logmean")
        for _, rows in test_bags
    ])
    return rank_lib.apply_class_offsets(test_probabilities, offsets)


def cross_validate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    players: np.ndarray,
    folds: int,
    train_per_player: int,
    test_per_player: int,
    seed: int,
) -> tuple[str, float, dict[str, float]]:
    """Selects the classifier and blend using player-disjoint folds."""
    logs = np.log(probabilities)
    player_rows = rows_by_group(players)
    player_labels = {
        player: int(labels[rows[0]])
        for player, rows in player_rows.items()
    }
    unique_players = np.asarray(sorted(player_rows))
    fold_of = stratified_folds(unique_players, player_labels, folds, seed)
    model_names = ("logistic_003", "logistic_01", "extra_trees")
    predictions = {name: [] for name in model_names}
    baseline_parts = []
    truth_parts = []
    for fold in range(folds):
        training = {
            player: rows
            for player, rows in player_rows.items()
            if fold_of[player] != fold
        }
        testing = {
            player: rows
            for player, rows in player_rows.items()
            if fold_of[player] == fold
        }
        train_bags = sample_bags(
            training,
            train_per_player,
            seed + fold,
        )
        test_bags = sample_bags(
            testing,
            test_per_player,
            seed + 1000 + fold,
        )
        train_x = bag_features(logs, train_bags)
        test_x = bag_features(logs, test_bags)
        train_y = bag_labels(train_bags, player_labels)
        test_y = bag_labels(test_bags, player_labels)
        train_rows = np.concatenate(list(training.values()))
        baseline_parts.append(
            baseline_probabilities(
                logs,
                labels,
                train_rows,
                train_bags,
                test_bags,
            ),
        )
        truth_parts.append(test_y)
        for name in model_names:
            model = make_model(name, seed + fold)
            model.fit(train_x, train_y)
            predictions[name].append(model.predict_proba(test_x))
        print(f"fold {fold + 1}/{folds} complete", flush=True)
    baseline = np.concatenate(baseline_parts)
    truth = np.concatenate(truth_parts)
    report: dict[str, float] = {
        "baseline": rank_lib.competition_score(
            rank_lib.decode(baseline),
            truth,
        ),
    }
    best_name = ""
    best_weight = 0.0
    best_score = -1.0
    for name, parts in predictions.items():
        learned = np.concatenate(parts)
        report[name] = rank_lib.competition_score(
            rank_lib.decode(learned),
            truth,
        )
        for weight in BLEND_GRID:
            blended = normalize_log_blend(baseline, learned, weight)
            score = rank_lib.competition_score(rank_lib.decode(blended), truth)
            report[f"{name}:blend={weight:.1f}"] = score
            if score > best_score:
                best_name = name
                best_weight = weight
                best_score = score
    report["selected_score"] = best_score
    return best_name, best_weight, report


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Computes a paired question-level bootstrap interval."""
    rng = np.random.default_rng(seed)
    difference = first - second
    samples = np.empty(5000, dtype=np.float64)
    for index in range(len(samples)):
        rows = rng.integers(0, len(difference), len(difference))
        samples[index] = difference[rows].mean()
    return (
        float(difference.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def fit_and_evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    players: np.ndarray,
    holdout_probabilities: np.ndarray,
    holdout_groups: np.ndarray,
    holdout_game_labels: np.ndarray,
    model_name: str,
    blend_weight: float,
    train_per_player: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Fits the locked choice and evaluates it once on holdout questions."""
    logs = np.log(probabilities)
    player_rows = rows_by_group(players)
    player_labels = {
        player: int(labels[rows[0]])
        for player, rows in player_rows.items()
    }
    train_bags = sample_bags(player_rows, train_per_player, seed)
    train_x = bag_features(logs, train_bags)
    train_y = bag_labels(train_bags, player_labels)
    model = make_model(model_name, seed)
    model.fit(train_x, train_y)

    holdout_rows = rows_by_group(holdout_groups)
    holdout_bags = [
        (question, rows)
        for question, rows in sorted(holdout_rows.items())
    ]
    holdout_truth = np.asarray(
        [holdout_game_labels[rows[0]] for _, rows in holdout_bags],
    )
    holdout_logs = np.log(holdout_probabilities)
    learned = model.predict_proba(bag_features(holdout_logs, holdout_bags))
    baseline = baseline_probabilities(
        logs,
        labels,
        np.arange(len(labels)),
        train_bags,
        holdout_bags,
        holdout_logs,
    )
    blended = normalize_log_blend(baseline, learned, blend_weight)
    baseline_utility = utility_scores(rank_lib.decode(baseline), holdout_truth)
    learned_utility = utility_scores(rank_lib.decode(learned), holdout_truth)
    blended_utility = utility_scores(rank_lib.decode(blended), holdout_truth)
    interval = bootstrap_difference(blended_utility, baseline_utility, seed)
    result = {
        "questions": len(holdout_bags),
        "selected_model": model_name,
        "selected_blend_weight": blend_weight,
        "baseline_score": float(baseline_utility.mean()),
        "learned_score": float(learned_utility.mean()),
        "blended_score": float(blended_utility.mean()),
        "delta_vs_baseline": interval[0],
        "delta_95_percent_ci": [interval[1], interval[2]],
    }
    return model, result


def main() -> None:
    """Runs model selection, final fitting, and holdout evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-per-player", type=int, default=80)
    parser.add_argument("--test-per-player", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    probabilities, labels, players = load_validation()
    selected_name, selected_weight, cross_validation = cross_validate(
        probabilities,
        labels,
        players,
        args.folds,
        args.train_per_player,
        args.test_per_player,
        args.seed,
    )
    holdout_probabilities, holdout_groups, holdout_labels = load_holdout()
    model, holdout = fit_and_evaluate(
        probabilities,
        labels,
        players,
        holdout_probabilities,
        holdout_groups,
        holdout_labels,
        selected_name,
        selected_weight,
        args.train_per_player,
        args.seed,
    )
    report = {
        "protocol": {
            "selection": "player-disjoint cross-validation only",
            "holdout_use": "one evaluation after model and blend selection",
            "models": [tag for tag, _ in MODEL_SPECS],
            "folds": args.folds,
            "train_bags_per_player": args.train_per_player,
            "test_bags_per_player": args.test_per_player,
            "seed": args.seed,
        },
        "cross_validation": cross_validation,
        "holdout": holdout,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, args.output / "model.joblib")
    print(json.dumps(report, indent=2))
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
