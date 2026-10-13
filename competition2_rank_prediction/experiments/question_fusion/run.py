"""Fits low-capacity fusion weights directly on multi-game questions.

Unlike the production fusion, which minimizes per-game NLL, this experiment
fits four shared model weights and ten class biases on sampled 5-20 game bags.
All selection is player-disjoint. Holdout evaluation is opt-in so that model
selection cannot accidentally inspect it.

Example:
    c1/.venv/bin/python -m experiments.question_fusion.run
    c1/.venv/bin/python -m experiments.question_fusion.run --evaluate-holdout
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import scipy.optimize

import blend_eval
import rank_lib
from experiments.bag_distribution import run as bag_run

REGULARIZATION = (0.0, 0.01, 0.1, 1.0)


@dataclasses.dataclass(frozen=True)
class RankData:
    """Contains aligned per-game predictions and player metadata."""

    probabilities: np.ndarray
    labels: np.ndarray
    players: np.ndarray


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Controls player-disjoint question sampling."""

    folds: int
    train_per_player: int
    test_per_player: int
    seed: int


@dataclasses.dataclass(frozen=True)
class FoldContext:
    """Contains immutable arrays and mappings shared by all CV folds."""

    logs: np.ndarray
    labels: np.ndarray
    player_rows: dict[str, np.ndarray]
    player_labels: dict[str, int]
    fold_of: dict[str, int]
    config: ExperimentConfig


def bag_mean_logs(
    logs: np.ndarray,
    bags: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Returns mean class log-probabilities with shape Q x M x C."""
    return np.stack([logs[:, rows].mean(axis=1) for _, rows in bags])


def softmax(values: np.ndarray) -> np.ndarray:
    """Normalizes rows of logits into probabilities."""
    shifted = values - values.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def probabilities_from_parameters(
    mean_logs: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Applies shared nonnegative model weights and per-rank biases."""
    model_count = mean_logs.shape[1]
    weights = parameters[:model_count]
    biases = parameters[model_count:]
    logits = np.tensordot(mean_logs, weights, axes=(1, 0)) + biases
    return softmax(logits)


def fit_question_fusion(
    mean_logs: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fits shared fusion weights and class biases by question-level NLL."""
    model_count = mean_logs.shape[1]
    class_count = mean_logs.shape[2]

    def objective(parameters: np.ndarray) -> float:
        probabilities = probabilities_from_parameters(mean_logs, parameters)
        nll = -np.log(
            np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0),
        ).mean()
        weights = parameters[:model_count]
        biases = parameters[model_count:]
        penalty = (
            np.square(weights).sum()
            + np.square(biases - biases.mean()).sum()
        )
        return float(nll + regularization * penalty / len(parameters))

    initial = np.concatenate(
        (np.full(model_count, 0.5), np.zeros(class_count)),
    )
    bounds = [(0.0, 5.0)] * model_count + [(-3.0, 3.0)] * class_count
    result = scipy.optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Question fusion optimization failed: {result.message}")
    return np.asarray(result.x)


def baseline_probabilities(
    logs: np.ndarray,
    labels: np.ndarray,
    training_rows: np.ndarray,
    train_bags: list[tuple[str, np.ndarray]],
    test_bags: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Builds the existing fusion and offset baseline for one fold."""
    weights = blend_eval.fit_weights(
        logs[:, training_rows],
        labels[training_rows],
    )
    fused = blend_eval.fuse(logs, weights)
    train_probabilities = np.stack([
        rank_lib.pool(fused[rows], "logmean")
        for _, rows in train_bags
    ])
    offsets = rank_lib.fit_class_offsets(train_probabilities)
    test_probabilities = np.stack([
        rank_lib.pool(fused[rows], "logmean")
        for _, rows in test_bags
    ])
    return rank_lib.apply_class_offsets(test_probabilities, offsets)


def cross_validate(
    data: RankData,
    config: ExperimentConfig,
) -> tuple[float, dict[str, object]]:
    """Selects regularization using stratified player-disjoint folds."""
    logs = np.log(data.probabilities)
    player_rows = bag_run.rows_by_group(data.players)
    player_labels = {
        player: int(data.labels[rows[0]])
        for player, rows in player_rows.items()
    }
    unique_players = np.asarray(sorted(player_rows))
    fold_of = bag_run.stratified_folds(
        unique_players,
        player_labels,
        config.folds,
        config.seed,
    )
    context = FoldContext(
        logs,
        data.labels,
        player_rows,
        player_labels,
        fold_of,
        config,
    )
    parts: dict[str, Any] = {
        "baseline": [],
        "truth": [],
        "learned": {value: [] for value in REGULARIZATION},
        "parameters": {value: [] for value in REGULARIZATION},
    }
    for fold in range(config.folds):
        fold_result = evaluate_fold(context, fold)
        parts["baseline"].append(fold_result["baseline"])
        parts["truth"].append(fold_result["truth"])
        for value in REGULARIZATION:
            parts["learned"][value].append(fold_result["learned"][value])
            parts["parameters"][value].append(
                fold_result["parameters"][value],
            )
        print(f"fold {fold + 1}/{config.folds} complete", flush=True)
    return summarize_cross_validation(parts)


def summarize_cross_validation(
    parts: dict[str, Any],
) -> tuple[float, dict[str, object]]:
    """Selects the best regularization from concatenated OOF predictions."""
    truth = np.concatenate(parts["truth"])
    baseline = np.concatenate(parts["baseline"])
    report: dict[str, object] = {
        "baseline": rank_lib.competition_score(
            rank_lib.decode(baseline),
            truth,
        ),
        "regularization": {},
    }
    best_value = 0.0
    best_score = -1.0
    for value in REGULARIZATION:
        probabilities_value = np.concatenate(parts["learned"][value])
        score = rank_lib.competition_score(
            rank_lib.decode(probabilities_value),
            truth,
        )
        report["regularization"][str(value)] = {
            "score": score,
            "fold_parameters": parts["parameters"][value],
        }
        if score > best_score:
            best_score = score
            best_value = value
    report["selected_regularization"] = best_value
    report["selected_score"] = best_score
    return best_value, report


def evaluate_fold(
    context: FoldContext,
    fold: int,
) -> dict[str, Any]:
    """Fits and evaluates all low-capacity candidates for one fold."""
    training = {
        player: rows
        for player, rows in context.player_rows.items()
        if context.fold_of[player] != fold
    }
    testing = {
        player: rows
        for player, rows in context.player_rows.items()
        if context.fold_of[player] == fold
    }
    train_bags = bag_run.sample_bags(
        training,
        context.config.train_per_player,
        context.config.seed + fold,
    )
    test_bags = bag_run.sample_bags(
        testing,
        context.config.test_per_player,
        context.config.seed + 1000 + fold,
    )
    train_labels = bag_run.bag_labels(train_bags, context.player_labels)
    test_labels = bag_run.bag_labels(test_bags, context.player_labels)
    training_rows = np.concatenate(list(training.values()))
    baseline = baseline_probabilities(
        context.logs,
        context.labels,
        training_rows,
        train_bags,
        test_bags,
    )
    learned, parameters_by_value = fit_fold_candidates(
        bag_mean_logs(context.logs, train_bags),
        train_labels,
        bag_mean_logs(context.logs, test_bags),
    )
    return {
        "baseline": baseline,
        "truth": test_labels,
        "learned": learned,
        "parameters": parameters_by_value,
    }


def fit_fold_candidates(
    train_mean_logs: np.ndarray,
    train_labels: np.ndarray,
    test_mean_logs: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, list[float]]]:
    """Fits every regularization candidate for one prepared fold."""
    learned = {}
    parameters_by_value = {}
    for value in REGULARIZATION:
        parameters = fit_question_fusion(
            train_mean_logs,
            train_labels,
            value,
        )
        learned[value] = probabilities_from_parameters(
            test_mean_logs,
            parameters,
        )
        parameters_by_value[value] = parameters.tolist()
    return learned, parameters_by_value


def evaluate_holdout(
    data: RankData,
    regularization: float,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Fits on all validation players and evaluates one locked holdout model."""
    parameters = fit_full_parameters(data, regularization, config)
    holdout_probabilities, groups, game_labels = bag_run.load_holdout()
    holdout_rows = bag_run.rows_by_group(groups)
    holdout_bags = list(sorted(holdout_rows.items()))
    truth = np.asarray([
        game_labels[rows[0]]
        for _, rows in holdout_bags
    ])
    learned = probabilities_from_parameters(
        bag_mean_logs(np.log(holdout_probabilities), holdout_bags),
        parameters,
    )
    prediction = rank_lib.decode(learned)
    return {
        "questions": len(truth),
        "score": rank_lib.competition_score(prediction, truth),
        "parameters": parameters.tolist(),
    }


def fit_full_parameters(
    data: RankData,
    regularization: float,
    config: ExperimentConfig,
) -> np.ndarray:
    """Fits locked question-fusion parameters on every validation player."""
    player_rows = bag_run.rows_by_group(data.players)
    player_labels = {
        player: int(data.labels[rows[0]])
        for player, rows in player_rows.items()
    }
    train_bags = bag_run.sample_bags(
        player_rows,
        config.train_per_player,
        config.seed,
    )
    train_labels = bag_run.bag_labels(train_bags, player_labels)
    return fit_question_fusion(
        bag_mean_logs(np.log(data.probabilities), train_bags),
        train_labels,
        regularization,
    )


def main() -> None:
    """Runs player-disjoint selection and optional holdout evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-per-player", type=int, default=80)
    parser.add_argument("--test-per-player", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help="reuse the saved CV selection and run only locked holdout fitting",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probabilities, labels, players = bag_run.load_validation()
    data = RankData(probabilities, labels, players)
    config = ExperimentConfig(
        args.folds,
        args.train_per_player,
        args.test_per_player,
        args.seed,
    )
    output_path = args.output / "report.json"
    if args.holdout_only:
        if not output_path.exists():
            raise FileNotFoundError(
                f"Cross-validation report does not exist: {output_path}",
            )
        report = json.loads(output_path.read_text(encoding="utf-8"))
        regularization = float(
            report["cross_validation"]["selected_regularization"],
        )
        report["protocol"]["holdout_evaluated"] = True
    else:
        regularization, cross_validation = cross_validate(data, config)
        report = {
            "protocol": {
                "selection": "stratified player-disjoint cross-validation",
                "holdout_evaluated": args.evaluate_holdout,
                "folds": args.folds,
                "train_bags_per_player": args.train_per_player,
                "test_bags_per_player": args.test_per_player,
                "seed": args.seed,
            },
            "cross_validation": cross_validation,
        }
    if args.evaluate_holdout or args.holdout_only:
        report["holdout"] = evaluate_holdout(
            data,
            regularization,
            config,
        )
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
