"""Tests class-count-constrained decoding on validation players only.

Competition questions may be rank-balanced, but that is not assumed by the
default inference pipeline. This experiment creates independent 400-player
validation cohorts with 40 players per rank and compares per-question decoding
against a Hungarian assignment with a known uniform rank count. It never reads
the holdout cohort.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.evaluate_global_decode_validation \
      --device cpu
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import scipy.optimize
import torch

from experiments.bag_distribution import run as bag_run
from experiments.dense_player_bags import evaluate_validation
from experiments.dense_player_bags import train_bags
from experiments.hierarchical_move import evaluate_fusion
from experiments.hierarchical_move import train as bag_train

import rank_lib


QUESTION_SIZES = (5, 10, 15, 20)


@dataclasses.dataclass(frozen=True)
class InputPaths:
    """Contains validation embeddings and their matching bag models."""

    original_embeddings: pathlib.Path
    move_embeddings: pathlib.Path
    original_checkpoint: pathlib.Path
    move_checkpoint: pathlib.Path


@dataclasses.dataclass(frozen=True)
class LockedWeights:
    """Contains blend weights selected before this validation experiment."""

    move_bag_weight: float
    production_weight: float


@dataclasses.dataclass(frozen=True)
class EvaluationConfiguration:
    """Contains immutable validation experiment settings."""

    paths: InputPaths
    weights: LockedWeights
    repeats: int
    seed: int
    device: torch.device


@dataclasses.dataclass(frozen=True)
class ModelBranch:
    """Pairs one feature matrix with the bag model trained on it."""

    split: dict[str, np.ndarray]
    model: torch.nn.Module


@dataclasses.dataclass(frozen=True)
class ValidationState:
    """Contains aligned inputs shared by every validation cohort."""

    original: ModelBranch
    move: ModelBranch
    player_data: bag_train.PlayerData
    production_games: np.ndarray
    offsets: np.ndarray


def _load_model(
    embeddings: dict[str, np.ndarray],
    checkpoint: pathlib.Path,
    device: torch.device,
) -> torch.nn.Module:
    """Restores one bag model for its matching embedding dimensions."""
    return evaluate_validation.load_model(
        checkpoint,
        embeddings["embeddings"].shape[1],
        device,
    )


def _balanced_decode(probabilities: np.ndarray) -> np.ndarray:
    """Returns the best uniform-count assignment under competition utility."""
    questions, classes = probabilities.shape
    if questions % classes != 0:
        raise ValueError("Uniform decoding requires a multiple of ten questions")
    utility = probabilities @ _competition_weights(classes)
    rows, columns = scipy.optimize.linear_sum_assignment(
        -np.repeat(utility, questions // classes, axis=1),
    )
    prediction = np.empty(questions, dtype=np.int64)
    prediction[rows] = columns // (questions // classes)
    return prediction


def _competition_weights(classes: int) -> np.ndarray:
    """Builds the task's exact-or-adjacent rank utility matrix."""
    identity = np.eye(classes)
    return identity + rank_lib.ADJACENT * (
        np.eye(classes, k=1) + np.eye(classes, k=-1)
    )


def _score(prediction: np.ndarray, labels: np.ndarray) -> float:
    """Computes the official score for a single validation cohort."""
    return rank_lib.competition_score(prediction, labels)


def _check_alignment(
    original: dict[str, np.ndarray],
    move: dict[str, np.ndarray],
) -> None:
    """Ensures both branch embeddings refer to identical validation games."""
    if not np.array_equal(
        original["game_key"].astype(str),
        move["game_key"].astype(str),
    ):
        raise ValueError("Original and move validation games are misaligned")


def _cross_probabilities(
    state: ValidationState,
    rows: np.ndarray,
    production: np.ndarray,
    configuration: EvaluationConfiguration,
) -> np.ndarray:
    """Returns locked production-plus-cross-feature bag probabilities."""
    original_bag = evaluate_validation.predict_ensemble(
        [state.original.model],
        state.original.split["embeddings"].astype(np.float32),
        rows,
        configuration.device,
    )
    move_bag = evaluate_validation.predict_ensemble(
        [state.move.model],
        state.move.split["embeddings"].astype(np.float32),
        rows,
        configuration.device,
    )
    cross_bag = bag_run.normalize_log_blend(
        original_bag,
        move_bag,
        configuration.weights.move_bag_weight,
    )
    return bag_run.normalize_log_blend(
        production,
        cross_bag,
        configuration.weights.production_weight,
    )


def _cohort_scores(
    configuration: EvaluationConfiguration,
    state: ValidationState,
    cohorts: np.ndarray,
    label_cohorts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scores every independent rank-balanced cohort."""
    individual_scores = []
    balanced_scores = []
    for repeat in range(configuration.repeats):
        cohort_rows = cohorts[:, repeat]
        cohort_labels = label_cohorts[:, repeat]
        production = rank_lib.apply_class_offsets(
            evaluate_fusion.pool_rows(state.production_games, cohort_rows),
            state.offsets,
        )
        probabilities = _cross_probabilities(
            state,
            cohort_rows,
            production,
            configuration,
        )
        individual_scores.append(_score(rank_lib.decode(probabilities), cohort_labels))
        balanced_scores.append(_score(_balanced_decode(probabilities), cohort_labels))
    return np.asarray(individual_scores), np.asarray(balanced_scores)


def _evaluate_size(
    configuration: EvaluationConfiguration,
    state: ValidationState,
    games_per_question: int,
) -> dict[str, Any]:
    """Scores individual and uniform-count decodes for one question size."""
    rows, labels = bag_train.fixed_bags(
        state.player_data,
        games_per_question,
        configuration.repeats,
        configuration.seed,
    )
    cohorts = rows.reshape(
        len(state.player_data.rows),
        configuration.repeats,
        -1,
    )
    label_cohorts = labels.reshape(
        len(state.player_data.rows),
        configuration.repeats,
    )
    individual, balanced = _cohort_scores(
        configuration,
        state,
        cohorts,
        label_cohorts,
    )
    return {
        "individual_score": float(individual.mean()),
        "balanced_score": float(balanced.mean()),
        "balanced_minus_individual": float((balanced - individual).mean()),
        "repeat_score_range": {
            "individual": [float(individual.min()), float(individual.max())],
            "balanced": [float(balanced.min()), float(balanced.max())],
        },
    }


def run_evaluation(
    configuration: EvaluationConfiguration,
) -> dict[str, Any]:
    """Runs all validation-only uniform-count decoder comparisons."""
    original = train_bags.load_split(
        configuration.paths.original_embeddings / "validation.npz",
    )
    move = train_bags.load_split(
        configuration.paths.move_embeddings / "validation.npz",
    )
    _check_alignment(original, move)
    original_model = _load_model(
        original,
        configuration.paths.original_checkpoint,
        configuration.device,
    )
    move_model = _load_model(
        move,
        configuration.paths.move_checkpoint,
        configuration.device,
    )
    production_games, offsets = evaluate_validation.production_probabilities(
        original["game_key"].astype(str),
    )
    player_data = train_bags.player_data(original)
    rank_counts = [len(players) for players in player_data.by_rank]
    if len(set(rank_counts)) != 1:
        raise ValueError("Validation players are not equally represented by rank")
    state = ValidationState(
        original=ModelBranch(original, original_model),
        move=ModelBranch(move, move_model),
        player_data=player_data,
        production_games=production_games,
        offsets=offsets,
    )
    return {
        "players": len(player_data.rows),
        "players_per_rank": rank_counts[0],
        "by_games": {
            str(games): _evaluate_size(
                configuration,
                state,
                games,
            )
            for games in QUESTION_SIZES
        },
    }


def main() -> None:
    """Evaluates uniform-count decoding without accessing holdout data."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--original-embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings_logits",
    )
    parser.add_argument(
        "--move-embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings_logits_moves",
    )
    parser.add_argument(
        "--original-checkpoint",
        type=pathlib.Path,
        default=experiment / "results" / "transformer_logits" / "model.pt",
    )
    parser.add_argument(
        "--move-checkpoint",
        type=pathlib.Path,
        default=(
            experiment / "results" / "transformer_logits_moves" / "model.pt"
        ),
    )
    parser.add_argument("--move-bag-weight", type=float, default=0.30)
    parser.add_argument("--production-weight", type=float, default=0.28)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            experiment
            / "results"
            / "global_decode_validation"
            / "report.json"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    for name in ("move_bag_weight", "production_weight"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    configuration = EvaluationConfiguration(
        paths=InputPaths(
            original_embeddings=args.original_embeddings,
            move_embeddings=args.move_embeddings,
            original_checkpoint=args.original_checkpoint,
            move_checkpoint=args.move_checkpoint,
        ),
        weights=LockedWeights(
            move_bag_weight=args.move_bag_weight,
            production_weight=args.production_weight,
        ),
        repeats=args.repeats,
        seed=args.seed,
        device=torch.device(args.device),
    )
    report = {
        "protocol": {
            "cohort": "external player-disjoint validation only",
            "holdout_read": False,
            "known_uniform_rank_count": True,
            "official_rank_balance_assumed": False,
            "repeats": args.repeats,
            "move_bag_weight": args.move_bag_weight,
            "production_weight": args.production_weight,
        },
        "validation": run_evaluation(configuration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
