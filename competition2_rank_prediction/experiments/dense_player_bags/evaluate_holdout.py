"""Runs one locked dense player-bag comparison on the internal holdout.

The blend weight must have been selected without this cohort. This script does
not search weights or train models: it only evaluates the supplied checkpoint
and supplied blend weight.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.evaluate_holdout \
      --checkpoint experiments/dense_player_bags/results/transformer/model.pt \
      --blend-weight 0.24 --device cpu
"""

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import torch

import blend_eval
import rank_lib
from experiments.bag_distribution import run as bag_run
from experiments.dense_player_bags import evaluate_validation
from experiments.dense_player_bags import train_bags
from experiments.hierarchical_move import evaluate_fusion
from experiments.model_selection import evaluate as model_selection


@dataclasses.dataclass(frozen=True)
class BlendConfiguration:
    """Stores a shared fallback and optional question-size blend weights."""

    shared_weight: float
    size_weights: dict[int, float] | None


def production_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Fits the locked production fusion using validation players only."""
    logs, labels, players = model_selection.validation_data(
        model_selection.TWO_MODELS,
    )
    weights = blend_eval.fit_weights(logs, labels)
    fused = blend_eval.fuse(logs, weights)
    offsets = model_selection.fit_offsets(fused, players, 0)
    return weights, offsets


def load_questions() -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray]]]:
    """Loads aligned internal-holdout per-game predictions and question rows."""
    path = rank_lib.ROOT / "features" / "holdout" / "probs_final.npz"
    with np.load(path, allow_pickle=True) as data:
        holdout = {key: data[key] for key in data.files}
    groups = bag_run.rows_by_group(holdout["player_id"].astype(str))
    return holdout, list(sorted(groups.items()))


def production_questions(
    holdout: dict[str, np.ndarray],
    questions: list[tuple[str, np.ndarray]],
    weights: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Pools locked production predictions into question probabilities."""
    logs = np.stack([
        np.log(np.clip(holdout[model], 1e-6, 1.0))
        for model in model_selection.TWO_MODELS
    ])
    fused = blend_eval.fuse(logs, weights)
    pooled = np.stack([
        rank_lib.pool(fused[rows], "logmean")
        for _, rows in questions
    ])
    return rank_lib.apply_class_offsets(pooled, offsets)


def bag_questions(
    models: list[torch.nn.Module],
    embeddings: np.ndarray,
    questions: list[tuple[str, np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicts dense bag probabilities, grouping variable-size questions."""
    sizes = np.asarray([len(rows) for _, rows in questions])
    probabilities = np.empty((len(questions), len(rank_lib.LEVELS)))
    for size in np.unique(sizes):
        selected = np.flatnonzero(sizes == size)
        rows = np.stack([questions[index][1] for index in selected])
        probabilities[selected] = evaluate_validation.predict_ensemble(
            models,
            embeddings,
            rows,
            device,
        )
    return probabilities, sizes


def label_questions(questions: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Returns one rank label for every internal holdout question."""
    _, _, game_labels = bag_run.load_holdout()
    return np.asarray([game_labels[rows[0]] for _, rows in questions])


def summarize(
    production: np.ndarray,
    bag_probabilities: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
    configuration: BlendConfiguration,
) -> dict[str, Any]:
    """Scores baseline and locked blend with paired bootstrap uncertainty."""
    blended = blend_probabilities(
        production,
        bag_probabilities,
        sizes,
        configuration,
    )
    baseline_utility = bag_run.utility_scores(rank_lib.decode(production), labels)
    blended_utility = bag_run.utility_scores(rank_lib.decode(blended), labels)
    delta = model_selection.bootstrap_interval(
        blended_utility - baseline_utility,
        20261010,
    )
    return {
        "production_score": float(baseline_utility.mean()),
        "bag_score": rank_lib.competition_score(
            rank_lib.decode(bag_probabilities),
            labels,
        ),
        "blended_score": float(blended_utility.mean()),
        "delta": delta[0],
        "delta_95_percent_ci": [delta[1], delta[2]],
        "production_by_games": evaluate_fusion.score_by_size(
            production,
            labels,
            sizes,
        ),
        "bag_by_games": evaluate_fusion.score_by_size(
            bag_probabilities,
            labels,
            sizes,
        ),
        "blended_by_games": evaluate_fusion.score_by_size(
            blended,
            labels,
            sizes,
        ),
    }


def blend_probabilities(
    production: np.ndarray,
    bag_probabilities: np.ndarray,
    sizes: np.ndarray,
    configuration: BlendConfiguration,
) -> np.ndarray:
    """Returns one shared or validation-locked size-conditioned blend."""
    if configuration.size_weights is None:
        return bag_run.normalize_log_blend(
            production,
            bag_probabilities,
            configuration.shared_weight,
        )
    blended = np.empty_like(production)
    for size in np.unique(sizes):
        selected = sizes == size
        # The official task uses 5/10/15/20 only. Internal probes include
        # intervening sizes, for which the shared validation-selected weight
        # is the conservative fallback rather than an extrapolated weight.
        weight = configuration.size_weights.get(
            int(size),
            configuration.shared_weight,
        )
        blended[selected] = bag_run.normalize_log_blend(
            production[selected],
            bag_probabilities[selected],
            weight,
        )
    return blended


def parse_size_weights(raw: str | None) -> dict[int, float] | None:
    """Parses a comma-separated, validation-locked size-to-weight mapping."""
    if raw is None:
        return None
    result: dict[int, float] = {}
    for item in raw.split(","):
        try:
            size_text, weight_text = item.split("=", maxsplit=1)
            size = int(size_text)
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(
                "size weights must use '5=0.12,10=0.28,15=0.30,20=0.36'",
            ) from error
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"Blend weight for question size {size} must be in [0, 1]",
            )
        result[size] = weight
    if not result:
        raise ValueError("At least one size-specific blend weight is required")
    return result


def run_evaluation(
    embeddings_path: pathlib.Path,
    checkpoint_paths: list[pathlib.Path],
    configuration: BlendConfiguration,
    device: torch.device,
) -> dict[str, Any]:
    """Builds aligned holdout predictions and returns the locked comparison."""
    split = train_bags.load_split(embeddings_path / "holdout.npz")
    holdout, questions = load_questions()
    if not np.array_equal(split["game_key"].astype(str), holdout["game_key"]):
        raise ValueError("Dense and production holdout games are misaligned")
    models = [
        evaluate_validation.load_model(
            path,
            split["embeddings"].shape[1],
            device,
        )
        for path in checkpoint_paths
    ]
    weights, offsets = production_parameters()
    production = production_questions(holdout, questions, weights, offsets)
    bag_probabilities, sizes = bag_questions(
        models,
        split["embeddings"].astype(np.float32),
        questions,
        device,
    )
    labels = label_questions(questions)
    return summarize(
        production,
        bag_probabilities,
        labels,
        sizes,
        configuration,
    )


def main() -> None:
    """Scores one pre-locked dense bag blend on the internal holdout."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        action="append",
        required=True,
        help="One or more checkpoints; probabilities are averaged.",
    )
    parser.add_argument("--blend-weight", type=float, required=True)
    parser.add_argument(
        "--size-weights",
        help="Validation-locked mapping, e.g. 5=0.12,10=0.28,15=0.30,20=0.36",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("blend weight must be between zero and one")
    size_weights = parse_size_weights(args.size_weights)
    configuration = BlendConfiguration(args.blend_weight, size_weights)
    device = torch.device(args.device)
    report = {
        "protocol": {
            "selection": "validation-locked blend weight",
            "blend_weight": configuration.shared_weight,
            "size_weights": size_weights,
            "checkpoint_count": len(args.checkpoint),
            "holdout_evaluations": 1,
        },
        "holdout": run_evaluation(
            args.embeddings,
            args.checkpoint,
            configuration,
            device,
        ),
    }
    default_output = args.checkpoint[0].parent / "holdout_fusion.json"
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
