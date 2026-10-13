"""Compares a dense player-bag checkpoint with production on validation.

The script never reads the locked holdout. It creates identical fixed
5/10/15/20-game questions for the selected production pipeline and the new
bag model, then selects a low-capacity geometric blend on this external,
player-disjoint validation cohort.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.evaluate_validation \
      --checkpoint experiments/dense_player_bags/results/deepsets/model.pt
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
from experiments.dense_player_bags import train_bags
from experiments.hierarchical_move import evaluate_fusion
from experiments.hierarchical_move import train as bag_train
from experiments.model_selection import evaluate as model_selection

QUESTION_SIZES = (5, 10, 15, 20)


@dataclasses.dataclass(frozen=True)
class ValidationContext:
    """Contains immutable arrays and model state shared by question sizes."""

    models: list[torch.nn.Module]
    split: dict[str, np.ndarray]
    fused: np.ndarray
    offsets: np.ndarray
    device: torch.device
    seed: int


def load_model(
    path: pathlib.Path,
    input_dimensions: int,
    device: torch.device,
) -> torch.nn.Module:
    """Restores one inner-validation-selected dense bag checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arguments = argparse.Namespace(**checkpoint["args"])
    model = bag_train.create_model(input_dimensions, arguments, device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


def production_probabilities(
    game_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns validation-aligned production probabilities and offsets."""
    logs, labels, players = model_selection.validation_data(
        model_selection.TWO_MODELS,
    )
    reference_path = rank_lib.ROOT / "results" / "val_probs_d96_h.npz"
    with np.load(reference_path, allow_pickle=True) as reference:
        reference_keys = np.asarray(reference["game_key"], dtype=str)
    if not np.array_equal(game_keys, reference_keys):
        raise ValueError("Dense and production validation games are misaligned")
    weights = blend_eval.fit_weights(logs, labels)
    fused = blend_eval.fuse(logs, weights)
    offsets = model_selection.fit_offsets(fused, players, 0)
    return fused, offsets


def aligned_questions(
    context: ValidationContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds identical production and dense-bag validation questions."""
    data = train_bags.player_data(context.split)
    embeddings: np.ndarray = context.split["embeddings"].astype(np.float32)
    parts = [
        question_size(context, data, embeddings, games_per_bag)
        for games_per_bag in QUESTION_SIZES
    ]
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        np.concatenate([part[3] for part in parts]),
    )


def question_size(
    context: ValidationContext,
    data: bag_train.PlayerData,
    embeddings: np.ndarray,
    games_per_bag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns aligned predictions for one fixed question size."""
    rows, labels = bag_train.fixed_bags(
        data,
        games_per_bag,
        10,
        context.seed,
    )
    production = rank_lib.apply_class_offsets(
        evaluate_fusion.pool_rows(context.fused, rows),
        context.offsets,
    )
    bag_probabilities = predict_ensemble(
        context.models,
        embeddings,
        rows,
        context.device,
    )
    sizes = np.full(len(labels), games_per_bag)
    return production, bag_probabilities, labels, sizes


def predict_ensemble(
    models: list[torch.nn.Module],
    embeddings: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Averages probability outputs from independently trained bag models."""
    if not models:
        raise ValueError("At least one bag checkpoint is required")
    probabilities = [
        bag_train.predict_bags(model, embeddings, rows, device, 512)
        for model in models
    ]
    return np.mean(probabilities, axis=0)


def summarize(
    production: np.ndarray,
    bag_probabilities: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, Any]:
    """Selects a blend and reports aligned validation scores."""
    weight, blend_grid = evaluate_fusion.select_weight(
        production,
        bag_probabilities,
        labels,
    )
    blended = bag_run.normalize_log_blend(
        production,
        bag_probabilities,
        weight,
    )
    return {
        "production_score": rank_lib.competition_score(
            rank_lib.decode(production),
            labels,
        ),
        "bag_score": rank_lib.competition_score(
            rank_lib.decode(bag_probabilities),
            labels,
        ),
        "selected_weight": weight,
        "blended_score": rank_lib.competition_score(
            rank_lib.decode(blended),
            labels,
        ),
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
        "blend_grid": blend_grid,
        "size_conditioned_selection": summarize_by_size(
            production,
            bag_probabilities,
            labels,
            sizes,
        ),
    }


def summarize_by_size(
    production: np.ndarray,
    bag_probabilities: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, Any]:
    """Reports independent, validation-only blend optima for every bag size.

    The main result intentionally keeps one shared blend coefficient. This
    diagnostic exposes whether the known question size is a useful calibration
    variable before a separate size-conditioned method is implemented.
    """
    result: dict[str, Any] = {}
    for games_per_bag in QUESTION_SIZES:
        selected = sizes == games_per_bag
        weight, grid = evaluate_fusion.select_weight(
            production[selected],
            bag_probabilities[selected],
            labels[selected],
        )
        blended = bag_run.normalize_log_blend(
            production[selected],
            bag_probabilities[selected],
            weight,
        )
        result[str(games_per_bag)] = {
            "production_score": rank_lib.competition_score(
                rank_lib.decode(production[selected]),
                labels[selected],
            ),
            "bag_score": rank_lib.competition_score(
                rank_lib.decode(bag_probabilities[selected]),
                labels[selected],
            ),
            "selected_weight": weight,
            "blended_score": rank_lib.competition_score(
                rank_lib.decode(blended),
                labels[selected],
            ),
            "blend_grid": grid,
        }
    return result


def main() -> None:
    """Runs the holdout-free dense bag validation comparison."""
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    split = train_bags.load_split(args.embeddings / "validation.npz")
    models = [
        load_model(path, split["embeddings"].shape[1], device)
        for path in args.checkpoint
    ]
    fused, offsets = production_probabilities(
        split["game_key"].astype(str),
    )
    context = ValidationContext(
        models,
        split,
        fused,
        offsets,
        device,
        args.seed,
    )
    questions = aligned_questions(context)
    report = {
        "protocol": {
            "selection": "fixed external player-disjoint validation",
            "holdout_read": False,
            "question_repeats_per_size": 10,
        },
        "validation": summarize(*questions),
    }
    default_output = args.checkpoint[0].parent / "validation_fusion.json"
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
