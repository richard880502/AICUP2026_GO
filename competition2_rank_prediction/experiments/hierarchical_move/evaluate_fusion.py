"""Tests whether hierarchical move-bag predictions complement production.

The blend coefficient is selected only on the fixed player-disjoint validation
cohort. The locked coefficient is then evaluated once on the 400-player
holdout, with paired bootstrap uncertainty against the production baseline.

Example:
    c1/.venv/bin/python -m \
      experiments.hierarchical_move.evaluate_fusion --device cuda:1
"""

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import torch

import blend_eval
import move_transformer
import rank_lib
from experiments.bag_distribution import run as bag_run
from experiments.hierarchical_move import extract_embeddings
from experiments.hierarchical_move import train
from experiments.model_selection import evaluate as model_selection

BLEND_WEIGHTS = tuple(np.linspace(0.0, 0.5, 26))


def load_model(
    path: pathlib.Path,
    input_dimensions: int,
    device: torch.device,
) -> train.BagRanker:
    """Restores the selected inner-validation checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arguments = checkpoint["args"]
    model = train.BagRanker(
        input_dimensions,
        arguments["dimensions"],
        arguments["heads"],
        arguments["layers"],
        arguments["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


def production_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits the selected two-model production weights and offsets."""
    models = model_selection.TWO_MODELS
    logs, labels, players = model_selection.validation_data(models)
    weights = blend_eval.fit_weights(logs, labels)
    fused = blend_eval.fuse(logs, weights)
    offsets = model_selection.fit_offsets(fused, players, 0)
    return weights, offsets, fused


def pool_rows(probabilities: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Pools per-game predictions for prepared equal-length bags."""
    return np.stack([
        rank_lib.pool(probabilities[bag_rows], "logmean")
        for bag_rows in rows
    ])


def validation_predictions(
    model: train.BagRanker,
    embeddings: np.ndarray,
    players: np.ndarray,
    ranks: np.ndarray,
    game_keys: np.ndarray,
    validation_directory: pathlib.Path,
    fused: np.ndarray,
    offsets: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds aligned production and hierarchy validation questions."""
    player_data = train.external_validation_data(
        players,
        ranks,
        game_keys,
        validation_directory,
    )
    validation = rank_lib.load_features(validation_directory)
    local_row = {
        key: index
        for index, key in enumerate(validation["game_key"])
    }
    production_parts = []
    hierarchy_parts = []
    label_parts = []
    size_parts = []
    for games_per_bag in (5, 10, 15, 20):
        global_rows, labels = train.fixed_bags(
            player_data,
            games_per_bag,
            10,
            seed,
        )
        local_rows = np.asarray([
            [local_row[game_keys[row]] for row in bag]
            for bag in global_rows
        ])
        production = rank_lib.apply_class_offsets(
            pool_rows(fused, local_rows),
            offsets,
        )
        hierarchy = train.predict_bags(
            model,
            embeddings,
            global_rows,
            device,
            512,
        )
        production_parts.append(production)
        hierarchy_parts.append(hierarchy)
        label_parts.append(labels)
        size_parts.append(np.full(len(labels), games_per_bag))
    return (
        np.concatenate(production_parts),
        np.concatenate(hierarchy_parts),
        np.concatenate(label_parts),
        np.concatenate(size_parts),
    )


def select_weight(
    production: np.ndarray,
    hierarchy: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Selects one low-capacity blend coefficient on validation only."""
    scores = {}
    for weight in BLEND_WEIGHTS:
        probabilities = bag_run.normalize_log_blend(
            production,
            hierarchy,
            weight,
        )
        scores[f"{weight:.2f}"] = rank_lib.competition_score(
            rank_lib.decode(probabilities),
            labels,
        )
    selected = max(BLEND_WEIGHTS, key=lambda value: scores[f"{value:.2f}"])
    return selected, scores


def holdout_predictions(
    model: train.BagRanker,
    weights: np.ndarray,
    offsets: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds aligned production and hierarchy holdout probabilities."""
    path = rank_lib.ROOT / "features" / "holdout" / "probs_final.npz"
    with np.load(path, allow_pickle=True) as data:
        holdout = {key: data[key] for key in data.files}
    rows_by_question = bag_run.rows_by_group(
        holdout["player_id"].astype(str),
    )
    bags = list(sorted(rows_by_question.items()))
    holdout_logs = np.stack([
        np.log(np.clip(holdout[model_name], 1e-6, 1.0))
        for model_name in model_selection.TWO_MODELS
    ])
    fused = blend_eval.fuse(holdout_logs, weights)
    production = rank_lib.apply_class_offsets(
        np.stack([
            rank_lib.pool(fused[rows], "logmean")
            for _, rows in bags
        ]),
        offsets,
    )
    holdout_embeddings = encode_holdout_games(device)
    hierarchy_parts = []
    sizes = np.asarray([len(rows) for _, rows in bags])
    for size in np.unique(sizes):
        selected = np.flatnonzero(sizes == size)
        rows = np.stack([bags[index][1] for index in selected])
        probabilities = train.predict_bags(
            model,
            holdout_embeddings,
            rows,
            device,
            512,
        )
        hierarchy_parts.append((selected, probabilities))
    hierarchy = np.empty_like(production)
    for selected, probabilities in hierarchy_parts:
        hierarchy[selected] = probabilities
    _, _, game_labels = bag_run.load_holdout()
    labels = np.asarray([
        game_labels[rows[0]]
        for _, rows in bags
    ])
    return production, hierarchy, labels, sizes


def encode_holdout_games(device: torch.device) -> np.ndarray:
    """Encodes official-format holdout games with the frozen safe encoder."""
    moves_path = (
        rank_lib.ROOT / "features" / "moves" / "holdout_moves.npz"
    )
    with np.load(moves_path, allow_pickle=True) as data:
        tokens = data["tokens"][:, :256]
        lengths = np.minimum(data["length"], 256).astype(np.int16)
        colours = data["colour"].astype(np.int8)
    encoder = move_transformer.MoveTransformer().to(device)
    checkpoint = torch.load(
        rank_lib.ROOT / "results" / "move_h.pt",
        map_location="cpu",
        weights_only=False,
    )
    encoder.load_state_dict(checkpoint["state"])
    encoder.eval()
    parts = []
    for start in range(0, len(tokens), 1024):
        parts.append(
            extract_embeddings.extract_batch(
                encoder,
                tokens[start:start + 1024],
                lengths[start:start + 1024],
                colours[start:start + 1024],
                device,
            ),
        )
    return np.concatenate(parts)


def score_by_size(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, float]:
    """Reports competition score for each official question size."""
    prediction = rank_lib.decode(probabilities)
    return {
        str(size): rank_lib.competition_score(
            prediction[sizes == size],
            labels[sizes == size],
        )
        for size in np.unique(sizes)
    }


def evaluate_holdout(
    production: np.ndarray,
    hierarchy: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
    weight: float,
) -> dict[str, Any]:
    """Scores the locked blend and paired uncertainty on holdout."""
    blended = bag_run.normalize_log_blend(production, hierarchy, weight)
    baseline_prediction = rank_lib.decode(production)
    blended_prediction = rank_lib.decode(blended)
    baseline_utility = bag_run.utility_scores(baseline_prediction, labels)
    blended_utility = bag_run.utility_scores(blended_prediction, labels)
    interval = model_selection.bootstrap_interval(
        blended_utility - baseline_utility,
        20261006,
    )
    return {
        "production_score": float(baseline_utility.mean()),
        "blended_score": float(blended_utility.mean()),
        "delta": interval[0],
        "delta_95_percent_ci": [interval[1], interval[2]],
        "production_by_games": score_by_size(production, labels, sizes),
        "blended_by_games": score_by_size(blended, labels, sizes),
    }


def main() -> None:
    """Selects the validation blend and performs one locked holdout test."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "move_embeddings_h",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results" / "model.pt",
    )
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat" / "val",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20271006)
    args = parser.parse_args()
    device = torch.device(args.device)
    embeddings = np.load(
        args.embeddings / "embeddings.npy",
        mmap_mode="r",
    )
    players, ranks, game_keys = train.load_metadata(args.embeddings)
    model = load_model(args.checkpoint, embeddings.shape[1], device)
    weights, offsets, fused = production_parameters()
    validation = validation_predictions(
        model,
        embeddings,
        players,
        ranks,
        game_keys,
        args.validation,
        fused,
        offsets,
        device,
        args.seed,
    )
    selected_weight, validation_scores = select_weight(
        validation[0],
        validation[1],
        validation[2],
    )
    holdout = holdout_predictions(
        model,
        weights,
        offsets,
        device,
    )
    report = {
        "protocol": {
            "blend_selected_on": "400-player external validation only",
            "holdout_evaluations": 1,
            "hierarchy_checkpoint_selected_on": "300-player inner validation",
        },
        "validation": {
            "selected_weight": selected_weight,
            "scores": validation_scores,
            "production_by_games": score_by_size(
                validation[0],
                validation[2],
                validation[3],
            ),
            "hierarchy_by_games": score_by_size(
                validation[1],
                validation[2],
                validation[3],
            ),
        },
        "holdout": evaluate_holdout(*holdout, selected_weight),
    }
    output = pathlib.Path(__file__).resolve().parent / "results"
    path = output / "fusion_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
