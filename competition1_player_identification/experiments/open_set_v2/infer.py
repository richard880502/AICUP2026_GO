"""Replays frozen cohort inference without fitting or reading outcome labels.

Use cohort_scores(root, recipe, "validation") to verify the final inference
implementation before sealing. Final-cohort inference requires an existing
matching lock and is driven by final_evaluate.py only after validation replay.
"""

import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import bank
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    decision_rank,
)
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import models
from competition1_player_identification.experiments.open_set_v2 import pair_data
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)
from competition1_player_identification.experiments.open_set_v2 import (
    pair_model,
)
from competition1_player_identification.experiments.open_set_v2 import prefix
from competition1_player_identification.experiments.open_set_v2 import (
    rank_prior,
)
from competition1_player_identification.experiments.open_set_v2 import training


def positions(root: pathlib.Path, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    """Maps fixed global game positions to the immutable selected-game table."""
    if cohort not in ("validation", "final_holdout"):
        raise ValueError(f"Unknown evaluation cohort: {cohort!r}")
    if (
        cohort == "final_holdout"
        and not (root / "locked_configuration.json").exists()
    ):
        raise ValueError(
            "Final inference requires a complete prior configuration lock"
        )
    selected = np.load(root / "engine_inputs/global_indices.npy")
    with np.load(root / "clean_splits.npz") as splits:
        global_rows = np.asarray(splits[cohort])
    local_rows = np.searchsorted(selected, global_rows)
    if (local_rows >= len(selected)).any() or not np.array_equal(
        selected[local_rows], global_rows
    ):
        raise ValueError(
            "Cohort games are absent from the selected feature corpus"
        )
    return global_rows, local_rows


def encoder_values(context: tuple) -> np.ndarray:
    """Encodes exactly 9,000 cohort games with saved train-only normalization."""
    root, name, local_rows, global_rows = context
    checkpoint = torch.load(
        root / f"encoder_{name}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    specification = models.Specification(**checkpoint["specification"])
    raw = dataset.load_arrays(root / "raw_cache", ("colors",))
    if specification.kind == "sequence":
        values = dataset.load_arrays(root / "raw_cache", ("tokens",))["tokens"]
        values = values[global_rows.ravel(), : specification.length]
    else:
        profiles = np.load(
            root / f"{specification.kind}_profiles.npy", mmap_mode="r"
        )
        values = profiles[local_rows.ravel()]
        values = np.clip(
            (values - checkpoint["mean"]) / checkpoint["scale"], -10, 10
        ).astype(np.float32)
    data = training.GameData(
        values,
        raw["colors"][global_rows.ravel()].astype(np.int64),
        [],
        np.full(len(values), -1, np.int64),
        np.zeros((0, 15), np.int64),
        checkpoint["mean"],
        checkpoint["scale"],
    )
    runtime = bank.frozen_runtime(checkpoint, data, "cuda:0")
    return np.reshape(
        training.encode(runtime, np.arange(len(values))), (600, 15, -1)
    )


def rank_values(root: pathlib.Path, local_rows: np.ndarray) -> np.ndarray:
    """Uses the cleanly trained C2 raw rank checkpoint and fixed temperature."""
    checkpoint = torch.load(
        root / "rank_prior/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = rank_prior.C2MoveRanker().cuda(0)
    model.load_state_dict(checkpoint["state"])
    selected = np.load(root / "engine_inputs/global_indices.npy")
    global_rows = selected[local_rows.ravel()]
    raw = dataset.load_arrays(root / "raw_cache", ("tokens", "colors"))
    data = rank_prior.RankData(
        raw["tokens"][global_rows, :256],
        raw["colors"][global_rows].astype(np.int64),
        np.full(len(global_rows), -1, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros((0, 15), np.int64),
    )
    probabilities = rank_prior.predict(model, data, np.arange(len(global_rows)))
    probabilities = probabilities.clip(min=1e-7) ** (
        1 / checkpoint["selection"]["temperature"]
    )
    probabilities /= probabilities.sum(1, keepdims=True)
    return np.reshape(probabilities, (600, 15, 10))


def pair_logits(
    root: pathlib.Path,
    pairs: pair_data.PairDataset,
    branch: str,
) -> np.ndarray:
    """Applies serialized pair normalization and weights without any answers."""
    checkpoint = torch.load(
        root / f"matching/pair_{branch}.pt",
        map_location="cpu",
        weights_only=False,
    )
    values = pair_model.normalize_features(
        pairs.features[:, :, checkpoint["columns"]],
        checkpoint["mean"],
        checkpoint["scale"],
    )
    model = pair_model.PairReranker(len(checkpoint["columns"])).cuda(0)
    model.load_state_dict(checkpoint["state"])
    return pair_model.predict(model, values, "cuda:0")


def decision_rank_values(
    root: pathlib.Path, local_rows: np.ndarray
) -> np.ndarray:
    """Replays the strong C2 decision-rank ensemble without reading grades."""
    values, masks = decision_rank.load_tokens(root)
    chosen = local_rows.ravel()
    with np.load(root / "rank_engine/preprocessing.npz") as preprocessing:
        normalized = decision_rank.normalize(
            values[chosen],
            masks[chosen],
            preprocessing["mean"],
            preprocessing["scale"],
        )
    data = decision_rank.RankData(
        normalized,
        masks[chosen],
        np.full(len(chosen), -1, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros((0, 15), np.int64),
    )
    predictions = decision_rank.ensemble(
        root / "rank_engine", data, np.arange(len(chosen))
    )
    temperature = json.loads((root / "rank_engine/completed.json").read_text())[
        "temperature"
    ]
    predictions = predictions.clip(min=1e-7) ** (1 / temperature)
    predictions /= predictions.sum(1, keepdims=True)
    return np.reshape(predictions, (600, 15, 10))


def matching_scores(context: tuple) -> dict:
    """Applies the saved reranker to a frozen 600-player bank with no answers."""
    root, local_rows, recipe = context
    first = bank.encode_checkpoint(root / "encoder_engine_512/checkpoint.pt")
    second = bank.encode_checkpoint(root / "encoder_engine_1024/checkpoint.pt")
    reference = pair_features.build_bank(first, second, local_rows)
    rows = []
    for count in evaluation.COUNTS:
        for query in range(600):
            task = pair_features.Task(query, count, np.arange(600))
            rows.append(
                pair_features.make_rows(
                    (
                        reference,
                        task,
                        recipe["stage_weight_b"],
                        "both",
                    )
                )
            )
    pairs = pair_data.assemble(rows, list(np.repeat(evaluation.COUNTS, 600)))
    # Answers are withheld even for validation replay and final evaluation.
    if (pairs.targets != -1).any() or (pairs.truth_positions != -1).any():
        raise ValueError(
            "Inference features unexpectedly contain answer labels"
        )
    stage = np.reshape(pairs.scores, (4, 600, 600))
    logits = pair_logits(root, pairs, recipe["branch"])
    return {
        "engine_ensemble": stage,
        "learned_pair": pair_features.reorder_scores(
            stage, pairs.top_ids, logits, recipe["blend"]
        ),
    }


def fisher_scores(root: pathlib.Path, rows: np.ndarray) -> np.ndarray:
    """Applies only serialized baseline statistics, never re-estimating them."""
    with np.load(root / "fisher_parameters.npz") as values:
        mean, scale, weights = (
            np.asarray(values[key]) for key in ("mean", "scale", "weights")
        )
    profiles = np.load(root / "engine_profiles.npy", mmap_mode="r")
    normalized = np.clip((profiles[rows] - mean) / scale, -10, 10) * weights
    return evaluation.similarity_scores(
        normalized.astype(np.float32), "cuda:0"
    )["centroid"]


def cohort_scores(
    root: pathlib.Path, recipe: dict, cohort: str
) -> tuple[dict, np.ndarray]:
    """Computes frozen source matrices and the fixed baseline for one cohort."""
    global_rows, local_rows = positions(root, cohort)
    required = set(recipe["weights"])
    matrices = {}
    for name, aggregation in recipe["aggregations"].items():
        if name in required:
            values = encoder_values((root, name, local_rows, global_rows))
            matrices[name] = evaluation.similarity_scores(values, "cuda:0")[
                aggregation
            ]
    if "prefix" in required:
        raw = dataset.load_arrays(root / "raw_cache", ("tokens", "colors"))
        hashes = prefix.fingerprints(
            raw["tokens"][global_rows.ravel()],
            raw["colors"][global_rows.ravel()],
        )
        matrices["prefix"] = prefix.score_prefixes(
            np.reshape(hashes, (600, 15, -1))
        )
    if "rank_prior" in required:
        probabilities = rank_values(root, local_rows)
        matrices["rank_prior"] = evaluation.rank_affinity(probabilities)
    if "rank_engine" in required:
        matrices["rank_engine"] = evaluation.rank_affinity(
            decision_rank_values(root, local_rows)
        )
    if required & {"engine_ensemble", "learned_pair"}:
        matched = matching_scores((root, local_rows, recipe["matching"]))
        for name in required & set(matched):
            matrices[name] = matched[name]
    if set(matrices) != required:
        raise ValueError(
            f"Missing selected inference sources: {required - set(matrices)}"
        )
    return matrices, fisher_scores(root, local_rows)
