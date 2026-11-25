"""Loads exact inputs and differentiably scores all fixed candidate policies.

game_scores never reads target ranks; every candidate is scored for each game.
"""

import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_finetune import data
from competition2_rank_prediction.experiments.policy_adaptation import (
    likelihood,
)


def load(root: pathlib.Path, split: str) -> dict:
    """Requires complete spatial/legal caches with identical game alignment."""
    spatial = root / "production/human_finetune_inputs_20261109"
    legal = root / "production/policy_legal_inputs_20261109"
    if not (spatial / "COMPLETE").exists() or not (legal / "COMPLETE").exists():
        raise ValueError("Spatial or legal cache is incomplete")
    payload = data.load(spatial / split)
    arrays: dict[str, list[np.ndarray]] = {}
    for path in sorted((legal / split).glob("legal_*.npz")):
        with np.load(path) as cache:
            for name in ("game_key", "points", "valid", "legal_packed"):
                arrays.setdefault(name, []).append(np.asarray(cache[name]))
    if not arrays:
        raise ValueError("No legal-support shards found")
    combined = {name: np.concatenate(values) for name, values in arrays.items()}
    for name in ("game_key", "points", "valid"):
        if not np.array_equal(payload[name], combined[name]):
            raise ValueError(f"Misaligned legal/spatial data: {name}")
    payload["legal_packed"] = combined["legal_packed"]
    return payload


def game_scores(context: dict, payload: dict, rows: np.ndarray) -> torch.Tensor:
    """Returns game-by-ten-candidate mean legal played-move log probabilities."""
    tensors = data.batch(payload, rows, context["device"])
    points = tensors[3].flatten()
    legal = torch.as_tensor(
        np.unpackbits(payload["legal_packed"][rows], axis=-1)[..., :362],
        device=context["device"],
        dtype=torch.bool,
    ).reshape(-1, 362)
    output = context["model"](
        tensors[0].repeat(10, 1, 1, 1),
        tensors[1].repeat(10, 1),
        input_meta=context["profiles"].repeat_interleave(len(points), dim=0),
    )
    played = likelihood.played_log_probability(
        output[0][0][:, 0], points.repeat(10), legal.repeat(10, 1)
    ).reshape(10, len(rows), 8)
    pooled = (played * tensors[4][None]).sum(2).transpose(0, 1)
    return pooled / tensors[4].sum(1, keepdim=True)
