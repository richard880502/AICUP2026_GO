"""Loads lossless HumanSL inputs and preserves masked game-level gradients.

Use load only after cache COMPLETE; batch reconstructs packed spatial planes.
"""

import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_finetune import probe


def load(directory: pathlib.Path) -> dict:
    """Requires aligned, finite, nonempty per-game input arrays."""
    parts: dict[str, list[np.ndarray]] = {}
    for path in sorted(directory.glob("inputs_*.npz")):
        with np.load(path) as payload:
            for name in payload.files:
                parts.setdefault(name, []).append(np.asarray(payload[name]))
    if not parts:
        raise ValueError(f"No input shards in {directory}")
    result = {name: np.concatenate(values) for name, values in parts.items()}
    games = len(result["game_key"])
    if len(set(result["game_key"])) != games:
        raise ValueError("Duplicate game keys in input cache")
    if result["valid"].shape != (games, 8) or not result["valid"].any(1).all():
        raise ValueError("Missing valid cached positions")
    for name in ("global", "meta"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite cached input: {name}")
    return result


def batch(payload: dict, rows: np.ndarray, device: str) -> list[torch.Tensor]:
    """Reconstructs exactly eight spatial inputs per selected game."""
    spatial = np.unpackbits(payload["spatial_packed"][rows], axis=-1)[..., :19]
    values = [spatial, payload["global"][rows], payload["meta"][rows]]
    tensors = [
        torch.as_tensor(
            value.reshape(-1, *value.shape[2:]),
            device=device,
            dtype=torch.float32,
        )
        for value in values
    ]
    tensors.append(
        torch.as_tensor(payload["points"][rows], device=device).long()
    )
    tensors.append(torch.as_tensor(payload["valid"][rows], device=device))
    return tensors


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pools game decisions while preserving graph and excluding padded slots."""
    if values.shape[:2] != valid.shape or not valid.any(1).all():
        raise ValueError("Invalid game position mask")
    return (values * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)


def representation(model, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Returns variable-batch global/played-point game representations."""
    captured = probe.GradientOutputs()
    model(*tensors[:3], extra_outputs=captured)
    trunk = captured.returned["trunkfinal"]
    points = tensors[3].flatten()
    local = trunk.flatten(2).transpose(1, 2)[
        torch.arange(len(points), device=points.device), points.clamp(max=360)
    ]
    local = torch.where((points < 361)[:, None], local, 0)
    values = torch.cat((trunk.mean((2, 3)), local), dim=1).reshape(-1, 8, 768)
    return masked_mean(values, tensors[4])
