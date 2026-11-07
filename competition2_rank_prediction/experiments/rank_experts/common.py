"""Provides shared CLI paths and player subgroup masks for expert analyses."""

import argparse
import pathlib

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import run


def parse_arguments(
    description: str,
    output_name: str,
    include_device: bool = False,
) -> argparse.Namespace:
    """Parses immutable pilot input and analysis output locations."""
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--research", type=pathlib.Path,
        default=repository / "runtime/competition2_research",
    )
    parser.add_argument(
        "--pilot", type=pathlib.Path,
        default=repository / "production/rank_experts_20261106/predictions.npz",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=repository / "production" / output_name,
    )
    if include_device:
        parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def subgroup_masks(data: run.PilotData) -> dict[str, np.ndarray]:
    """Returns fixed true-rank, game-count and upstream-half audit groups."""
    groups = {}
    for index, name in enumerate(run.LEVELS):
        groups[name] = data.labels == index
    for low, high in ((5, 9), (10, 14), (15, 20)):
        groups[f"games_{low}_{high}"] = (data.counts >= low) & (data.counts <= high)
    for half in (0, 1):
        groups[f"half_{half}"] = data.halves == half
    return groups
