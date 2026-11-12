"""Shares query-count score curves for aggregate scientific figures."""

import matplotlib.axes

from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)


def query_curve(
    axis: matplotlib.axes.Axes,
    by_count: dict,
    label: str,
) -> None:
    """Plots the four registered query counts with explicit metric labels."""
    axis.plot(
        evaluation.COUNTS,
        [by_count[str(count)]["score"] for count in evaluation.COUNTS],
        marker="o",
        label=label,
    )
    axis.set_xticks(evaluation.COUNTS)
    axis.set_xlabel("Query games")
    axis.set_ylabel("Exponential Top-5 score")
