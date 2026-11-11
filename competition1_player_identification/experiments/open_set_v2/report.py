"""Produces validation-only phase-one figures and color subgroup diagnostics.

Example:
    MPLBACKEND=Agg python -m \
      competition1_player_identification.experiments.open_set_v2.report

Figures contain aggregate metrics only. Source SGFs and player IDs stay local.
"""

import json
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)

DESTINATION = pathlib.Path(__file__).parent / "figures"


def phase_one_scores(root: pathlib.Path) -> dict:
    """Loads the explicitly defined early ablations, excluding later engine
    fits.
    """
    sources = {}
    with np.load(root / "opening_validation_scores.npz") as arrays:
        sources["Opening statistics"] = np.asarray(arrays["centroid"])
    for name, directory in (
        ("Opening encoder", "encoder_opening"),
        ("Move encoder", "encoder_sequence"),
    ):
        selection = json.loads(
            (root / directory / "completed.json").read_text()
        )
        with np.load(root / directory / "validation_scores.npz") as arrays:
            sources[name] = np.asarray(arrays[selection["aggregation"]])
    sources["Opening prefixes"] = np.load(root / "prefix_validation_scores.npy")
    sources["Rank prior only"] = np.load(
        root / "rank_prior/validation_scores.npy"
    )
    snapshot = root / "phase1_fusion_scores.npy"
    if not snapshot.exists():
        fusion = json.loads((root / "fusion_validation.json").read_text())
        if any(name.startswith("engine") for name in fusion["source_manifest"]):
            raise ValueError("Phase-one snapshot must precede engine fusion")
        shutil.copy2(root / "fusion_validation_scores.npy", snapshot)
        shutil.copy2(
            root / "fusion_validation.json",
            root / "phase1_fusion_selection.json",
        )
    sources["Selected fusion"] = np.load(snapshot)
    return sources


def draw_figure(root: pathlib.Path, sources: dict, reports: dict) -> None:
    """Draws aggregate validation results and actual training curves."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )
    labels = list(sources)
    values = [reports[name]["score"] for name in labels]
    axes[0].barh(
        labels,
        values,
        color=["#7986cb"] * 5 + ["#26a69a"],
    )
    axes[0].set_xlabel("Exponential Top-5 score")
    axes[0].set_title("600-way validation; 10 references")
    for index, value in enumerate(values):
        axes[0].text(
            value + 0.001,
            index,
            f"{value:.4f}",
            va="center",
        )
    axes[0].set_xlim(0, max(values) * 1.4)
    for name in (
        "Opening prefixes",
        "Opening encoder",
        "Move encoder",
        "Selected fusion",
    ):
        by_count = reports[name]["by_query_count"]
        axes[1].plot(
            evaluation.COUNTS,
            [by_count[str(count)]["score"] for count in evaluation.COUNTS],
            marker="o",
            label=name,
        )
    axes[1].set_xticks(evaluation.COUNTS)
    axes[1].set_xlabel("Query games")
    axes[1].set_ylabel("Exponential Top-5 score")
    axes[1].set_title("Nested queries; same 600 players")
    axes[1].legend(fontsize=8)
    for name, directory in (
        ("Opening", "encoder_opening"),
        ("Moves", "encoder_sequence"),
    ):
        history = json.loads((root / directory / "history.json").read_text())
        axes[2].plot(
            [row["epoch"] for row in history],
            [row["loss"] for row in history],
            marker="o",
            label=name,
        )
    axes[2].set_xlabel("Epoch (256 player episodes)")
    axes[2].set_ylabel("ArcFace + episodic + SupCon loss")
    axes[2].set_title("Training players only")
    axes[2].legend()
    figure.suptitle(
        "Phase 1: exploratory validation, before engine profiles; "
        "final holdout unopened",
        fontsize=12,
    )
    figure.tight_layout()
    DESTINATION.mkdir(exist_ok=True)
    figure.savefig(DESTINATION / "phase1_summary.svg")
    figure.savefig(root / "phase1_summary.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Writes portable aggregate evidence without accessing final outcomes."""
    root = dataset.ROOT
    sources = phase_one_scores(root)
    reports = {
        name: evaluation.summarize(scores) for name, scores in sources.items()
    }
    arrays = dataset.load_arrays(root / "raw_cache", ("colors",))
    with np.load(root / "clean_splits.npz") as splits:
        colors = arrays["colors"][np.asarray(splits["validation"])[:, 10:]]
    artifacts.write_report(
        root / "phase1_color_subgroups.json",
        {
            name: evaluation.color_subgroups(scores, colors)
            for name, scores in sources.items()
        },
    )
    draw_figure(
        root,
        sources,
        reports,
    )
    artifacts.write_report(
        root / "phase1_aggregate.json",
        reports,
        emit=True,
    )


if __name__ == "__main__":
    main()
