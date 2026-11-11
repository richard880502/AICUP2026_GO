"""Prepares and verifies the training-only, mixed-color v2 experiment data.

Run from the repository root:
    python -m competition1_player_identification.experiments.open_set_v2.prepare

Unlike the superseded pointer-only manifest, this entry point reads actual
CSV content, rejects changed sources, and verifies existing locked splits.
"""

import argparse
import subprocess
import sys

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    export_engine,
)

PACKAGE = "competition1_player_identification.experiments.open_set_v2"


def main() -> None:
    """Runs idempotent cache preparation without overwriting cohort choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts.add_workers(parser)
    arguments = parser.parse_args()
    for command in (
        ["cache", "--workers", str(arguments.workers)],
        ["source_audit"],
    ):
        subprocess.run(
            [sys.executable, "-m", f"{PACKAGE}.{command[0]}", *command[1:]],
            check=True,
        )
    root = dataset.ROOT
    if (root / "clean_splits.npz").exists():
        dataset.verify_locked_splits(root)
    else:
        dataset.build_splits(root, 20261111)
    if (root / "engine_inputs/global_indices.npy").exists():
        expected = export_engine.selected_games(root, 16)
        actual = np.load(root / "engine_inputs/global_indices.npy")
        if not np.array_equal(expected, actual):
            raise ValueError(
                "Existing extraction inputs do not match the locked split"
            )
    else:
        export_engine.export(
            root,
            dataset.ROOT.parent.parent / "data/training",
            16,
        )
    print("Training-only preparation and immutable split verification passed")


if __name__ == "__main__":
    main()
