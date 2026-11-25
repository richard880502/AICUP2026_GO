# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Writes immutable prediction and report files for decision-alignment runs."""

import json
import pathlib

import numpy as np


def save(
    output: pathlib.Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, object],
) -> None:
    """Stores prediction and report files, then prints the report."""
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz", allow_pickle=False, **arrays
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
