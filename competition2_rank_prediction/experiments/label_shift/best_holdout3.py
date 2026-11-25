"""Evaluates no-label prior adaptation on the current C1+C2 cross-fit."""

import pathlib

import numpy as np

from competition2_rank_prediction.experiments.label_shift import run


def main() -> None:
    """Adapts current-best predictions and saves a separate report."""
    root = pathlib.Path.cwd()
    source = root / "production/decision_alignment_best_holdout3_20261109"
    output = root / "production/label_shift_best_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    with np.load(source / "predictions.npz") as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    report, predictions = run.summarize(arrays)
    run.save_result(output, arrays, predictions, report)


if __name__ == "__main__":
    main()
