"""Screens the fixed eight-game pilot as an extra original-pipeline member.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
This post-replacement exploratory screen does not select a submission.
"""

import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_confirm,
)
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


def main() -> None:
    """Fits four-member fusion on fitting players only, keeping original d96."""
    root = pathlib.Path.cwd()
    directory = root / "production/game_coverage_pilot_20261109/training"
    report = directory / "additive_report.json"
    if report.exists():
        raise FileExistsError(f"Refusing existing additive result: {report}")
    torch.set_num_threads(4)
    data = coverage_confirm.prepare(root, directory)
    data["coverage"] = data.pop("B")
    del data["A"]
    result, arrays = calibration.cross_validate(
        data, ("d96", "resnet8c", "opp_scratch", "coverage")
    )
    if abs(result["utility"]["baseline"] - 0.7106057000201695) > 1e-12:
        raise ValueError("Original pipeline replay failed")
    delta = statistics.player_scores(
        arrays["candidate"], arrays["y"]
    ) - statistics.player_scores(arrays["baseline"], arrays["y"])
    interval = statistics.bootstrap_players(
        np.asarray(
            [
                delta[arrays["player"] == player].mean()
                for player in np.unique(arrays["player"])
            ]
        ),
        confidence=0.975,
    )
    record = {
        "scope": "post-replacement exploratory additive screen; not Public",
        "result": result,
        "paired_player_97_5ci": interval,
        "official_submission_created": False,
    }
    np.savez_compressed(
        directory / "additive_predictions.npz",
        baseline=arrays["baseline"],
        candidate=arrays["candidate"],
        y=arrays["y"],
        player=arrays["player"],
    )
    report.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
