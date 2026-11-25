# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Checks coverage encoders in the complete frozen-other-member C1+C2 pipeline.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
This is exploratory historical validation, never Public confirmation.
"""

import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.question_supervision import (
    confirm,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


def prepare(root: pathlib.Path, directory: pathlib.Path) -> dict:
    """Aligns frozen original branches with neural-unseen pilot predictions."""
    with np.load(root / "production/rank_joint_20261108/prepared.npz") as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *confirm.MEMBERS)
        }
    for cap, arm in ((2, "A"), (8, "B")):
        predictions = []
        for seed in range(3):
            checkpoint = torch.load(
                directory / f"cap{cap}_seed{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            excluded = set(checkpoint["optimizer_players"]) | set(
                checkpoint["inner_players"]
            )
            if excluded & set(data["player"]):
                raise ValueError("Confirmation players used in neural fitting")
            with np.load(
                directory / f"cap{cap}_seed{seed}_validation.npz"
            ) as cache:
                lookup = {
                    str(key): row for row, key in enumerate(cache["game_key"])
                }
                rows = np.asarray(
                    [lookup[str(key)] for key in data["game_key"]]
                )
                for stored, original in (("player", "player"), ("y", "y")):
                    if not np.array_equal(
                        np.asarray(cache[stored])[rows], data[original]
                    ):
                        raise ValueError(
                            f"Prediction metadata mismatch: {stored}"
                        )
                predictions.append(np.asarray(cache["probabilities"])[rows])
        data[arm] = np.mean(predictions, axis=0)
    return data


def summarize(results: dict) -> dict:
    """Reports two planned paired-player contrasts using 97.5% intervals."""
    rewards = {
        method: statistics.player_scores(results[method], results["y"])
        for method in ("original", "A", "B")
    }
    intervals = {}
    for reference in ("original", "A"):
        difference = rewards["B"] - rewards[reference]
        clustered = np.asarray(
            [
                difference[results["player"] == player].mean()
                for player in np.unique(results["player"])
            ]
        )
        intervals[f"B_minus_{reference}"] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
    if abs(float(rewards["original"].mean()) - 0.7106057000201695) > 1e-12:
        raise ValueError("Original production calibration CV failed to replay")
    return {
        "scope": "exploratory historical400-player calibration CV, not Public",
        "arms": {"A": "cap2", "B": "cap8"},
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "original_replay_matches": True,
        "official_submission_created": False,
    }


def main() -> None:
    """Runs the same five-fold calibration protocol for all three arms."""
    root = pathlib.Path.cwd()
    directory = root / "production/game_coverage_pilot_20261109/training"
    report = directory / "confirmation_report.json"
    if report.exists():
        raise FileExistsError(f"Refusing existing confirmation: {report}")
    if not json.loads((directory / "report.json").read_text())["screen_pass"]:
        raise RuntimeError("Coverage pilot failed the advancement screen")
    torch.set_num_threads(4)
    results: dict = confirm.cross_validate(prepare(root, directory))
    result = summarize(results)
    np.savez_compressed(directory / "confirmation_predictions.npz", **results)
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
