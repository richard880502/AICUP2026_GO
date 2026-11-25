"""Tests the fixed adapted HumanSL branch as an addition to original C1+C2.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Requires the preregistered own-branch screen; never replaces original d96.
"""

import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import evaluate


def prepare(root: pathlib.Path, directory: pathlib.Path) -> dict:
    """Aligns the adapted ensemble to neural-unseen original validation games."""
    with np.load(root / "production/rank_joint_20261108/prepared.npz") as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in (
                "y",
                "player",
                "game_key",
                "final",
                *evaluate.MEMBERS[:3],
            )
        }
    predictions = []
    for seed in range(3):
        checkpoint = torch.load(
            directory / f"adapt1_seed{seed}.pt",
            weights_only=True,
            map_location="cpu",
        )
        excluded = set(checkpoint["optimizer_players"]) | set(
            checkpoint["inner_players"]
        )
        if excluded & set(data["player"]):
            raise ValueError("Adapted branch saw evaluation players")
        with np.load(directory / f"adapt1_seed{seed}_validation.npz") as cache:
            lookup = {
                str(key): row for row, key in enumerate(cache["game_key"])
            }
            rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
            for name in ("player", "y"):
                if not np.array_equal(
                    np.asarray(cache[name])[rows], data[name]
                ):
                    raise ValueError(f"Mismatched adapted metadata: {name}")
            predictions.append(np.asarray(cache["probabilities"])[rows])
    # Shared evaluator's historical member name; this contains adapted outputs.
    data["human_trunk"] = np.mean(predictions, axis=0)
    return data


def main() -> None:
    """Runs one additional-member calibration screen with fitting-only weights."""
    root = pathlib.Path.cwd()
    directory = root / "production/human_finetune_20261109"
    report = directory / "confirmation_report.json"
    if report.exists():
        raise FileExistsError(
            f"Refusing existing adapted confirmation: {report}"
        )
    if not json.loads((directory / "report.json").read_text())["screen_pass"]:
        raise RuntimeError("Matched adaptation pilot failed advancement")
    torch.set_num_threads(4)
    arrays, records = evaluate.cross_validate(prepare(root, directory))
    result = evaluate.summarize(arrays, records)
    result["candidate"] = "original C1+C2 plus adapted last-two-block HumanSL"
    if abs(result["utility"]["baseline"] - 0.7106057000201695) > 1e-12:
        raise ValueError("Original C1+C2 replay failed")
    np.savez_compressed(
        directory / "confirmation_predictions.npz",
        baseline=arrays["baseline"],
        candidate=arrays["candidate"],
        y=arrays["y"],
        player=arrays["player"],
        count=arrays["count"],
    )
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
