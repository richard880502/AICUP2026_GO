# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Audits and scores the completed matched HumanSL adaptation pilot.

Run with PYTHONPATH=competition2_rank_prediction MPLBACKEND=Agg python -m
plus this path. Missing runs block scoring rather than silently dropping seeds.
"""

import hashlib
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_evaluate,
)


def check_checkpoint(directory: pathlib.Path, tag: str, metadata: dict) -> list:
    """Checks full budget, player isolation and inner-only checkpoint selection."""
    history = json.loads((directory / f"{tag}_history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, 41)):
        raise ValueError(f"Incomplete forty-epoch budget: {tag}")
    selected = [row for row in history if "inner_nll" in row]
    if [row["epoch"] for row in selected] != list(range(5, 41, 5)):
        raise ValueError("Unexpected checkpoint selection schedule")
    if not np.isfinite([row["inner_nll"] for row in selected]).all():
        raise ValueError("Nonfinite inner checkpoint metric")
    checkpoint = torch.load(
        directory / f"{tag}.pt", weights_only=True, map_location="cpu"
    )
    epoch = min(selected, key=lambda row: row["inner_nll"])["epoch"]
    if checkpoint["epoch"] != epoch:
        raise ValueError("Checkpoint is not earliest minimum inner-NLL epoch")
    optimizer = set(checkpoint["optimizer_players"])
    inner = set(checkpoint["inner_players"])
    if len(optimizer) != 400 or len(inner) != 100 or optimizer & inner:
        raise ValueError("Incorrect optimizer/inner player partitions")
    if (optimizer | inner) & set(metadata["player"]):
        raise ValueError("Evaluation players used in neural fitting/selection")
    if bool(checkpoint["backbone_partial"]) != checkpoint["adapt"]:
        raise ValueError("Incorrect frozen/adapted checkpoint payload")
    return history


def load(directory: pathlib.Path) -> tuple[dict, dict]:
    """Requires aligned normalized predictions from all six completed fits."""
    predictions: dict = {2: [], 8: []}
    metadata: dict = {}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in range(3):
        for adapt, arm in ((False, 2), (True, 8)):
            tag = f"adapt{int(adapt)}_seed{seed}"
            with np.load(directory / f"{tag}_validation.npz") as payload:
                for name in ("game_key", "player", "y"):
                    values = np.asarray(payload[name])
                    if name in metadata and not np.array_equal(
                        metadata[name], values
                    ):
                        raise ValueError(
                            f"Prediction alignment differs: {name}"
                        )
                    metadata[name] = values
                probability = np.asarray(payload["probabilities"])
                if (
                    probability.shape != (8000, 10)
                    or not np.isfinite(probability).all()
                    or (probability < 0).any()
                    or not np.allclose(probability.sum(1), 1, atol=1e-5)
                ):
                    raise ValueError("Invalid validation probabilities")
                predictions[arm].append(probability)
            history = check_checkpoint(directory, tag, metadata)
            axes[0].plot([row["train_nll"] for row in history], label=tag)
            axes[1].plot(
                [row["epoch"] for row in history if "inner_nll" in row],
                [row["inner_nll"] for row in history if "inner_nll" in row],
                label=tag,
            )
    axes[0].set_title("Training per-game NLL")
    axes[1].set_title("Inner NLL: checkpoint selection only")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(figure)
    return predictions, metadata


def main() -> None:
    """Checks source provenance before computing the locked question comparison."""
    root = pathlib.Path.cwd()
    directory = root / "production/human_finetune_20261109"
    report = directory / "report.json"
    if report.exists():
        raise FileExistsError(f"Refusing existing adaptation report: {report}")
    hashes = json.loads((directory / "input_hashes.json").read_text())
    for path, expected in hashes.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed training source/input: {path}")
    predictions, metadata = load(directory)
    result = coverage_evaluate.summarize(predictions, metadata)
    result["scope"] = "exploratory HumanSL adaptation own branch, not Public"
    result["arms"] = {"2": "frozen", "8": "last_two_blocks_adapted"}
    result["evaluation_source_sha256"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
