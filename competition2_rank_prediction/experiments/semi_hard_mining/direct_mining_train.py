"""Trains three C arms using the fixed direct-mining preregistration.

The matched A/B controls are read from the completed multi-seed experiment.
"""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import (
    multiseed_train,
)
from competition2_rank_prediction.experiments.semi_hard_mining import train

OUTPUT_NAME = "semi_hard_mining_direct_20261108"


def controls_directory(output: pathlib.Path) -> pathlib.Path:
    """Returns the frozen matched-budget A/B experiment directory."""
    return output.parent / multiseed_train.OUTPUT_NAME


def verify_controls(context: train.TrainingContext) -> dict[str, str]:
    """Checks original splits and fingerprints every matched control."""
    directory = controls_directory(context.output)
    with np.load(directory / "splits.npz") as saved:
        for name, players in context.data.splits.items():
            if not np.array_equal(saved[name], players):
                raise ValueError("Direct mining and A/B players differ")
    fingerprints = {}
    for seed in multiseed_train.SEEDS:
        for arm in ("A", "B"):
            path = directory / f"seed_{seed}/{arm}.pt"
            fingerprints[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return fingerprints


def main() -> None:
    """Fits only C while preserving the preceding study's gate and results."""
    arguments = common.parse_arguments(__doc__ or "", OUTPUT_NAME, True)
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    torch.set_num_threads(4)
    context = multiseed_train.load_context(arguments)
    controls = verify_controls(context)
    arguments.output.mkdir(parents=True)
    source = pathlib.Path(__file__).parent
    (arguments.output / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": list(multiseed_train.SEEDS),
                "epochs": multiseed_train.EPOCHS,
                "steps": multiseed_train.STEPS,
                "reason": "user requested exploratory mining after prior gate failure",
                "control_checkpoints_sha256": controls,
                "preregistration_sha256": hashlib.sha256(
                    (source / "DIRECT_MINING_PREREGISTRATION.md").read_bytes(),
                ).hexdigest(),
                "source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source.glob("*.py")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        context.output / "splits.npz",
        train=context.data.splits["train"],
        inner=context.data.splits["inner"],
        evaluation=context.data.splits["evaluation"],
    )
    for seed in multiseed_train.SEEDS:
        directory = context.output / f"seed_{seed}"
        directory.mkdir()
        recipe = train.TrainingRecipe(
            seed=seed,
            epochs=multiseed_train.EPOCHS,
            steps=multiseed_train.STEPS,
        )
        print(f"seed={seed} arm=C start", flush=True)
        train.train_arm(
            dataclasses.replace(context, output=directory), "C", recipe
        )


if __name__ == "__main__":
    main()
