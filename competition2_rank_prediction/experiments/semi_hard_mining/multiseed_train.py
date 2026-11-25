"""Trains the fixed three-seed, larger-budget A/B replication.

Run multiseed_evaluate only after all six checkpoints have finished.
"""

import argparse
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)
from competition2_rank_prediction.experiments.semi_hard_mining import train

SEEDS = (20261118, 20261119, 20261120)
EPOCHS = 60
STEPS = 64
OUTPUT_NAME = "semi_hard_mining_multiseed_20261108"


def load_context(arguments: argparse.Namespace) -> train.TrainingContext:
    """Loads the original deterministic data and verifies its split identity."""
    data = player_sampler.load_data(arguments.research)
    first_output = arguments.output.parent / "semi_hard_mining_20261108"
    sources = [first_output / "splits.npz"]
    if arguments.output.exists():
        sources.append(arguments.output / "splits.npz")
    for source in sources:
        with np.load(source) as saved:
            for name, players in data.splits.items():
                if not np.array_equal(saved[name], players):
                    raise ValueError(
                        f"Saved split mismatch in {source}: {name}"
                    )
    return train.TrainingContext(
        data=data,
        tokens=torch.from_numpy(data.tokens).to(arguments.device),
        masks=torch.from_numpy(data.masks).to(arguments.device),
        labels=torch.from_numpy(data.labels).to(arguments.device),
        output=arguments.output,
    )


def main() -> None:
    """Writes the fixed recipe and fits all A/B replications."""
    arguments = common.parse_arguments(__doc__ or "", OUTPUT_NAME, True)
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    torch.set_num_threads(4)
    context = load_context(arguments)
    arguments.output.mkdir(parents=True)
    source = pathlib.Path(__file__).parent
    protocol = {
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "steps": STEPS,
        "primary": "mean paired per-player utility delta over all three seeds",
        "data_audit": context.data.provenance.audit,
        "preregistration_sha256": hashlib.sha256(
            (source / "MULTISEED_PREREGISTRATION.md").read_bytes(),
        ).hexdigest(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.glob("*.py")
        },
    }
    (arguments.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        arguments.output / "splits.npz",
        train=context.data.splits["train"],
        inner=context.data.splits["inner"],
        evaluation=context.data.splits["evaluation"],
    )
    for seed in SEEDS:
        directory = arguments.output / f"seed_{seed}"
        directory.mkdir()
        seeded_context = dataclasses.replace(context, output=directory)
        recipe = train.TrainingRecipe(seed=seed, epochs=EPOCHS, steps=STEPS)
        for arm in ("A", "B"):
            print(f"seed={seed} arm={arm} start", flush=True)
            train.train_arm(seeded_context, arm, recipe)


if __name__ == "__main__":
    main()
