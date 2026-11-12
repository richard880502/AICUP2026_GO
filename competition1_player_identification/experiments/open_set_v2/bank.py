"""Encodes frozen experiment checkpoints without updating the stage-one model.

Use encode_checkpoint(path) after the selected checkpoint has been saved.
Banks are keyed by the full checkpoint digest to avoid stale embedding reuse.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import models
from competition1_player_identification.experiments.open_set_v2 import train
from competition1_player_identification.experiments.open_set_v2 import training


def frozen_runtime(
    checkpoint: dict,
    data: training.GameData,
    device: str,
) -> training.Runtime:
    """Creates a read-only inference context with no optimizer updates."""
    specification = models.Specification(**checkpoint["specification"])
    model = models.Encoder(specification).to(device)
    model.load_state_dict(checkpoint["state"])
    return training.Runtime(
        model,
        torch.optim.SGD(model.parameters(), lr=0),
        data,
        device,
        np.random.default_rng(20261111),
        specification.kind,
    )


def encode_checkpoint(path: pathlib.Path, device: str = "cuda:0") -> np.ndarray:
    """Returns unit game embeddings from a fixed, digest-verified checkpoint."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    precision = "bf16" if device.startswith("cuda") else "fp32"
    destination = path.parent / f"bank_{digest[:12]}_{precision}.npy"
    if destination.exists():
        return np.load(destination, mmap_mode="r")
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    specification = models.Specification(**checkpoint["specification"])
    data = train.load_data(
        argparse.Namespace(
            root=dataset.ROOT,
            kind=specification.kind,
            length=specification.length,
        )
    )
    if not np.array_equal(data.mean, checkpoint["mean"]) or not np.array_equal(
        data.scale,
        checkpoint["scale"],
    ):
        raise ValueError(
            "Training normalization no longer matches frozen stage one"
        )
    runtime = frozen_runtime(checkpoint, data, device)
    embeddings = training.encode(runtime, np.arange(len(data.values)))
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite frozen game embeddings")
    temporary = destination.with_suffix(".tmp.npy")
    np.save(temporary, embeddings)
    temporary.replace(destination)
    artifacts.write_report(
        path.parent / "bank.json",
        {
            "checkpoint_sha256": digest,
            "bank": destination.name,
            "games": len(embeddings),
            "dimension": specification.dimension,
            "frozen": True,
            "precision": precision,
        },
    )
    return embeddings


def main() -> None:
    """Encodes an explicit checkpoint for matching ablations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    values = encode_checkpoint(arguments.checkpoint)
    print(json.dumps({"games": len(values), "dimension": values.shape[-1]}))


if __name__ == "__main__":
    main()
