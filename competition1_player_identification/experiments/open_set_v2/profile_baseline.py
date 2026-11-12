"""Evaluates frozen Fisher-weighted rich profiles on unseen v2 players.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.profile_baseline

The representation and Fisher weights are fitted on training players only.
This avoids claiming player separation for legacy encoders whose source-player
aliases cannot be reconciled with the current training identities.
"""

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import opening


def main() -> None:
    """Saves a training-normalized non-neural rich-profile retrieval baseline."""
    root = dataset.ROOT
    selected = np.load(root / "engine_inputs/global_indices.npy")
    raw = dataset.load_arrays(root / "raw_cache", ("players",))
    profiles = np.load(root / "engine_profiles.npy", mmap_mode="r")
    with np.load(root / "clean_splits.npz") as splits:
        training = np.flatnonzero(np.isin(selected, splits["train"]))
        validation = np.searchsorted(selected, splits["validation"])
    mean = profiles[training].mean(0)
    scale = profiles[training].std(0).clip(min=1e-4)
    normalized = np.clip(
        (profiles[training] - mean) / scale,
        -10,
        10,
    )
    weights = opening.fisher_weights(
        normalized, raw["players"][selected[training]]
    )
    with (root / "fisher_parameters.npz").open("wb") as output:
        np.savez_compressed(
            output,
            mean=mean,
            scale=scale,
            weights=weights,
        )
    values = (
        np.clip(
            (profiles[validation] - mean) / scale,
            -10,
            10,
        )
        * weights
    )
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    scores = evaluation.similarity_scores(values.astype(np.float32), "cuda:0")
    evaluation.save_scores(root / "fisher_validation_scores.npz", scores)
    artifacts.write_report(
        root / "fisher_validation.json",
        {name: evaluation.summarize(matrix) for name, matrix in scores.items()},
        emit=True,
    )


if __name__ == "__main__":
    main()
