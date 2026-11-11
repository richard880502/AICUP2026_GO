"""Builds an engine-free opening-position baseline on the clean v2 validation.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.opening

Fisher reliability and normalization are fitted only on training identities.
This cheap baseline is separate from the HumanSL rich-profile representation.
"""

import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)


def position_histograms(tokens: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Summarizes both players' opening locations at three temporal scales."""
    tokens = tokens[:, :128].astype(np.int32)
    position = (tokens - 1).clip(min=0) % 362
    actor = ((tokens - 1) // 362) != colors[:, None]
    valid = (tokens > 0) & (position < 361)
    region = (position // 19 // 4).clip(max=4) * 5
    region += (position % 19 // 4).clip(max=4)
    results = []
    row_indices = np.broadcast_to(np.arange(len(tokens))[:, None], tokens.shape)
    for phases in (
        1,
        2,
        4,
    ):
        for start in range(
            0,
            128,
            128 // phases,
        ):
            stop = start + 128 // phases
            for other in (False, True):
                selected = valid[:, start:stop] & (
                    actor[:, start:stop] == other
                )
                features = np.zeros((len(tokens), 25), np.float32)
                np.add.at(
                    features,
                    (
                        row_indices[:, start:stop][selected],
                        region[:, start:stop][selected],
                    ),
                    1,
                )
                features /= features.sum(1, keepdims=True).clip(min=1)
                results.append(np.sqrt(features))
    return np.concatenate(results, axis=1)


def fisher_weights(values: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Estimates shrunk between/within-player reliability on training games."""
    _, groups = dataset.player_groups(players)
    means = np.stack([values[group].mean(0) for group in groups])
    total_variance = values.var(0)
    within = np.stack([values[group].var(0) for group in groups]).mean(0)
    reliability = means.var(0) / (within + total_variance * 0.1 + 1e-6)
    return np.sqrt(
        np.clip(
            reliability,
            0.05,
            20,
        )
    ).astype(np.float32)


def make_cache(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Materializes opening statistics for the same rows as engine extraction."""
    selected = np.load(root / "engine_inputs/global_indices.npy")
    destination = root / "opening_profiles.npy"
    if not destination.exists():
        raw = dataset.load_arrays(root / "raw_cache", ("tokens", "colors"))
        chunks = []
        for start in range(
            0,
            len(selected),
            4096,
        ):
            indices = selected[start : start + 4096]
            chunks.append(
                position_histograms(
                    raw["tokens"][indices],
                    raw["colors"][indices],
                )
            )
        np.save(destination, np.concatenate(chunks))
    return np.load(destination), selected


def main() -> None:
    """Saves validation-only baseline scores without reading final outcomes."""
    root = dataset.ROOT
    profiles, selected = make_cache(root)
    raw = dataset.load_arrays(root / "raw_cache", ("players",))
    with np.load(root / "clean_splits.npz") as splits:
        training = np.isin(selected, splits["train"])
        validation = np.asarray(splits["validation"])
    mean = profiles[training].mean(0)
    scale = profiles[training].std(0).clip(min=1e-4)
    scaled = ((profiles - mean) / scale).astype(np.float32)
    weights = fisher_weights(
        scaled[training], raw["players"][selected[training]]
    )
    rows = np.searchsorted(selected, validation)
    scores = evaluation.similarity_scores(scaled[rows] * weights, "cpu")
    evaluation.save_scores(root / "opening_validation_scores.npz", scores)
    report = {
        name: evaluation.summarize(matrix) for name, matrix in scores.items()
    }
    artifacts.write_report(
        root / "opening_validation.json",
        report,
        emit=True,
    )


if __name__ == "__main__":
    main()
