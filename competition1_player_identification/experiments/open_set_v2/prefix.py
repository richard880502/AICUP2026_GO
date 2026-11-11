"""Compares exact, symmetry-invariant opening prefixes on v2 validation.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.prefix

Prefix counts are conditioned on target color and weighted by reference-set
frequency. They are an opening-match feature, not an identity classifier.
"""

import hashlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)

PREFIX_LENGTHS = (
    4,
    8,
    16,
    32,
    64,
)


def symmetry_tables() -> np.ndarray:
    """Maps combined color/position tokens under eight board symmetries."""
    tables = []
    for transpose in (False, True):
        for reverse_column in (False, True):
            for reverse_row in (False, True):
                tokens = np.arange(725)
                position = (tokens - 1).clip(min=0) % 362
                column, row = position // 19, position % 19
                if transpose:
                    column, row = row, column
                if reverse_column:
                    column = 18 - column
                if reverse_row:
                    row = 18 - row
                mapped = column * 19 + row + 1
                mapped += ((tokens - 1).clip(min=0) // 362) * 362
                tables.append(
                    np.where(
                        (tokens > 0) & (position < 361),
                        mapped,
                        tokens,
                    )
                )
    return np.asarray(tables, dtype=np.uint16)


def fingerprints(tokens: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Hashes prefix length, target color and canonical move sequence."""
    tables = symmetry_tables()
    result = np.zeros((len(tokens), len(PREFIX_LENGTHS)), np.uint64)
    for index, sequence in enumerate(tokens):
        for column, length in enumerate(PREFIX_LENGTHS):
            if (sequence[:length] > 0).sum() < length:
                continue
            variants = tables[:, sequence[:length]]
            canonical = min(variant.tobytes() for variant in variants)
            prefix = bytes((int(colors[index]), length)) + canonical
            result[index, column] = int.from_bytes(
                hashlib.blake2b(prefix, digest_size=8).digest(),
                "little",
            )
    return result


def score_prefixes(hashes: np.ndarray) -> np.ndarray:
    """Aggregates informative reference matches over nested query subsets."""
    references = hashes[:, :10].reshape(-1, len(PREFIX_LENGTHS))
    queries = hashes[:, 10:].reshape(-1, len(PREFIX_LENGTHS))
    pairs = np.zeros((len(queries), len(references)), np.float32)
    for column in range(len(PREFIX_LENGTHS)):
        _, inverse, counts = np.unique(
            references[:, column],
            return_inverse=True,
            return_counts=True,
        )
        weights = np.log((len(references) + 1) / (counts[inverse] + 1))
        matches = queries[:, column, None] == references[None, :, column]
        matches &= queries[:, column, None] != 0
        pairs += matches * weights[None] * (column + 1)
    per_game = pairs.reshape(
        len(hashes),
        5,
        len(hashes),
        10,
    ).max(3)
    return np.stack(
        [per_game[:, :count].mean(1) for count in evaluation.COUNTS]
    )


def main() -> None:
    """Writes only validation scores; final cohort remains unopened."""
    root = dataset.ROOT
    arrays = dataset.load_arrays(root / "raw_cache", ("tokens", "colors"))
    with np.load(root / "clean_splits.npz") as splits:
        validation = np.asarray(splits["validation"])
    indices = validation.ravel()
    hashes = fingerprints(arrays["tokens"][indices], arrays["colors"][indices])
    scores = score_prefixes(
        np.reshape(
            hashes,
            (
                600,
                15,
                -1,
            ),
        )
    )
    np.save(root / "prefix_validation_scores.npy", scores)
    report = evaluation.summarize(scores)
    artifacts.write_report(
        root / "prefix_validation.json",
        report,
        emit=True,
    )


if __name__ == "__main__":
    main()
