"""Writes portable experiment reports without including licensed game content.

Use write_report(path, values, emit=True) for a local JSON report plus stdout.
"""

import argparse
import collections.abc
import hashlib
import json
import pathlib

import numpy as np


def file_digest(path: pathlib.Path) -> str:
    """Hashes large local artifacts without loading them into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def root_parser(
    description: str | None, root: pathlib.Path
) -> argparse.ArgumentParser:
    """Creates consistent experiment-root arguments for command-line tools."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=root,
    )
    return parser


def add_seed(parser: argparse.ArgumentParser) -> None:
    """Adds the registered default seed to an experiment command."""
    parser.add_argument(
        "--seed",
        type=int,
        default=20261111,
    )


def add_workers(parser: argparse.ArgumentParser, default: int = 10) -> None:
    """Adds consistent bounded process-pool settings to an experiment command."""
    parser.add_argument(
        "--workers",
        type=int,
        default=default,
    )


def batch_indices(indices: np.ndarray) -> collections.abc.Iterator[np.ndarray]:
    """Yields bounded row-index batches for frozen encoder inference.

    Yields:
        Up to 128 original row positions per batch.
    """
    for start in range(
        0,
        len(indices),
        128,
    ):
        yield indices[start : start + 128]


def write_report(
    path: pathlib.Path,
    values: dict,
    emit: bool = False,
) -> None:
    """Writes JSON once and optionally emits the same text for a run log."""
    serialized = json.dumps(values, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    if emit:
        print(
            serialized,
            end="",
            flush=True,
        )
