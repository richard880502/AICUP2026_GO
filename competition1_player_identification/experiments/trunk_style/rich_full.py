"""Trains the v4 recipe on the 960-D richer tokens of all 248k games (style_v7).

Reproduces Competition 1 style submission v9, step 3 of 5 (after extract_rich.py
has written the `*_rich` token shards of the training corpus and the pools):
trains the `style_v7` encoder. rich_pipeline.py and Competition 2's
opponent_identity/h3_v7.py import this module for its 960-D layout.

The token array (about 76 GB) exceeds RAM, so it is a disk-backed memmap.

Importing this module makes train.use_all_moves() select the 960-D layout.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.rich_full
"""

import sys

import numpy

from competition1_player_identification.experiments.trunk_style import train

# The research machine's location for the memmapped token array.
STORE = "/mnt/share_data_77/richard/AICUP2026_GO/trunk_style_20261113/rich_tokens.npy"


class DiskBackedNumpy:
    """numpy, except that very large zero arrays live on disk."""

    def __getattr__(self, name):
        return getattr(numpy, name)

    @staticmethod
    def zeros(shape, dtype=float):
        """numpy.zeros, or a memmap at STORE when the array exceeds 30 GB."""
        size = int(numpy.prod(shape)) * numpy.dtype(dtype).itemsize
        if size > 30e9:
            return numpy.lib.format.open_memmap(
                STORE, mode="w+", dtype=dtype, shape=tuple(shape)
            )
        return numpy.zeros(shape, dtype)


def use_rich() -> None:
    """Selects the 960-D richer-token layout (160 moves, `_rich` shards)."""
    train.LAYOUT.length, train.LAYOUT.trunk, train.LAYOUT.engine = 160, 960, 42
    train.LAYOUT.suffix = "_rich"


train.use_all_moves = use_rich


def main() -> None:
    """Trains style_v7 with the v4 options on the richer tokens."""
    # train.np is the numpy module; this duck-typed stand-in only overrides zeros().
    train.np = DiskBackedNumpy()  # type: ignore[assignment]
    sys.argv = [
        "train", "--name", "style_v7", "--all-moves", "--extra",
        "--engine-dropout", "0.3", "--epochs", "30",
    ]
    train.main()


if __name__ == "__main__":
    main()
