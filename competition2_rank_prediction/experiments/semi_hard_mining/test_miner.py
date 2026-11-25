"""Tests training-only mining, strict margins, fallback and mode restoration.

Run with unittest before starting the direct C experiment.
"""

import unittest

import numpy as np
import torch

from competition2_rank_prediction.experiments.semi_hard_mining import pair_miner
from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)


def synthetic_data() -> player_sampler.PlayerData:
    """Creates eight training and four held-out dense players."""
    players = np.repeat(np.asarray([f"p{index}" for index in range(12)]), 20)
    labels = np.repeat(np.asarray([4] * 6 + [5] * 6), 20)
    rows = {
        str(player): np.flatnonzero(players == player)
        for player in np.unique(players)
    }
    return player_sampler.PlayerData(
        tokens=np.zeros((240, 8, 1), dtype=np.float32),
        masks=np.ones((240, 8), dtype=bool),
        labels=labels,
        rows=rows,
        splits={
            "train": np.asarray(
                [f"p{index}" for index in (0, 1, 2, 3, 6, 7, 8, 9)]
            )
        },
        questions={},
        provenance=player_sampler.Provenance({}, {}),
    )


class RecordingEncoder(torch.nn.Module):
    """Rejects forbidden NaN games and records the rows consumed by mining.

    Attributes:
        score: A differentiable scalar projection.
        rows_seen: Total number of training games encoded.
    """

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Linear(1, 1, bias=False)
        self.rows_seen = 0

    def encode(self, tokens: torch.Tensor, _mask: torch.Tensor) -> torch.Tensor:
        """Rejects held-out sentinel inputs and returns their first channel."""
        if not torch.isfinite(tokens).all():
            raise ValueError("Miner accessed held-out sentinel games")
        self.rows_seen += len(tokens)
        return tokens[:, 0, :1]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns scalar strengths using the recorded encoding path."""
        return self.score(self.encode(tokens, mask))


class MiningTest(unittest.TestCase):
    """Verifies the exact eligible population and semi-hard window."""

    def test_window_uniqueness_and_exclusion(self) -> None:
        """Mined pairs have positive bounded margins and unique training players."""
        data = synthetic_data()
        miner = pair_miner.SemiHardMiner(data)
        values = {
            str(player): (
                0.0 if data.labels[data.rows[str(player)][0]] == 4 else 0.5
            )
            for player in data.splits["train"]
        }
        miner.update_strengths(values)
        generator = np.random.default_rng(7)
        for _ in range(30):
            used = []
            for low, high in miner.sample_pairs(generator):
                self.assertTrue(np.all(data.labels[low] == 4))
                self.assertTrue(np.all(data.labels[high] == 5))
                for bag in (low, high):
                    player = next(
                        player
                        for player, rows in data.rows.items()
                        if bag[0] in rows
                    )
                    self.assertIn(player, data.splits["train"])
                    self.assertTrue(5 <= len(bag) <= 20)
                    self.assertEqual(len(np.unique(bag)), len(bag))
                    used.append(player)
            self.assertEqual(len(set(used)), 8)
        self.assertGreater(miner.counts["mined"], 0)
        self.assertEqual(miner.counts["mined_margin_min"], 0.5)
        self.assertEqual(miner.counts["mined_margin_max"], 0.5)
        with self.assertRaises(ValueError):
            miner.update_strengths({**values, "p4": 0})

    def test_invalid_margins_fall_back(self) -> None:
        """Zero, inverted and overly large margins never enter mined pairs."""
        data = synthetic_data()
        for margin in (-1.0, 0.0, 2.0):
            with self.subTest(margin=margin):
                miner = pair_miner.SemiHardMiner(data)
                miner.update_strengths(
                    {
                        str(player): (
                            0.0
                            if data.labels[data.rows[str(player)][0]] == 4
                            else margin
                        )
                        for player in data.splits["train"]
                    }
                )
                generator = np.random.default_rng(8)
                for _ in range(20):
                    miner.sample_pairs(generator)
                self.assertEqual(miner.counts["mined"], 0)
                self.assertEqual(miner.counts["random"], 80)
                self.assertGreater(miner.counts["fallback"], 0)

    def test_refresh_is_training_only_and_detached(self) -> None:
        """Refresh ignores held-out games and restores training mode."""
        data = synthetic_data()
        tokens = torch.full((240, 8, 1), float("nan"))
        for player in data.splits["train"]:
            tokens[data.rows[str(player)]] = 1
        model = RecordingEncoder()
        model.train()
        miner = pair_miner.SemiHardMiner(data)
        miner.refresh(model, tokens, torch.ones((240, 8), dtype=torch.bool))
        self.assertTrue(model.training)
        self.assertEqual(model.rows_seen, 160)
        self.assertIsNone(model.score.weight.grad)
        self.assertEqual(set(miner.strengths), set(data.splits["train"]))


if __name__ == "__main__":
    unittest.main()
