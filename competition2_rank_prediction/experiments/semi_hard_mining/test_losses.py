"""Checks ranking-loss gradients and ordinal player sampling invariants.

Run with unittest before starting GPU training.
"""

import unittest

import numpy as np
import torch

import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.semi_hard_mining import losses
from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)


class PairLossTest(unittest.TestCase):
    """Verifies that auxiliary supervision can update the encoder."""

    def test_monotonic_and_zero_identity(self) -> None:
        """Larger correct margins reduce loss; zero weight is exact identity."""
        low = torch.zeros(4)
        self.assertLess(
            losses.pairwise_loss(low, low + 1), losses.pairwise_loss(low, low)
        )
        nll = torch.tensor(2.0, requires_grad=True)
        self.assertIs(
            losses.combined_loss(nll, torch.tensor(float("nan")), 0), nll
        )
        with self.assertRaises(ValueError):
            losses.pairwise_loss(torch.zeros(2), torch.zeros(3))

    def test_encoder_gradient(self) -> None:
        """The pooled scalar path has nonzero encoder parameter gradients."""
        torch.manual_seed(1)
        model = transformer_rank.RankTransformer(6, 8, 16, 4, 32, 1, 0)
        inputs = torch.randn(10, 8, 6)
        mask = torch.ones(10, 8, dtype=torch.bool)
        strength = model.score(model.encode(inputs, mask)).squeeze(-1)
        losses.pairwise_loss(
            strength[:5].mean()[None], strength[5:].mean()[None]
        ).backward()
        gradient = model.project[0].weight.grad
        self.assertIsNotNone(gradient)
        if gradient is None:
            raise RuntimeError("Missing encoder gradients")
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_sampling(self) -> None:
        """Every pair uses separate adjacent-rank training players."""
        players = np.repeat(
            np.asarray([f"p{index}" for index in range(12)]), 20
        )
        labels = np.repeat(np.asarray([4] * 6 + [5] * 6), 20)
        rows = {
            str(player): np.flatnonzero(players == player)
            for player in np.unique(players)
        }
        data = player_sampler.PlayerData(
            np.empty((240, 0)),
            np.empty((240, 0)),
            labels,
            rows,
            {
                "train": np.asarray(
                    [f"p{index}" for index in (0, 1, 2, 3, 6, 7, 8, 9)]
                )
            },
            {},
            player_sampler.Provenance({}, {}),
        )
        pairs = player_sampler.sample_pairs(data, np.random.default_rng(2))
        used = []
        for low, high in pairs:
            self.assertTrue(np.all(labels[low] == 4))
            self.assertTrue(np.all(labels[high] == 5))
            for bag in (low, high):
                self.assertTrue(5 <= len(bag) <= 20)
                self.assertEqual(len(np.unique(players[bag])), 1)
                self.assertEqual(len(np.unique(bag)), len(bag))
                self.assertIn(players[bag[0]], data.splits["train"])
                used.append(players[bag[0]])
        self.assertEqual(len(set(used)), 8)


if __name__ == "__main__":
    unittest.main()
