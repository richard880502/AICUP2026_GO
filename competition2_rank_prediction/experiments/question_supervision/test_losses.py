"""Checks question pooling, bag balance and encoder-reaching gradients.

Run: python -m unittest competition2_rank_prediction.experiments.question_supervision.test_losses
"""

import pathlib
import sys
import unittest

import numpy as np
import torch

from competition2_rank_prediction.experiments.question_supervision import losses

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


class QuestionLossTest(unittest.TestCase):
    """Verifies pooling matches inference and never detaches the training graph."""

    def test_pool_matches_production(self) -> None:
        """Checks normalized geometric pooling rather than mean probabilities."""
        generator = np.random.default_rng(12)
        probabilities = generator.uniform(0.05, 1, (8, 10))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        _, actual = losses.question_losses(
            torch.from_numpy(np.log(probabilities)),
            torch.full((8,), 4),
            (8,),
        )
        expected = -np.log(rank_lib.pool(probabilities)[4])
        self.assertAlmostEqual(actual.item(), expected, places=10)

    def test_gradient_reaches_every_input(self) -> None:
        """Backpropagates whole-question error through a toy decision encoder."""
        encoder = torch.nn.Linear(3, 10)
        inputs = torch.randn(15, 3, requires_grad=True)
        _, question = losses.question_losses(
            encoder(inputs).log_softmax(dim=1),
            torch.full((15,), 4),
            (15,),
        )
        question.backward()
        if inputs.grad is None or encoder.weight.grad is None:
            self.fail("Whole-question loss detached the decision encoder")
        self.assertTrue(torch.all(inputs.grad.norm(dim=1) > 0))
        self.assertGreater(encoder.weight.grad.norm().item(), 0)

    def test_pool_probability_floor(self) -> None:
        """Preserves production clipping even for very unlikely ranks."""
        probabilities = np.full((5, 10), 1e-9, dtype=np.float64)
        probabilities[:, 4] = 1 - 9e-9
        _, actual = losses.question_losses(
            torch.from_numpy(np.log(probabilities)),
            torch.zeros(5, dtype=torch.long),
            (5,),
        )
        expected = -np.log(rank_lib.pool(probabilities)[0])
        self.assertAlmostEqual(actual.item(), expected, places=10)

    def test_balances_bags(self) -> None:
        """A twenty-game query must not receive four times a five-game weight."""
        logp = torch.randn(25, 10).log_softmax(dim=1)
        labels = torch.tensor([1] * 5 + [7] * 20)
        game, _ = losses.question_losses(logp, labels, (5, 20))
        expected = (
            torch.nn.functional.nll_loss(logp[:5], labels[:5])
            + torch.nn.functional.nll_loss(logp[5:], labels[5:])
        ) / 2
        self.assertTrue(torch.equal(game, expected))

    def test_invalid_bags(self) -> None:
        """Rejects incomplete bags and inconsistent player labels."""
        with self.assertRaises(ValueError):
            losses.question_losses(torch.zeros(5, 10), torch.zeros(5), (4,))
        with self.assertRaises(ValueError):
            losses.question_losses(
                torch.zeros(5, 10), torch.tensor([0, 0, 1, 0, 0]), (5,)
            )
