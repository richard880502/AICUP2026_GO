"""Tests legal-policy normalization and retained played-move gradients.

Run with python -m unittest and this module's full package path.
"""

import unittest

import torch

from competition2_rank_prediction.experiments.policy_adaptation import (
    likelihood,
)


class LikelihoodTest(unittest.TestCase):
    """Checks illegal mass cannot influence supervised policy likelihood."""

    def test_normalization_and_gradient(self) -> None:
        """A huge illegal logit has exactly zero effect and gradient."""
        logits = torch.tensor([[0.0, 1000.0, 0.0]], requires_grad=True)
        result = likelihood.played_log_probability(
            logits, torch.tensor([0]), torch.tensor([[True, False, True]])
        )
        torch.testing.assert_close(result, torch.tensor([-0.69314718]))
        (-result.sum()).backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[-0.5, 0, 0.5]]))

    def test_invalid_or_illegal_point_rejected(self) -> None:
        """Targets outside legal support cannot silently train on invalid data."""
        logits = torch.zeros((1, 3))
        legal = torch.tensor([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "illegal"):
            likelihood.played_log_probability(logits, torch.tensor([1]), legal)
        with self.assertRaisesRegex(ValueError, "outside"):
            likelihood.played_log_probability(logits, torch.tensor([3]), legal)


if __name__ == "__main__":
    unittest.main()
