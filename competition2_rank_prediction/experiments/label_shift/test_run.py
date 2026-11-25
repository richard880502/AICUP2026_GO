"""Tests that prior adaptation uses only the unlabeled probability matrix."""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.label_shift import run


class LabelShiftTest(unittest.TestCase):
    """Checks the uniform marginal constraint and probability preservation."""

    def test_adaptation_matches_uniform_marginal(self) -> None:
        """Offsets align a skewed unlabeled class marginal to uniform."""
        probabilities = np.tile(
            np.array([[0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0, 0, 0]]),
            (200, 1),
        )
        adjusted, offsets = run.adapt(probabilities)
        self.assertEqual(offsets.shape, (10,))
        np.testing.assert_allclose(adjusted.sum(1), 1.0)
        np.testing.assert_allclose(adjusted.mean(0), np.full(10, 0.1), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
