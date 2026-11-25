"""Tests the fixed reward table and score-optimal prediction utility."""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.reward_calibration import run


class RewardCalibrationTest(unittest.TestCase):
    """Checks exact and adjacent labels use the official score values."""

    def test_reward_matrix(self) -> None:
        """The diagonal and adjacent entries match the competition metric."""
        matrix = run.reward_matrix()
        np.testing.assert_array_equal(np.diag(matrix), np.ones(10))
        self.assertAlmostEqual(matrix[3, 4], np.exp(-1.0))
        self.assertEqual(matrix[3, 5], 0.0)

    def test_official_scores(self) -> None:
        """Predictions at exact/adjacent/distant ranks receive right scores."""
        probabilities = np.eye(10)[[3, 3, 3]]
        labels = np.array([3, 4, 5])
        np.testing.assert_allclose(
            run.official_scores(probabilities, labels),
            [1.0, np.exp(-1.0), 0.0],
        )


if __name__ == "__main__":
    unittest.main()
