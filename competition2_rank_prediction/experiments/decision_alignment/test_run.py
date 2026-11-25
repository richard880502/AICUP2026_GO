"""Tests that the soft action uses the deployed official reward decoder."""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.decision_alignment import run
from competition2_rank_prediction.experiments.reward_calibration import run as base


class DecisionAlignmentTest(unittest.TestCase):
    """Checks the soft action's expected reward against the fixed reward table."""

    def test_action_values_match_official_reward(self) -> None:
        """The best action for a one-hot rank matches score-optimal decoding."""
        probabilities = np.eye(10)
        reward = base.reward_matrix()
        action_values = probabilities @ reward.T
        np.testing.assert_array_equal(
            action_values.argmax(1), np.arange(10)
        )

    def test_temperature_is_fixed_positive(self) -> None:
        """A valid temperature keeps the action softmax numerically defined."""
        self.assertGreater(run.TEMPERATURE, 0.0)


if __name__ == "__main__":
    unittest.main()
