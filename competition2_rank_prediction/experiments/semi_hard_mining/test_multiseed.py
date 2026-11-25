"""Checks seed aggregation, bootstrap units and configurable training guards.

Run with unittest alongside test_losses before evaluating replications.
"""

import pathlib
import tempfile
import unittest
import unittest.mock

import numpy as np

from competition2_rank_prediction.experiments.semi_hard_mining import (
    multiseed_evaluate,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics
from competition2_rank_prediction.experiments.semi_hard_mining import train


class ReplicationTest(unittest.TestCase):
    """Verifies that replication does not inflate independent sample count."""

    def test_bootstrap_unit(self) -> None:
        """Rejects seed-by-player matrices unless seeds are aggregated first."""
        with self.assertRaises(ValueError):
            statistics.bootstrap_players(np.zeros((3, 20)))
        self.assertEqual(
            statistics.bootstrap_players(np.zeros(20)),
            {"delta": 0.0, "low": 0.0, "high": 0.0},
        )

    def test_adjusted_interval(self) -> None:
        """Higher confidence widens the paired-player interval."""
        values = np.linspace(-1, 1, 20)
        ordinary = statistics.bootstrap_players(values)
        adjusted = statistics.bootstrap_players(values, confidence=0.975)
        self.assertLessEqual(adjusted["low"], ordinary["low"])
        self.assertGreaterEqual(adjusted["high"], ordinary["high"])
        with self.assertRaises(ValueError):
            statistics.bootstrap_players(values, confidence=1)

    def test_seed_average(self) -> None:
        """Includes weak seeds and keeps each player's repeated predictions paired."""
        labels = np.tile(np.arange(10), 2)
        baseline = np.tile(np.eye(10)[labels][None], (3, 1, 1))
        candidate = baseline.copy()
        candidate[0, 0] = np.eye(10)[3]
        candidate[1, 10] = np.eye(10)[3]
        report = multiseed_evaluate.summarize(
            {"A": baseline, "B": candidate},
            labels,
            np.full(20, 10),
        )
        interval = report["primary_paired_player_95ci"]
        if not isinstance(interval, dict):
            raise TypeError("Missing paired-player confidence interval")
        self.assertAlmostEqual(interval["delta"], -1 / 30)
        self.assertEqual(report["positive_utility_seeds"], 0)

    def test_invalid_budget(self) -> None:
        """Rejects invalid update budgets before creating an encoder."""
        with tempfile.TemporaryDirectory() as directory:
            context = unittest.mock.Mock(spec=train.TrainingContext)
            context.output = pathlib.Path(directory)
            with self.assertRaises(ValueError):
                train.train_arm(context, "A", train.TrainingRecipe(epochs=0))

    def test_screening_requires_pair_signal(self) -> None:
        """A small utility improvement alone does not pass the fixed gate."""
        baseline = {"utility": 0.6, "pair_auc": 0.7, "large_error": 0.1}
        candidate = {"utility": 0.61, "pair_auc": 0.7, "large_error": 0.1}
        result = statistics.screening_conditions(baseline, candidate, 0)
        self.assertTrue(result["utility_positive"])
        self.assertFalse(result["pair_auc_positive"])


if __name__ == "__main__":
    unittest.main()
