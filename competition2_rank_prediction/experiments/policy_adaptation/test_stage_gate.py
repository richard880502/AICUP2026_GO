"""Tests the fixed inner-only seed0 cost gate without running a GPU fit.

Run with python -m unittest and this module's full package path.
"""

import json
import pathlib
import tempfile
import unittest

from competition2_rank_prediction.experiments.policy_adaptation import train


class StageGateTest(unittest.TestCase):
    """Ensures later frozen epochs cannot alter the matched early-budget gate."""

    def test_early_budget_and_threshold(self) -> None:
        """Only epochs5/10 participate, with the declared0.01NLL improvement."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            (directory / "adapt0_seed0_history.json").write_text(
                json.dumps(
                    [
                        {"epoch": 5, "inner_nll": 2.0},
                        {"epoch": 10, "inner_nll": 1.9},
                        {"epoch": 40, "inner_nll": 1.0},
                    ]
                )
            )
            self.assertTrue(
                train.cost_gate(
                    directory,
                    [
                        {"epoch": 5, "inner_nll": 2.1},
                        {"epoch": 10, "inner_nll": 1.88},
                    ],
                )
            )
            self.assertFalse(
                train.cost_gate(
                    directory,
                    [
                        {"epoch": 5, "inner_nll": 2.1},
                        {"epoch": 10, "inner_nll": 1.895},
                    ],
                )
            )


if __name__ == "__main__":
    unittest.main()
