"""Tests move-only overlap diagnostics without reading rank labels.

Run with unittest using this module's full package path.
"""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.training_overlap import audit


class AuditTest(unittest.TestCase):
    """Checks actor attribution and absent-match behavior."""

    def test_root_and_whitespace_invariance(self) -> None:
        """Root metadata does not change the legacy move hash."""
        self.assertEqual(
            audit.move_hash("(;SZ[19]HD[2];B[pd];W[dd])"),
            audit.move_hash("(;SZ[19]; B[pd]; W[dd])"),
        )

    def test_actor_and_missing_matches(self) -> None:
        """Opposite-color games cannot identify the queried actor."""
        sgf = "(;SZ[19];B[pd];W[dd])"
        index = {
            "hash": np.array([audit.move_hash(sgf)], dtype=np.uint64),
            "colour": np.array(["B"]),
            "player": np.array([123]),
        }
        row = {"question_id": "toy", "num_games": "5"}
        for game in range(1, 6):
            row[f"sgf_{game}"] = sgf
            row[f"color_{game}"] = "B"
        result = audit.check_question(row, index)
        self.assertEqual(result["same_actor_move_matches"], 5)
        self.assertEqual(result["distinct_anonymous_training_owners"], 1)
        for game in range(1, 6):
            row[f"color_{game}"] = "W"
        result = audit.check_question(row, index)
        self.assertEqual(result["same_actor_move_matches"], 0)
        self.assertEqual(result["opposite_actor_move_matches"], 5)
        self.assertEqual(result["distinct_anonymous_training_owners"], 0)
        for game in range(1, 6):
            row[f"sgf_{game}"] = "(;B[aa])"
        result = audit.check_question(row, index)
        self.assertEqual(result["same_actor_move_matches"], 0)
        self.assertEqual(result["opposite_actor_move_matches"], 0)


if __name__ == "__main__":
    unittest.main()
