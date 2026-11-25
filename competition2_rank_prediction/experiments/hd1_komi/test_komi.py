"""Tests paired-query isolation, actor conversion and mandatory visit checks.

Run with python -m unittest and this module's full package path.
"""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.hd1_komi import evaluate
from competition2_rank_prediction.experiments.hd1_komi import extract


class KomiTest(unittest.TestCase):
    """Ensures a komi hypothesis cannot silently change other representations."""

    def test_queries_differ_only_by_komi(self) -> None:
        """HD1 requires no stones or color swaps in this paired test."""
        queries = extract.queries([{"sgf_content": "(;HD[1];B[pd];W[dd])"}], 0)
        self.assertEqual([query["komi"] for query in queries], [6.5, 0.5])
        for query in queries:
            self.assertEqual(query["maxVisits"], 64)
            self.assertEqual(query["analyzeTurns"], [2])
            del query["id"]
            del query["komi"]
        self.assertEqual(queries[0], queries[1])

    def test_actor_conversion_and_fact_preservation(self) -> None:
        """Black/White values flip correctly and unselected rows stay exact."""
        original = np.arange(24, dtype=float).reshape(4, 6)
        data = {
            "game_key": np.array(["g0", "g1", "g2", "g3"]),
            "player": np.array(["p0", "p1", "p2", "p3"]),
            "y": np.arange(4),
            "final": original.copy(),
        }
        roots = {
            "game_key": np.array(["g1", "g2"]),
            "player": np.array(["p1", "p2"]),
            "color": np.array(["B", "W"]),
            "roots": np.array(
                [[[64, 0.8, 5], [64, 0.9, 11]], [[64, 0.8, 5], [64, 0.9, 11]]]
            ),
        }
        evaluate.splice(data, roots)
        np.testing.assert_allclose(
            data["final65"][1:3, :2], [[0.8, 5], [0.2, -5]]
        )
        np.testing.assert_allclose(
            data["final05"][1:3, :2], [[0.9, 11], [0.1, -11]]
        )
        np.testing.assert_array_equal(data["final05"][:, 2:], original[:, 2:])
        np.testing.assert_array_equal(data["final05"][[0, 3]], original[[0, 3]])
        np.testing.assert_array_equal(data["final"], original)

    def test_root_budget_guard(self) -> None:
        """Under-budget roots cannot silently enter the comparison."""
        with self.assertRaisesRegex(ValueError, "64visit"):
            extract.root_values(
                {"rootInfo": {"visits": 63, "winrate": 0.5, "scoreLead": 0}}
            )


if __name__ == "__main__":
    unittest.main()
