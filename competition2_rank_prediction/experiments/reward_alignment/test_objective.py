"""Checks reward alignment, gradient behavior and baseline equivalence.

Run with python -m unittest <this module's full package path>.
"""

import math
import pathlib
import sys
import unittest
import unittest.mock

import numpy as np
import torch

from competition2_rank_prediction.experiments.reward_alignment import objective
from competition2_rank_prediction.experiments.reward_alignment import train

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


class ObjectiveTest(unittest.TestCase):
    """Verifies the official matrix and loss rather than a proxy reward."""

    def test_matrix_matches_scorer(self) -> None:
        """Checks all 100 prediction/ground-truth combinations."""
        matrix = objective.reward_matrix(torch.zeros(1)).numpy()
        for truth in range(10):
            for predicted in range(10):
                expected = rank_lib.competition_score(
                    np.asarray([predicted]),
                    np.asarray([truth]),
                )
                self.assertAlmostEqual(matrix[truth, predicted], expected)

    def test_zero_weight_exact_baseline(self) -> None:
        """Preserves both baseline values and gradients exactly."""
        logits = torch.randn(8, 10, requires_grad=True)
        labels = torch.arange(8)
        logp = logits.log_softmax(dim=1)
        baseline = torch.nn.functional.nll_loss(logp, labels)
        candidate = objective.mixed_loss(logp, labels, 0)
        self.assertTrue(torch.equal(baseline, candidate))
        first = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        second = torch.autograd.grad(candidate, logits)[0]
        self.assertTrue(torch.equal(first, second))

    def test_reward_and_gradient(self) -> None:
        """Checks analytic uniform reward and finite nonzero gradients."""
        logits = torch.zeros(1, 10, requires_grad=True)
        loss = objective.mixed_loss(
            logits.log_softmax(dim=1),
            torch.tensor([4]),
            0.2,
        )
        expected = math.log(10) + 0.2 * (1 - (1 + 2 / math.e) / 10)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        if logits.grad is None:
            self.fail("Missing objective gradient")
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 4].item(), 0)

    def test_invalid_inputs(self) -> None:
        """Rejects invalid coefficient or shape."""
        for weight in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                objective.mixed_loss(
                    torch.zeros(1, 10), torch.tensor([0]), weight
                )
        with self.assertRaises(ValueError):
            objective.mixed_loss(torch.zeros(1, 9), torch.tensor([0]), 0.2)

    def test_dense_questions(self) -> None:
        """Checks reproducible disjoint-game questions of official sizes."""
        players = np.asarray(["dense"] * 20 + ["sparse"] * 2)
        first = train.fixed_questions(players)
        second = train.fixed_questions(players)
        self.assertEqual(len(first), 20)
        self.assertEqual({len(rows) for rows in first}, {5, 10, 15, 20})
        for rows, repeated in zip(first, second):
            self.assertTrue(np.array_equal(rows, repeated))
            self.assertEqual(len(rows), len(np.unique(rows)))
            self.assertTrue(np.all(players[rows] == "dense"))
        with self.assertRaises(ValueError):
            train.fixed_questions(np.asarray(["sparse"] * 2))

    def test_supplement_excludes_optimizer_players(self) -> None:
        """Adds only inner players and removes duplicated game identifiers."""
        original = {
            "player_id": np.asarray(["optimizer", "inner"]),
            "game_key": np.asarray(["train0", "inner0"]),
            "rank": np.asarray(["D", "C"]),
        }
        dense = {
            "player_id": np.asarray(["optimizer", "inner", "inner", "outer"]),
            "game_key": np.asarray(["newtrain", "inner0", "inner1", "outer0"]),
            "rank": np.asarray(["D", "C", "C", "B"]),
        }
        with unittest.mock.patch.object(
            rank_lib, "load_features", return_value=dense
        ):
            merged, inner = train.supplement_inner(
                pathlib.Path("unused"), original, np.asarray([1])
            )
        self.assertEqual(
            merged["game_key"].tolist(), ["train0", "inner0", "inner1"]
        )
        self.assertEqual(inner.tolist(), [1, 2])
        self.assertEqual(merged["player_id"][0], "optimizer")
