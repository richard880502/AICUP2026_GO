"""Checks color conventions, transitions, masking and capability rejection.

Run with python -m unittest <this module's full package path>.
"""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.human_context import extract
from competition2_rank_prediction.experiments.human_context import evaluate
from competition2_rank_prediction.experiments.human_context import probe


def root(winrate: float, score: float) -> dict[str, float]:
    """Builds matching normal/human fields with known units."""
    result = {}
    for fields in (probe.FIELDS, extract.NORMAL_FIELDS):
        for name, value in zip(fields, (winrate, score, 20, 0.1, 10)):
            result[name] = value
    return result


class FeatureTest(unittest.TestCase):
    """Verifies uncertainty is unsigned while player value changes perspective."""

    def test_color_conversion(self) -> None:
        """Converts winrate/lead but not uncertainty under a color reversal."""
        black = extract.root_values(root(0.6, 10), probe.FIELDS, "B")
        white = extract.root_values(root(0.6, 10), probe.FIELDS, "W")
        np.testing.assert_allclose(black, (0.6, 0.5, 1, 0.1, 0.5))
        np.testing.assert_allclose(white, (0.4, -0.5, 1, 0.1, 0.5))

    def test_post_move_and_missing_mask(self) -> None:
        """Uses the actual post-move position, masking absent root responses."""
        responses: dict[str, list[dict[str, object]]] = {}
        for index in range(3):
            responses[f"game:h{index}"] = [
                {"turnNumber": 2, "rootInfo": root(0.6, 10)},
                {"turnNumber": 3, "rootInfo": root(0.7, 12)},
            ]
        tokens, masks = extract.game_tokens(
            {"color": "W"}, [2, 8], responses, "game"
        )
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1:].any())
        np.testing.assert_allclose(tokens[0, 1, 5:7], (-0.1, -0.1), atol=1e-6)

    def test_missing_capability(self) -> None:
        """Does not declare an empty or incomplete response supported."""
        report = probe.summarize({}, 8)
        self.assertFalse(report["all_fields_present"])
        self.assertEqual(report["mean_profile_range"], {})

    def test_invalid_root(self) -> None:
        """Rejects invalid perspective, root type and nonfinite values."""
        with self.assertRaises(ValueError):
            extract.root_values(root(0.5, 0), probe.FIELDS, "unknown")
        with self.assertRaises(ValueError):
            extract.root_values(None, probe.FIELDS, "B")
        with self.assertRaises(ValueError):
            extract.root_values(root(float("nan"), 0), probe.FIELDS, "B")

    def test_statistics_and_missing(self) -> None:
        """Keeps missing streams NaN and lays out normal features first."""
        tokens = np.zeros((2, 16, 4, 10), dtype=np.float32)
        masks = np.zeros((2, 16, 4), dtype=bool)
        tokens[0, 0, 0] = 1
        tokens[0, 1, 0] = 3
        masks[0, :2, 0] = True
        summaries = evaluate.game_statistics(tokens, masks)
        np.testing.assert_allclose(summaries[0, :10], 2)
        np.testing.assert_allclose(summaries[0, 10:20], 1)
        self.assertTrue(np.isnan(summaries[0, 20:]).all())
        self.assertTrue(np.isnan(summaries[1]).all())

    def test_imputation_fitting_only(self) -> None:
        """Never uses testing values to fill a missing feature."""
        fitting, testing = evaluate.impute(
            np.asarray([[1, np.nan], [3, np.nan]]),
            np.asarray([[np.nan, 999]]),
        )
        np.testing.assert_allclose(fitting, [[1, 0], [3, 0]])
        np.testing.assert_allclose(testing, [[2, 999]])
