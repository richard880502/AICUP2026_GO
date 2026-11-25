"""Checks search-budget, color, sampling, controls and saved-trace invariants.

Run with python -m unittest <this module's full package path>.
"""

import pathlib
import sys
import unittest

import numpy as np

from competition2_rank_prediction.experiments.search_regret import extract
from competition2_rank_prediction.experiments.search_regret import evaluate
from competition2_rank_prediction.experiments.search_regret import features
from competition2_rank_prediction.experiments.search_regret import probe
from competition2_rank_prediction.experiments.search_regret import selection

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def response(
    raw_lead: float, search_lead: float, probabilities: tuple[float, float]
) -> dict:
    """Creates known BLACK-perspective32visit root/best-child values."""
    return {
        "rootInfo": {
            "visits": 32,
            "rawLead": raw_lead,
            "scoreLead": search_lead,
            "rawWinrate": probabilities[0],
            "winrate": probabilities[1],
        },
        "moveInfos": [
            {
                "order": 0,
                "visits": 16,
                "scoreLead": search_lead,
                "winrate": probabilities[1],
            }
        ],
    }


class SearchFeatureTest(unittest.TestCase):
    """Verifies search information rather than sign/selection artifacts."""

    def test_color_symmetry(self) -> None:
        """Mirrored BLACK outputs give equal White-perspective errors."""
        black = features.decision_values(
            response(2, 3, (0.65, 0.7)),
            response(1, 1, (0.55, 0.6)),
            "B",
            2.3,
            0.5,
        )
        white = features.decision_values(
            response(-2, -3, (0.35, 0.3)),
            response(-1, -1, (0.45, 0.4)),
            "W",
            2.3,
            0.5,
        )
        np.testing.assert_allclose(black, (0.05, 0.1, 0.1, 0.1, 2.3, 0.5))
        np.testing.assert_allclose(black, white)

    def test_real_search_budget_required(self) -> None:
        """Rejects one-visit or unvisited-child estimates as searched loss."""
        before, after = response(2, 3, (0.65, 0.7)), response(1, 1, (0.55, 0.6))
        before["rootInfo"]["visits"] = 1
        with self.assertRaises(ValueError):
            features.source_values(before, after)
        before["rootInfo"]["visits"] = 32
        before["moveInfos"][0]["visits"] = 0
        with self.assertRaises(ValueError):
            features.source_values(before, after)

    def test_control_and_search_layout(self) -> None:
        """Keeps raw controls distinct from incremental searched loss."""
        raw, searched = features.aggregate(
            [np.asarray([0.05, 0.1, 0.1, 0.1, 2.3, 0.5])]
        )
        np.testing.assert_allclose(raw, (0.05, 0.05, 0.1, 0.1, 2.3, 0.5, 0.25))
        np.testing.assert_allclose(searched, (0.1, 0.1, 0.1, 0.1, 0.05, 0.05))
        missing, missing_search = features.aggregate([])
        self.assertEqual(missing[-1], 0)
        self.assertTrue(np.isnan(missing[:-1]).all())
        self.assertTrue(np.isnan(missing_search).all())

    def test_selection_prefers_middle_and_masks(self) -> None:
        """Prefers cached middle-game decisions over opening/end outliers."""
        moves = []
        for index in range(300):
            moves.append(
                extract_rank_features.katago.SgfMove(
                    "B" if index % 2 == 0 else "W",
                    index % 19,
                    (index // 19) % 19,
                )
            )
        surprise = np.arange(96, dtype=float)
        masks = np.ones(96, dtype=bool)
        first = selection.select_turns(moves, "B", surprise, masks)
        self.assertEqual(
            first, selection.select_turns(moves, "B", surprise, masks)
        )
        self.assertEqual(len(first), 4)
        self.assertTrue(all(20 <= turn <= 250 for turn, _ in first))
        self.assertEqual(
            selection.select_turns(
                moves, "B", surprise, np.zeros(96, dtype=bool)
            ),
            [],
        )

    def test_saved_source_and_actor_order(self) -> None:
        """Places own/opponent controls first and masks missing decisions."""
        prepared = extract.Prepared("game", 100, "B", [(22, 2.3)], [(23, 1.3)])
        before = response(2, 3, (0.65, 0.7)) | {"turnNumber": 22}
        after = response(1, 1, (0.55, 0.6)) | {"turnNumber": 23}
        final = response(2, 3, (0.65, 0.7)) | {"turnNumber": 24}
        result = extract.assemble(prepared, [before, after, final])
        self.assertEqual(result.values.shape, (26,))
        self.assertEqual(int(result.masks.sum()), 2)
        np.testing.assert_allclose(result.values[:4], result.values[7:11])
        np.testing.assert_allclose(result.values[14:20], result.values[20:26])
        self.assertEqual(result.turns[0, 0], 22)
        np.testing.assert_array_equal(result.visits[0, 0], (32, 32, 16))

    def test_probe_summary(self) -> None:
        """Checks actual roots and raw/search differences in the probe."""
        report = probe.summarize({"game": [response(2, 3, (0.65, 0.7))]})
        self.assertEqual(report["positions"], 1)
        self.assertEqual(report["actual_visits_min"], 32)
        self.assertEqual(report["mean_abs_searched_minus_raw_lead"], 1)

    def test_identical_controls_cannot_pass(self) -> None:
        """Repeated queries do not invent gain or pass the advancement rule."""
        labels = np.repeat(np.arange(10), 40)
        probabilities = np.full((400, 10), 0.01)
        probabilities[np.arange(400), labels] = 0.91
        arrays: dict[str, np.ndarray] = {
            name: probabilities.copy() for name in evaluate.METHODS
        }
        arrays["y"] = labels
        arrays["player"] = np.arange(400)
        report = evaluate.summarize(arrays)
        self.assertFalse(report["screen_pass"])
        self.assertEqual(
            report["paired_player_97_5ci"]["search_minus_original"]["delta"], 0
        )

    def test_player_cluster_repeat_invariance(self) -> None:
        """Duplicating queries cannot shrink player-level uncertainty."""
        labels = np.repeat(np.arange(10), 40)
        probabilities = np.full((400, 10), 0.01)
        probabilities[np.arange(400), labels] = 0.91
        arrays: dict[str, np.ndarray] = {
            name: probabilities.copy() for name in evaluate.METHODS
        }
        arrays["y"], arrays["player"] = labels, np.arange(400)
        arrays["search"][:20] = 0.01
        arrays["search"][:20, 2] = 0.91
        first = evaluate.summarize(arrays)["paired_player_97_5ci"][
            "search_minus_original"
        ]
        repeated = {
            name: np.repeat(values, 20, axis=0)
            for name, values in arrays.items()
        }
        second = evaluate.summarize(repeated)["paired_player_97_5ci"][
            "search_minus_original"
        ]
        for field in ("delta", "low", "high"):
            self.assertAlmostEqual(first[field], second[field], places=12)


if __name__ == "__main__":
    unittest.main()
