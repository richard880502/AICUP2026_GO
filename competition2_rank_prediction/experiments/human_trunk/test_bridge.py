"""Tests metadata, safe vendor namespace handling and policy comparison.

Run: python -m unittest competition2_rank_prediction.experiments.human_trunk.test_bridge
"""

import pathlib
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch

import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import classifier
from competition2_rank_prediction.experiments.human_trunk import equivalence
from competition2_rank_prediction.experiments.human_trunk import extract
from competition2_rank_prediction.experiments.human_trunk import smoke
from competition2_rank_prediction.experiments.human_trunk import temporal


class BridgeTest(unittest.TestCase):
    """Checks code-level invariants without requiring CUDA or loading weights."""

    def test_metadata_is_fixed(self) -> None:
        """Uses neutral1d metadata and preserves any existing vendor-name alias."""
        repository = pathlib.Path(__file__).resolve().parents[3]
        sentinel = sys.modules.get("board")
        modules = bridge.load_vendor(
            repository
            / "runtime/competition2_research/vendor/KataGo-human-v1.15.0/python"
        )
        self.assertIs(sys.modules.get("board"), sentinel)
        metadata = smoke.neutral_metadata(modules)
        self.assertEqual(metadata.inverseBRank, 9)
        self.assertEqual(metadata.inverseWRank, 9)
        self.assertEqual(metadata.source, 2)
        self.assertEqual(str(metadata.gameDate), "2020-03-01")
        first = metadata.get_metadata_row("B", 361)
        second = metadata.get_metadata_row("W", 361)
        np.testing.assert_array_equal(first, second)

    def test_legal_move_renormalization(self) -> None:
        """Ignores network mass on engine-illegal moves before comparison."""
        actual = equivalence.compare(
            np.asarray([[0.1, 0.2, 0.1]]), np.asarray([[0.25, 0.5, -1.0]])
        )
        distance = actual["mean_total_variation"]
        if not isinstance(distance, float):
            self.fail("Policy distance must be a float")
        self.assertAlmostEqual(distance, 0)
        self.assertTrue(actual["passes_tolerance"])

    def test_invalid_comparison(self) -> None:
        """Rejects malformed policies instead of reporting a misleading NaN."""
        with self.assertRaises(ValueError):
            equivalence.compare(np.zeros((1, 2)), np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            equivalence.compare(
                np.asarray([[np.nan, 1]]), np.asarray([[0.5, 0.5]])
            )
        with self.assertRaises(ValueError):
            equivalence.compare(np.ones((1, 2)), np.full((1, 2), -1))

    def test_failed_import_restores_aliases(self) -> None:
        """A module loader failure cannot leave legacy top-level names behind."""
        original = {name: sys.modules.get(name) for name in bridge.MODULES}
        with unittest.mock.patch.object(
            bridge.importlib.util, "spec_from_file_location", return_value=None
        ):
            with self.assertRaises(ImportError):
                bridge.load_vendor(pathlib.Path("unused"))
        for name, value in original.items():
            self.assertIs(sys.modules.get(name), value)

    def test_masked_phase_statistics(self) -> None:
        """Padding never affects moments and an empty phase remains finite."""
        projected = np.full((1, 16, 64), 999, dtype=np.float32)
        projected[:, :8] = 2
        masks = np.zeros((1, 16), dtype=bool)
        masks[:, :8] = True
        summaries = classifier.phase_statistics(projected, masks)
        self.assertEqual(summaries.shape, (1, 896))
        np.testing.assert_allclose(summaries[0, :64], 2)
        np.testing.assert_allclose(summaries[0, 64:128], 0)
        self.assertTrue(np.isfinite(summaries).all())

    def test_no_eligible_decisions_are_masked(self) -> None:
        """A pass-only record cannot become an invented valid embedding."""
        inputs, points, error = extract.encode_job(
            "unused", {}, "(;SZ[19];B[])", "B"
        )
        self.assertIsNone(inputs)
        self.assertEqual(len(points), 0)
        self.assertEqual(error, "no eligible decisions")

    def test_temporal_training_matches_original(self) -> None:
        """History recording does not change the original trainer's updates."""
        torch.set_num_threads(2)
        settings = temporal.recipe()
        settings.device = "cpu"
        settings.epochs = 2
        settings.batch = 8
        generator = np.random.default_rng(14)
        data = (
            generator.normal(size=(64, 4, 5)).astype(np.float32),
            np.ones((64, 4), dtype=bool),
            np.repeat(np.arange(8), 8),
            np.repeat(np.arange(8), 8),
        )
        original, original_info = transformer_rank.train_one(
            data, settings, 17, torch.device("cpu")
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate, candidate_info = temporal.train_seed(
                data, settings, 17, pathlib.Path(directory)
            )
        self.assertEqual(original_info, candidate_info)
        for name, value in original.state_dict().items():
            self.assertTrue(
                torch.equal(value, candidate.state_dict()[name]), name
            )
