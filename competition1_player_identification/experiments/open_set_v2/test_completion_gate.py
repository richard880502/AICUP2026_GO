"""Checks that incomplete or changed experiments cannot open final outcomes.

All fixtures use temporary synthetic artifacts. No real final cohort is read.
"""

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import (
    final_evaluate,
)
from competition1_player_identification.experiments.open_set_v2 import fusion
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.open_set_v2 import replay
from competition1_player_identification.experiments.open_set_v2 import seal


class CompletionGateInvariants(unittest.TestCase):
    """Protects the independent cohort from incomplete or changing decisions."""

    def test_final_requires_a_lock_before_reading_any_cohort_files(
        self,
    ) -> None:
        """Rejects unsealed final inference before trying to load its arrays."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "prior configuration lock"):
                infer.positions(root, "final_holdout")
            self.assertEqual(list(root.iterdir()), [])

    def test_incomplete_work_cannot_be_sealed(self) -> None:
        """Missing stages do not produce a misleading completed lock."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaises(FileNotFoundError):
                seal.create_lock(root)
            self.assertFalse((root / "locked_configuration.json").exists())

    def test_artifact_changes_invalidate_the_seal(self) -> None:
        """Detects model or normalization edits after decisions were frozen."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.bin"
            path.write_bytes(b"frozen checkpoint")
            configuration = {"files": {str(path): artifacts.file_digest(path)}}
            seal.verify_files(configuration)
            path.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                seal.verify_files(configuration)

    def test_existing_final_outcomes_prevent_late_sealing(self) -> None:
        """Cannot pretend a configuration was fixed before final was opened."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "final_results.json").write_text("{}")
            with self.assertRaisesRegex(
                ValueError, "without the required prior lock"
            ):
                seal.create_lock(root)

    def test_a_different_final_attempt_cannot_replace_the_recipe(self) -> None:
        """Rejects configuration swaps before invoking any real final inference."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "locked_configuration.json").write_text(
                json.dumps({"files": {}})
            )
            (root / "final_attempt.json").write_text(
                json.dumps({"configuration_sha256": "different"})
            )
            with mock.patch.object(infer, "cohort_scores") as prediction:
                with self.assertRaisesRegex(ValueError, "Cannot replace"):
                    final_evaluate.run(root)
                prediction.assert_not_called()

    def test_serialized_final_metrics_reject_nonfinite_predictions(
        self,
    ) -> None:
        """Checks the prediction artifact rather than trusting a report flag."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "scores.npz"
            scores = np.tile(np.eye(600, dtype=np.float32), (4, 1, 1))
            with path.open("wb") as output:
                np.savez_compressed(output, baseline=scores, proposed=scores)
            metrics = final_evaluate.stored_metrics(path)
            self.assertEqual(metrics["proposed"]["score"], 1.0)
            scores[0, 0, 0] = np.nan
            with path.open("wb") as output:
                np.savez_compressed(output, baseline=scores, proposed=scores)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                final_evaluate.stored_metrics(path)

    def test_validation_replay_detects_changed_rankings(self) -> None:
        """Allows affine score normalization but rejects changed predictions."""
        scores = np.tile(np.eye(6, dtype=np.float32), (4, 1, 1))
        expected = {"branch": fusion.standardize(scores)}
        same = replay.source_differences({"branch": scores}, expected)
        self.assertEqual(same["branch"]["rank_disagreements"], 0)
        scores[0, 0, 1] = 2
        changed = replay.source_differences({"branch": scores}, expected)
        self.assertEqual(changed["branch"]["rank_disagreements"], 1)


if __name__ == "__main__":
    unittest.main()
