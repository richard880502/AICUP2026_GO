"""Tests label independence and ranking invariants of learned game matching.

Run with unittest using this module's full package path from the repo root.
"""

import unittest

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import pair_data
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)
from competition1_player_identification.experiments.open_set_v2 import (
    pair_model,
)


class MatchingInvariants(unittest.TestCase):
    """Checks mathematical behavior rather than reproducing the implementation."""

    def test_local_statistics_distinguish_same_mean_distributions(self) -> None:
        """Matches hand-computed two-direction similarities and local maxima."""
        query = np.eye(2, dtype=np.float32)
        references = np.stack(
            (
                np.tile(query[0], (10, 1)),
                np.tile(query, (5, 1)),
            )
        )
        features = pair_features.match_statistics(query, references)
        self.assertEqual(features.shape, (2, 13))
        self.assertTrue(np.allclose(features[:, 2:4], 0.5))
        self.assertAlmostEqual(features[0, 4], 0.5)
        self.assertAlmostEqual(features[1, 4], 1.0)
        self.assertAlmostEqual(features[0, 6], 0.0)
        self.assertAlmostEqual(features[1, 6], 1.0)
        self.assertTrue(np.allclose(features[:, 11:], [[0, 1], [0, 1]]))

    def test_matching_is_set_invariant_and_finite_for_single_query(
        self,
    ) -> None:
        """Permuting reference or query games cannot change aggregate features."""
        generator = np.random.default_rng(17)
        query = pair_features.normalize(generator.normal(size=(5, 8)))
        references = pair_features.normalize(generator.normal(size=(20, 10, 8)))
        original = pair_features.match_statistics(query, references)
        permuted = pair_features.match_statistics(
            query[::-1], references[:, ::-1]
        )
        self.assertTrue(np.allclose(original, permuted, atol=1e-6))
        self.assertTrue(
            np.isfinite(
                pair_features.match_statistics(query[:1], references)
            ).all()
        )

    def test_answers_are_not_pair_features(self) -> None:
        """Changing or withholding the answer affects labels, never features."""
        generator = np.random.default_rng(19)
        first = generator.normal(size=(375, 8)).astype(np.float32)
        second = generator.normal(size=(375, 12)).astype(np.float32)
        positions = np.arange(375).reshape(25, 15)
        bank = pair_features.build_bank(first, second, positions)
        known = pair_features.Task(0, 3, np.arange(25), truth_owner=0)
        hidden = pair_features.Task(0, 3, np.arange(25))
        first_rows = pair_features.make_rows((bank, known, 0.5, "both"))
        second_rows = pair_features.make_rows((bank, hidden, 0.5, "both"))
        self.assertTrue(
            np.array_equal(first_rows.features, second_rows.features)
        )
        self.assertTrue(np.array_equal(first_rows.top_ids, second_rows.top_ids))
        self.assertEqual(second_rows.label, -1)
        self.assertEqual(second_rows.truth_position, -1)
        self.assertEqual(first_rows.truth_position, 0)

    def test_missing_answer_is_never_forced_into_top_twenty(self) -> None:
        """Keeps a rank-25 true player out of the naturally mined candidates."""
        values = np.ones((25, 15, 2), np.float32)
        values[0, :10] *= -1
        flattened = values.reshape(-1, 2)
        bank = pair_features.build_bank(
            flattened, flattened, np.arange(375).reshape(25, 15)
        )
        rows = pair_features.make_rows(
            (
                bank,
                pair_features.Task(0, 5, np.arange(25), truth_owner=0),
                0.5,
                "b",
            )
        )
        self.assertEqual(rows.label, -1)
        self.assertEqual(len(rows.top_ids), 20)
        self.assertNotIn(0, rows.top_ids)
        self.assertEqual(rows.features.shape[1], 20)

    def test_reranking_cannot_change_tail_or_recover_a_stage_miss(self) -> None:
        """Allows within-pool promotion while preserving all outside scores."""
        stage = np.tile(np.linspace(1, 0, 25, dtype=np.float32), (2, 1))
        top = np.argsort(-stage, axis=1)[:, :20]
        logits = np.zeros((2, 20), np.float32)
        logits[:, 1] = 10
        changed = pair_features.reorder_scores(stage, top, logits, 1.0)
        self.assertEqual(int(changed[0].argmax()), 1)
        self.assertTrue(np.array_equal(changed[:, 20:], stage[:, 20:]))
        self.assertNotIn(24, np.argsort(-changed[1])[:20])
        self.assertTrue(
            np.array_equal(stage[0], np.linspace(1, 0, 25, dtype=np.float32))
        )

    def test_zero_blend_preserves_ranks_including_ties(self) -> None:
        """Retains the exact baseline when the selected model should be rejected."""
        stage = np.ones((2, 30), np.float32)
        stage[1] = np.linspace(0.9, -0.1, 30)
        order = np.argsort(-stage, axis=1, kind="stable")
        changed = pair_features.reorder_scores(
            stage, order[:, :20], np.zeros((2, 20)), 0.0
        )
        self.assertTrue(
            np.array_equal(order, np.argsort(-changed, axis=1, kind="stable"))
        )

    def test_pair_model_is_candidate_permutation_equivariant(self) -> None:
        """Uses candidate content rather than hard-coded identity positions."""
        model = pair_model.PairReranker(20).eval()
        features = torch.randn(3, 20, 20)
        order = torch.randperm(20)
        self.assertTrue(
            torch.allclose(
                model(features)[:, order], model(features[:, order]), atol=1e-6
            )
        )

    def test_pair_scaling_excludes_validation_and_stage_misses(self) -> None:
        """Fits scaling only on accepted training groups, with separate labels."""
        features = np.full((40, 20, 33), 2.0, np.float32)
        features[0] = 1e6
        targets = np.zeros(40, np.int64)
        targets[0] = -1
        train = pair_data.PairDataset(
            features,
            targets,
            np.zeros((40, 20), int),
            np.zeros((40, 600)),
            np.ones(40, int),
            np.zeros(40, int),
        )
        validation = pair_data.PairDataset(
            np.full((2, 20, 33), 100.0, np.float32),
            np.full(2, -1, int),
            np.zeros((2, 20), int),
            np.zeros((2, 600)),
            np.ones(2, int),
            np.full(2, -1, int),
        )
        runtime, mean, _ = pair_model.prepare_fit(
            (
                train,
                validation,
                pair_data.feature_columns("b"),
                "cpu",
            )
        )
        self.assertTrue(np.array_equal(mean, np.full(20, 2.0)))
        self.assertEqual(len(runtime.targets), 39)
        self.assertTrue(
            np.array_equal(runtime.validation, np.full((2, 20, 20), 10.0))
        )


if __name__ == "__main__":
    unittest.main()
