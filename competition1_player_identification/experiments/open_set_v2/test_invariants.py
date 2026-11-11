"""Tests leakage, perspective, metric and mixed-precision training invariants.

Run with python -m unittest followed by this module's full package path.
"""

import unittest
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

from competition1_player_identification import extract_submission_features
from competition1_player_identification.experiments.open_set_v2 import cache
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import models
from competition1_player_identification.experiments.open_set_v2 import prefix
from competition1_player_identification.experiments.open_set_v2 import training
from competition1_player_identification.experiments.open_set_v2 import (
    rank_prior,
)


class RetrievalInvariants(unittest.TestCase):
    """Checks failures that can silently inflate an identity-retrieval score."""

    def test_dynamic_shards_exclude_concurrent_owners(self) -> None:
        """Keeps the same lock inode and permits one writer per shard."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "shard_00000.npz"
            with extract_submission_features.claim_shard(path, True) as owner:
                with extract_submission_features.claim_shard(
                    path, True
                ) as other:
                    self.assertTrue(owner)
                    self.assertFalse(other)
            with extract_submission_features.claim_shard(path, True) as resumed:
                self.assertTrue(resumed)

    def test_dynamic_claim_recovers_after_process_termination(self) -> None:
        """Relies on OS lock release rather than deleting a stale lock file."""
        script = (
            "import fcntl\nimport sys\nimport time\n"
            "with open(sys.argv[1], 'a+b') as handle:\n"
            "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
            "    print('locked', flush=True)\n"
            "    time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "shard_00000.npz"
            with subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    script,
                    str(path.with_suffix(".lock")),
                ],
                stdout=subprocess.PIPE,
                text=True,
            ) as child:
                try:
                    if child.stdout is None:
                        raise RuntimeError("Child stdout was not created")
                    self.assertEqual(child.stdout.readline().strip(), "locked")
                    with extract_submission_features.claim_shard(
                        path, True
                    ) as owner:
                        self.assertFalse(owner)
                finally:
                    child.terminate()
                    child.wait(timeout=5)
            with extract_submission_features.claim_shard(path, True) as owner:
                self.assertTrue(owner)

    def test_cohorts_exclude_identity_and_shared_game_content(self) -> None:
        """Separates two full cohorts even when input contains duplicate games."""
        groups = [
            np.arange(owner * 16, (owner + 1) * 16) for owner in range(1200)
        ]
        hashes = np.asarray([str(index) for index in range(19200)])
        hashes[16] = hashes[0]
        protected: set[str] = set()
        generator = np.random.default_rng(13)
        first, owners = dataset.choose_cohort(
            groups,
            hashes,
            protected,
            generator,
            set(),
        )
        second, other_owners = dataset.choose_cohort(
            groups,
            hashes,
            protected,
            generator,
            owners,
        )
        self.assertFalse(owners & other_owners)
        self.assertFalse(
            set(hashes[first].ravel()) & set(hashes[second].ravel())
        )
        self.assertEqual(len(protected), 18000)

    def test_rank_prior_ordinal_probabilities_are_normalized(self) -> None:
        """Preserves the Competition 2 ordinal-head probability invariant."""
        model = rank_prior.C2MoveRanker()
        values = model.rank_log_probabilities(torch.randn(4, 192))
        self.assertTrue(torch.allclose(values.exp().sum(1), torch.ones(4)))
        (-values[:, 4].mean()).backward()
        self.assertIsNotNone(model.parameters_by_name["first"].grad)

    def test_full_game_hash_ignores_metadata_but_keeps_late_moves(self) -> None:
        """Includes actual moves while excluding anonymized player metadata."""
        base = "(;SZ[19]PB[Anonymous];B[aa];W[bb])"
        renamed = "(;SZ[19]PB[Different];B[aa];W[bb])"
        first = cache.parse_game(base)
        self.assertEqual(first[0], cache.parse_game(renamed)[0])
        self.assertEqual(
            cache.parse_game("(;SZ[19]HD[0];B[aa];W[bb])")[0],
            cache.parse_game("(;SZ[19]HD[3];B[aa];W[bb])")[0],
        )
        self.assertNotEqual(
            first[0], cache.parse_game(base[:-1] + ";B[cc])")[0]
        )
        self.assertEqual(first[1][:2].tolist(), [1, 383])

    def test_sampler_never_reuses_reference_as_query(self) -> None:
        """Separates query and reference games within every sampled episode."""
        groups = [
            np.arange(owner * 20, (owner + 1) * 20) for owner in range(32)
        ]
        generator = np.random.default_rng(17)
        for _ in range(20):
            episode = training.sample_episode(groups, generator)
            self.assertEqual(
                len(np.unique(episode.indices)), len(episode.indices)
            )
            for references, queries in zip(episode.references, episode.queries):
                self.assertFalse(
                    set(episode.indices[references])
                    & set(
                        episode.indices[queries],
                    )
                )

    def test_singletons_contribute_no_trivial_query(self) -> None:
        """Avoids teaching a perfect match from an identical reference game."""
        groups = [np.asarray([owner]) for owner in range(16)]
        episode = training.sample_episode(groups, np.random.default_rng(19))
        self.assertTrue(
            all(part.start == part.stop for part in episode.queries)
        )
        values = torch.randn(
            16,
            8,
            requires_grad=True,
        )
        self.assertEqual(float(training.episodic_loss(values, episode)), 0.0)

    def test_augmentation_preserves_actor_padding_and_pass(self) -> None:
        """Rotates board points without changing whose turn or pass it was."""
        values = torch.tensor(
            [
                [
                    1,
                    383,
                    362,
                    724,
                    0,
                ]
            ]
        ).repeat(16, 1)
        changed = training.augment_moves(values, np.random.default_rng(13))
        self.assertTrue(torch.equal(changed[:, 2:], values[:, 2:]))
        self.assertTrue(
            torch.equal((changed[:, :2] - 1) // 362, (values[:, :2] - 1) // 362)
        )

    def test_prefix_matching_is_invariant_under_board_symmetry(self) -> None:
        """Matches equivalent openings without losing target-player color."""
        tokens = np.asarray(
            [
                [
                    1,
                    383,
                    44,
                    401,
                    121,
                    629,
                    17,
                    700,
                ]
            ]
        )
        tables = prefix.symmetry_tables()
        colors = np.asarray([0])
        first = prefix.fingerprints(tokens, colors)
        for table in tables:
            self.assertTrue(
                np.array_equal(
                    first,
                    prefix.fingerprints(table[tokens], colors),
                )
            )
        self.assertFalse(
            np.array_equal(
                first,
                prefix.fingerprints(tokens, np.asarray([1])),
            )
        )

    def test_metric_uses_exponential_reward_and_stable_ties(self) -> None:
        """Checks exact contest rewards including deterministic score ties."""
        scores = np.tile(
            np.eye(6),
            (
                4,
                1,
                1,
            ),
        )
        self.assertEqual(evaluation.summarize(scores)["score"], 1.0)
        tied = np.zeros(
            (
                4,
                6,
                6,
            )
        )
        expected = np.exp(-np.arange(5)).sum() / 6
        self.assertAlmostEqual(evaluation.summarize(tied)["score"], expected)

    def test_arcface_backpropagates_under_autocast(self) -> None:
        """Checks autocast dtype compatibility and ordinary gradient flow."""
        model = models.Encoder(
            models.Specification(
                "opening",
                8,
                3,
                16,
            )
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            embeddings = model(torch.randn(6, 8), torch.zeros(6).long())
            loss = model.arcface_loss(embeddings, torch.arange(6) % 3)
        self.assertTrue(torch.isfinite(loss))
        loss = model.arcface_loss(
            model(torch.randn(6, 8), torch.zeros(6).long()),
            torch.arange(6) % 3,
        )
        loss.backward()
        self.assertIsNotNone(model.class_weights.grad)


if __name__ == "__main__":
    unittest.main()
