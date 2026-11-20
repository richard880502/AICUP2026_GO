"""Synthetic CPU regression checks; no private corpus or checkpoints required."""
import unittest

import numpy as np

from competition2_rank_prediction.experiments.opponent_rank_posterior_audit.audit import (
    analyze,
    rank_posterior,
    split_players,
    temperature_crossfit,
    validate,
)


def toy() -> dict[str, np.ndarray]:
    """Repeated players, two gallery classes for each rank in a small simulation."""
    rng = np.random.default_rng(42)
    players = np.repeat(np.arange(80), 2)
    truth_class = players % 20
    class_rank = np.arange(20) // 2
    logits = rng.normal(0, 0.04, size=(len(players), 20))
    logits[np.arange(len(players)), truth_class] += 0.13
    # A controlled identity error within the same registered rank.
    logits[0, 1] = logits[0, 0] + 0.02
    return {
        "cosines": logits.astype(np.float32),
        "class_rank": class_rank,
        "true_class": truth_class.astype(int),
        "true_rank": class_rank[truth_class],
        "player_id": np.char.add("p", players.astype(str)),
        "game_id": np.char.add("g", np.arange(len(players)).astype(str)),
    }


class RankPosteriorTests(unittest.TestCase):
    """Deterministic regression checks on the highest-risk assumptions."""

    def test_rank_posterior_sums_classes_not_argmax(self):
        sims = np.asarray([[0.30, 0.31, 0.32], [0.5, 0.5, 0.0]], dtype=np.float32)
        ranks = np.asarray([0, 0, 1])
        p = rank_posterior(sims, ranks, 0.10)
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0))
        self.assertGreater(p[0, 0], p[0, 1])  # two weaker-class votes beat one top class
        self.assertGreater(p[1, 0], 0.98)

    def test_player_split_disjoint(self):
        data = toy()
        folds = split_players(data["player_id"], data["true_rank"])
        a = set(data["player_id"][folds == 0])
        b = set(data["player_id"][folds == 1])
        self.assertFalse(a.intersection(b))
        self.assertEqual({0, 1}, set(folds))

    def test_rejects_duplicate_and_wrong_rank(self):
        data = toy()
        bad = dict(data, game_id=data["game_id"].copy())
        bad["game_id"][2] = bad["game_id"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(bad)
        bad = dict(data, true_rank=data["true_rank"].copy())
        bad["true_rank"][0] = 9
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate(bad)

    def test_temperature_selection_on_opposite_player_fold(self):
        data = toy()
        folds = split_players(data["player_id"], data["true_rank"])
        a, weights = temperature_crossfit(
            data["cosines"], data["class_rank"], data["true_rank"], folds
        )
        self.assertEqual(a.shape, (len(data["player_id"]), 10))
        self.assertEqual(set(weights.keys()), {"0", "1"})
        # Perturb eval fold labels: the other fold's fitted temperature remains unchanged.
        shifted = data["true_rank"].copy()
        shifted[folds == 0] = (shifted[folds == 0] + 2) % 10
        _, shifted_weights = temperature_crossfit(
            data["cosines"], data["class_rank"], shifted, folds
        )
        self.assertEqual(weights["0"], shifted_weights["0"])

    def test_full_report_is_exploratory_not_v47_score(self):
        report = analyze(toy(), draws=120)
        self.assertEqual(report["sample"]["games"], 160)
        self.assertEqual(report["sample"]["players"], 80)
        self.assertTrue(report["no_official_test_labels_used"])
        self.assertTrue(report["not_v47_question_utility"])
        self.assertAlmostEqual(report["rank_oracle_top1"], 1.0)
        self.assertGreater(report["hard_identity_rank"]["rank_top1"],
                           report["identity_top1"])
        self.assertTrue(0 <= report["v47_soft_rank_fixed_t04"]["rank_top1"] <= 1)


if __name__ == "__main__":
    unittest.main()
