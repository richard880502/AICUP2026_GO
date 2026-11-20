"""CPU unit tests for known/unknown control calibration and optional v47 proxy."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import known_unknown_audit as audit


class ControlTests(unittest.TestCase):
    def test_player_disjoint_stable(self) -> None:
        ids = np.asarray(["known-a", "known-a", "known-b", "absent-a", "absent-a"])
        folds = audit.player_folds(ids)
        self.assertEqual(int(folds[0]), int(folds[1]))
        self.assertEqual(int(folds[3]), int(folds[4]))
        self.assertTrue(np.array_equal(folds, audit.player_folds(ids)))

    def test_known_absent_separability(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        y = np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]
        x = audit.features(np.r_[rng.normal(.70, .07, n), rng.normal(.25, .07, n)],
                           np.r_[rng.normal(.15, .04, n), np.maximum(rng.normal(.03, .015, n), .001)])
        groups = np.asarray([f"p-{i}" for i in range(2*n)])
        report, p, _ = audit.crossfit({"x": x, "y": y, "groups": groups,
                                        "n_known": n, "n_absent": n})
        self.assertGreater(report["control_auc"], .95)
        self.assertEqual(len(p), 2*n)
        self.assertIn("0.01", report["predeclared_fpr_screens"])
        self.assertFalse(np.isnan(p).any())

    def test_weight_equal_per_player(self) -> None:
        y = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
        group = np.asarray(["known-x", "known-x", "known-y", "absent-x"] * 1 + ["absent-x"] * 3)
        w = audit.group_weights(y, group)
        self.assertAlmostEqual(float(w[y == 1].sum()), .5)
        self.assertAlmostEqual(float(w[y == 0].sum()), .5)
        self.assertAlmostEqual(float(w[0:2].sum()), float(w[2]))

    def test_auc_tied(self) -> None:
        self.assertAlmostEqual(audit.auc(np.asarray([1, 0, 1, 0]), np.ones(4)), .5)

    def test_features_require_finite_nonnegative_margin(self) -> None:
        with self.assertRaises(ValueError):
            audit.features(np.asarray([.3, np.nan]), np.asarray([.1, .2]))
        with self.assertRaises(ValueError):
            audit.features(np.asarray([.3, .4]), np.asarray([-.2, .1]))

    def test_control_alignment_and_absent_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            csv_path = d / "present.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["player_id", "game_id", "note"])
                w.writeheader()
                for i in range(200):
                    w.writerow({"player_id": f"player{i}", "game_id": str(i),
                                "note": f"{i}:123"})
            controls = d / "controls.npz"
            np.savez(controls, p_top=np.full(200, .75), p_margin=np.full(200, .2),
                     a_top=np.full(18000, .3), a_margin=np.full(18000, .04))
            dset = audit.load_controls(controls, csv_path, 15)
            self.assertEqual(dset["n_absent"], 18000)
            self.assertEqual(len(np.unique(dset["groups"][200:])), 1200)
            report, _, _ = audit.crossfit(dset)
            self.assertGreater(report["control_auc"], .99)
            with self.assertRaises(ValueError):
                audit.load_controls(controls, csv_path, 14)

    def test_optional_v47_proxy_metadata(self) -> None:
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            state = d / "h3_sims.pt"
            oof = d / "v47_error_decomposition.npz"
            rng = np.random.default_rng(1)
            sims = rng.uniform(.2, .8, (8, 4)).astype(np.float32)
            torch.save({"sims": torch.from_numpy(sims),
                        "game_question": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
                        "order": ["q3", "q1", "q2", "q0"],
                        "kind": np.asarray([0, 0, 0, 0, 3, 3, 0, 0]),
                        "holdout_classes": [2],
                        "truth": torch.tensor([3, 2, 1, 0]),
                        "fold": np.asarray([0, 1, 0, 1])}, state)
            np.savez(oof, question_id=np.asarray(["q0", "q1", "q2", "q3"]),
                     truth=np.asarray([0, 2, 1, 3]), fold=np.asarray([1, 1, 0, 0]),
                     confidence_proxy=np.asarray([.6, .4, .2, .3]),
                     qprob_full=np.full((4, 10), .1),
                     official_compatible=np.asarray([1, 1, 0, 1]),
                     player=np.asarray(["p0", "p1", "p2", "p3"]))
            x = audit.features(np.full(40, .4), np.full(40, .1))
            y = np.r_[np.ones(20, int), np.zeros(20, int)]
            names = np.asarray([f"p{i}" for i in range(40)])
            m = audit.fit_logistic(x, y, names)
            report, arrays = audit.proxy_v47(state, oof, [m, m],
                                               {"x": x, "n_known": 20})
            self.assertEqual(report["questions"], 4)
            self.assertEqual(len(arrays["v4_known_proxy"]), 4)
            self.assertEqual(int(arrays["valid_game_count"][2]), 0)
            with np.load(oof) as original:
                bad = {k: np.asarray(original[k]) for k in original.files}
            bad["question_id"] = np.asarray(["q0", "BAD", "q2", "q3"])
            np.savez(oof, **bad)
            with self.assertRaises(ValueError):
                audit.proxy_v47(state, oof, [m, m], {"x": x, "n_known": 20})


if __name__ == "__main__":
    unittest.main()