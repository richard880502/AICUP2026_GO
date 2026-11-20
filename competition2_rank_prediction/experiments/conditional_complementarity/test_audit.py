"""CPU-only synthetic smoke tests; never depend on competition SGFs/checkpoints."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from competition2_rank_prediction.experiments.conditional_complementarity import (
    audit, extract_confidence,
)


class AuditTests(unittest.TestCase):
    """Alignment, exclusions, exact baseline identity, and deterministic outputs."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.p2 = root / "p2_report.npz"
        self.confidence = root / "confidence.npz"
        self.labels = root / "question_folds.csv"
        n = 80
        self.qid = np.array([f"q{i:03d}" for i in range(n)])
        truth = np.arange(n) % 10
        fold = np.arange(n) % 2
        rng = np.random.default_rng(27)
        base_logits = rng.normal(0, 0.3, (n, 10))
        base_logits[np.arange(n), truth] += 2
        original = np.exp(audit.log_softmax(base_logits))
        tenp3 = original.copy()
        tenp10 = original.copy()  # zero matched residual, regardless of nuisance
        np.savez(self.p2, question_id=self.qid, truth=truth, fold=fold,
                 v47_original=original, v47_tenp3=tenp3, v47_tenp10=tenp10)
        rows = list(range(n - 1, -1, -1))
        np.savez(self.confidence, question_id=self.qid[rows],
                 identity_margin=np.linspace(0, 0.05, n)[rows],
                 usable_games=np.full(n, 5)[rows],
                 all_games=np.full(n, 5)[rows],
                 has_hd2plus=(np.arange(n) % 7 == 0)[rows])
        with self.labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", "player", "rank", "fold"])
            writer.writeheader()
            for i in rows:
                writer.writerow({"question_id": self.qid[i], "player": f"player{i}",
                                 "rank": audit.LEVELS[truth[i]], "fold": fold[i]})

    def test_identical_arms_preserve_production(self) -> None:
        data = audit.load_inputs(self.p2, self.confidence, self.labels)
        report, predictions = audit.run(data, 100)
        mask = predictions["eligible"]
        np.testing.assert_allclose(predictions["conditional"][mask],
                                   data["original"][mask], atol=1e-12)
        self.assertEqual(report["paired"]["conditional_vs_original"]["delta"], 0)
        self.assertFalse(report["production_gate_passed"])
        self.assertEqual(report["n_excluded_hd2plus"], 12)
        self.assertFalse(report["exploratory_screen_only"])
        self.assertTrue(json.dumps(report, allow_nan=False))

    def test_cannot_guess_misaligned_ids(self) -> None:
        with np.load(self.confidence) as cache:
            arrays = {key: cache[key] for key in cache.files}
        arrays["question_id"][0] = "NOT_PRESENT"
        np.savez(self.confidence, **arrays)
        with self.assertRaisesRegex(ValueError, "question sets differ"):
            audit.load_inputs(self.p2, self.confidence, self.labels)

    def test_extract_masks_holdout_class(self) -> None:
        path = Path(self.tmp.name) / "sims.pt"
        torch.save({
            "sims": torch.tensor([[0.95, 0.7, 0.6], [0.1, 0.3, 0.2],
                                  [0.8, 0.4, 0.3]], dtype=torch.float16),
            "order": ["q0", "q1"], "game_question": torch.tensor([0, 0, 1]),
            "kind": np.asarray([0, 3, 1]), "holdout_classes": [0],
        }, path)
        result = extract_confidence.extract(path, batch=2)
        self.assertAlmostEqual(float(result["identity_margin"][0]), 0.1, places=2)
        self.assertAlmostEqual(float(result["identity_margin"][1]), 0.1, places=2)
        self.assertEqual(result["usable_games"].tolist(), [1, 1])
        self.assertEqual(result["has_hd2plus"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
