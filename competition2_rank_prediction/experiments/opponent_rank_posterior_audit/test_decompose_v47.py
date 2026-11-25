"""CPU-only tests of v47 posterior ablation and non-oracle diagnostics."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from . import decompose_v47 as m


def fixture():
    """Synthetic, distinct players with two OOF folds and an HD2+ game."""
    rng = np.random.default_rng(13)
    n, games, classes = 12, 48, 30
    ranks = np.repeat(np.arange(10), 3)
    truth = np.array([3, 4, 5, 2, 3, 4, 5, 6, 7, 1, 2, 3], dtype=np.int64)
    base = rng.uniform(0.2, 1.5, size=(n, 10))
    base /= base.sum(1, keepdims=True)
    return {
        "order": [f"q{i}" for i in range(n)],
        "truth": torch.tensor(truth),
        "fold": np.tile([0, 1], n // 2),
        "player_rank": ranks,
        "holdout_classes": [1, 19],
        "kind": np.array(([0, 1, 2, 0] * (games // 4)), dtype=np.int64),
        "game_question": torch.tensor(np.repeat(np.arange(n), games // n)),
        "sims": torch.tensor(rng.normal(0, 0.2, size=(games, classes)), dtype=torch.float16),
        "base": torch.tensor(base, dtype=torch.float32),
    }


def labels(data):
    """Builds the per-question metadata rows for the synthetic data."""
    return [{"question_id": q, "player": f"p{i}",
             "rank": m.LEVELS[int(data["truth"][i])],
             "fold": str(int(data["fold"][i]))}
            for i, q in enumerate(data["order"])]


class TestV47Decomposition(unittest.TestCase):
    """Synthetic data and explicit reference mathematics; no rank/identity leakage."""

    def test_alignment_and_isolation(self):
        """Checks alignment and isolation."""
        data = fixture()
        rows = labels(data)
        meta = m.validate_metadata(data, rows[::-1])
        self.assertEqual(meta["truth"].tolist(), data["truth"].tolist())
        rows[1]["player"] = rows[0]["player"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            m.validate_metadata(data, rows)

    def test_mismatch_and_duplicate_ids(self):
        """Checks mismatch and duplicate ids."""
        data = fixture()
        with self.assertRaisesRegex(ValueError, "1:1"):
            m.validate_metadata(data, labels(data)[:-1])
        bad = fixture()
        bad["order"][1] = bad["order"][0]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            m.validate_metadata(bad, labels(data))

    # Archived research script: one linear procedure, intentionally not split.
    # pylint: disable-next=too-many-locals
    def test_production_forward_is_exact_reference(self):
        """Checks production forward is exact reference."""
        data = fixture()
        meta = m.validate_metadata(data, labels(data))
        ctx = m.prepare(data, meta, torch.device("cpu"))
        model = m.V47(ctx["ranks"], ctx["gallery_prior"])
        scores, kind, question, base = (ctx[x] for x in ("sims", "kind", "game_question", "base"))
        prediction = model(scores, kind, question, base)
        posterior = (scores / model.log_temperature.exp()).softmax(1) @ ctx["ranks"]
        unknown = torch.sigmoid(model.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * model.prior.softmax(0)
        offset = (torch.arange(10)[None] - torch.arange(10)[:, None]).clamp(-3, 3) + 3
        transition = model.offsets[:, offset].log_softmax(2)
        game = torch.logsumexp(posterior.clamp(min=1e-9).log()[:, :, None]
                               + transition[kind.clamp(max=2)], dim=1)
        game = game * (kind < 3)[:, None]
        total = torch.zeros(len(base), 10).index_add_(0, question, game)
        used = torch.zeros(len(base)).index_add_(0, question, (kind < 3).float()).clamp(min=1)
        reference = model.log_alpha.exp() * total / used[:, None] ** model.gamma + model.bias
        reference = reference + torch.sigmoid(model.base_weight) * 2 * base.log()
        torch.testing.assert_close(prediction, reference, rtol=0, atol=0)

    def test_no_retrieval_has_no_game_specific_similarity(self):
        """Checks no retrieval has no game specific similarity."""
        data = fixture()
        meta = m.validate_metadata(data, labels(data))
        ctx = m.prepare(data, meta, torch.device("cpu"))
        model = m.V47(ctx["ranks"], ctx["gallery_prior"])
        args = [ctx[x] for x in ("sims", "kind", "game_question", "base")]
        with torch.no_grad():
            model.offsets[0] = torch.linspace(-0.3, 0.5, 7)
        old = model(*args, mode="no_retrieval")
        args[0] = torch.rand_like(args[0])
        new = model(*args, mode="no_retrieval")
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertFalse(torch.equal(model(*args), model(*[ctx[x] for x in
                          ("sims", "kind", "game_question", "base")])))

    def test_perturbations_keep_simplex_and_folds(self):
        """Checks perturbations keep simplex and folds."""
        data = fixture()
        meta = m.validate_metadata(data, labels(data))
        ctx = m.prepare(data, meta, torch.device("cpu"))
        outputs, conf, fit = m.run_models(ctx, 3)
        self.assertEqual(len(fit), 2)
        self.assertEqual(set(outputs), {"full", "no_retrieval", "retrieval_softer",
                                        "retrieval_sharper", "transition_softer",
                                        "transition_sharper"})
        for probabilities in outputs.values():
            self.assertEqual(probabilities.shape, (12, 10))
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(conf).all())
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())

    def test_summary_has_no_oracle_or_unsupported_attribution(self):
        """Checks summary has no oracle or unsupported attribution."""
        data = fixture()
        meta = m.validate_metadata(data, labels(data))
        ctx = m.prepare(data, meta, torch.device("cpu"))
        out, conf, _fit = m.run_models(ctx, 3)
        report, arrays = m.summarize(ctx, meta, out, conf, bootstrap=200)
        self.assertFalse(report["production_gate_passed"])
        self.assertIn("no true opponent rank labels", report["status"])
        counts = report["error_sensitivity_not_causal_attribution"]
        self.assertEqual(sum(x["all_eligible"] for x in counts.values()),
                         report["official_compatible_questions"])
        self.assertEqual(arrays["qprob_base"].shape, (12, 10))
        self.assertEqual(report["by_arm"]["full"]["utility"],
                         float(m.utility(out["full"], meta["truth"]).mean()))

    def test_hd2_masks_official_compatibility(self):
        """Checks hd2 masks official compatibility."""
        data = fixture()
        data["kind"][0] = 3
        data["kind"][1:4] = 3
        meta = m.validate_metadata(data, labels(data))
        ctx = m.prepare(data, meta, torch.device("cpu"))
        out, conf, _ = m.run_models(ctx, 2)
        report, arrays = m.summarize(ctx, meta, out, conf, bootstrap=200)
        self.assertEqual(report["excluded_hd2plus_questions"], 1)
        self.assertEqual(report["all_hd2plus_questions"], 1)
        self.assertEqual(report["official_compatible_questions"], 11)
        self.assertFalse(arrays["official_compatible"][0])


if __name__ == "__main__":
    unittest.main()
