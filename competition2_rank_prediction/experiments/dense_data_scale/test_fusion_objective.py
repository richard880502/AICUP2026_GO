"""Tests bounded fusion invariants and both differentiable objectives.

Run with python -m unittest and this module's full package path.
"""

import unittest

import numpy as np
import torch

from competition2_rank_prediction.experiments.dense_data_scale import (
    fusion_objective,
)


class FusionObjectiveTest(unittest.TestCase):
    """Checks that fitting cannot silently alter total scale or lose gradients."""

    def test_weight_bounds_and_graph(self) -> None:
        """Zero shift preserves weights; extreme shifts remain bounded."""
        baseline = torch.tensor([0.7, 0.3, 0.5], dtype=torch.float64)
        raw = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        values = fusion_objective.weights(raw, baseline)
        torch.testing.assert_close(values, baseline)
        values[0].backward()
        if raw.grad is None:
            raise ValueError("Relative weight gradient is missing")
        self.assertGreater(float(raw.grad.abs().sum()), 0)
        values = fusion_objective.weights(
            torch.tensor([-1000, 0, 1000]), baseline
        )
        torch.testing.assert_close(values.sum(), baseline.sum())
        self.assertTrue((values >= baseline * np.exp(-0.4)).all())
        self.assertTrue((values <= baseline * np.exp(0.4)).all())

    def test_objectives_move_toward_correct_member(self) -> None:
        """Both objectives favor the member predicting the synthetic true class."""
        probability = np.full((4, 2, 10), 0.2 / 9)
        probability[:, 0, 0] = 0.8
        probability[:, 1, 1] = 0.8
        baseline = np.array([0.2, 0.8])
        inputs = {
            "members": np.log(probability),
            "rows": np.arange(4).reshape(4, 1),
            "mask": np.ones((4, 1), dtype=bool),
        }
        for method in ("nll", "utility"):
            values = fusion_objective.fit(
                inputs,
                np.zeros((4, 10)),
                np.zeros(4, dtype=int),
                baseline,
                method,
            )
            self.assertGreater(values[0], baseline[0])
            self.assertAlmostEqual(float(values.sum()), 1)
            predicted = fusion_objective.predict(
                inputs, np.zeros((4, 10)), values
            )
            np.testing.assert_allclose(predicted.sum(1), 1)

    def test_probability_floor_and_padding_parity(self) -> None:
        """Extreme fused probabilities retain the production pre-pooling floor."""
        probability = np.full((2, 2, 10), 1e-6)
        probability[0, :, 0] = 0.999991
        probability[1, :, 9] = 0.999991
        values = np.array([1.0, 0.8])
        inputs = {
            "members": np.log(probability),
            "rows": np.array([[0, 1, 0], [1, 0, 0]]),
            "mask": np.array([[True, True, False], [True, False, False]]),
        }
        logits = (inputs["members"] * values[None, :, None]).sum(1)
        fused = np.exp(logits - logits.max(1, keepdims=True))
        fused /= fused.sum(1, keepdims=True)
        expected = []
        for rows in (np.array([0, 1]), np.array([1])):
            pooled = np.log(np.clip(fused[rows], 1e-6, 1)).mean(0)
            result = np.exp(pooled - pooled.max())
            expected.append(result / result.sum())
        actual = fusion_objective.predict(inputs, np.zeros((2, 10)), values)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
