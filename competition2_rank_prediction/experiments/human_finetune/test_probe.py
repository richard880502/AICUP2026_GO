"""Tests graph retention in the isolated fine-tuning output adapter.

Run with python -m unittest and this module's full package path.
"""

import unittest

import torch

from competition2_rank_prediction.experiments.human_finetune import probe


class GradientOutputsTest(unittest.TestCase):
    """Ensures a reported trunk can actually update its producing parameters."""

    def test_retains_graph(self) -> None:
        """Capturing a trunk preserves the exact tensor and nonzero gradients."""
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        value = parameter.square()
        outputs = probe.GradientOutputs()
        outputs.report("trunkfinal", value)
        self.assertIs(outputs.returned["trunkfinal"], value)
        outputs.returned["trunkfinal"].sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([4.0]))

    def test_ignores_other_diagnostics(self) -> None:
        """Independent collectors cannot share tensors or accumulate other heads."""
        first = probe.GradientOutputs()
        second = probe.GradientOutputs()
        first.report("policy", torch.ones(1))
        self.assertFalse(first.returned)
        first.report("trunkfinal", torch.ones(1))
        self.assertFalse(second.returned)


if __name__ == "__main__":
    unittest.main()
