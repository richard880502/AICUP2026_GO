"""Tests zero-initialized query-meta residual behavior."""

import unittest

import numpy as np
import torch

from competition2_rank_prediction.experiments.query_meta import model


class QueryMetaTest(unittest.TestCase):
    """Ensures the residual MLP starts as the exact identity mapping."""

    def test_initial_network_preserves_logits(self) -> None:
        """Zero output initialization leaves baseline logits unchanged."""
        network = model.build(7)
        features = torch.zeros((4, 7), dtype=torch.float32)
        logits = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        with torch.no_grad():
            result = logits + network(features)
        np.testing.assert_array_equal(result.numpy(), logits.numpy())


if __name__ == "__main__":
    unittest.main()
