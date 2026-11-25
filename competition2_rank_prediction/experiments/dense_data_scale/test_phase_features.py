"""Tests masked phase summaries independently of trained probe outcomes.

Run with python -m unittest and this module's full package path.
"""

import unittest

import numpy as np

from competition2_rank_prediction.experiments.dense_data_scale import (
    phase_features,
)


class PhaseFeaturesTest(unittest.TestCase):
    """Checks masked padding, empty phases and robust statistic semantics."""

    def test_masks_and_dimensions(self) -> None:
        """Masked garbage cannot change either matched representation."""
        generator = np.random.default_rng(1)
        values = generator.normal(size=(3, 96, 4, 3)).astype(np.float32)
        masks = np.ones((3, 96), dtype=bool)
        masks[0, 2:] = False
        masks[1] = False
        corrupted = np.where(masks[..., None, None], values, 1000)
        for rich, width in ((False, 210), (True, 840)):
            original = phase_features.summarize(values, masks, rich)
            changed = phase_features.summarize(corrupted, masks, rich)
            self.assertEqual(original.shape, (3, width))
            self.assertTrue(np.isfinite(original).all())
            np.testing.assert_allclose(original, changed)
            np.testing.assert_array_equal(original[1], 0)

    def test_quantiles_and_moments(self) -> None:
        """Known equally spaced values produce expected mean/median/skew."""
        values = np.arange(5, dtype=float).reshape(1, 5, 1)
        result = phase_features.phase_statistics(
            values, np.ones((1, 5), dtype=bool), True
        )
        np.testing.assert_allclose(
            result[0], [2, np.sqrt(2), 0.4, 1, 2, 3, 3.6, 0], rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
