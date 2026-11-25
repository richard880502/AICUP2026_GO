"""Tests lossless board-input packing and padded-position masks.

Run with python -m unittest and this module's full package path.
"""

import unittest
import unittest.mock

import numpy as np

from competition2_rank_prediction.experiments.human_finetune import cache


class CacheTest(unittest.TestCase):
    """Checks that input compression cannot silently change model inputs."""

    def test_short_game_padding_and_lossless_packing(self) -> None:
        """Two sampled decisions remain two valid positions after padding."""
        spatial = np.zeros((2, 22, 19, 19), dtype=np.float32)
        spatial[:, 0] = 1
        spatial[1, 1, 3, 4] = 1
        inputs = (spatial, np.zeros((2, 19)), np.ones((2, 192)))
        with (
            unittest.mock.patch.object(
                cache.equivalence, "make_inputs", return_value=inputs
            ),
            unittest.mock.patch.object(
                cache.extract, "worker_modules", return_value={}
            ),
            unittest.mock.patch.object(
                cache.extract_rank_features.katago,
                "sample_player_turns",
                return_value=[0, 2],
            ),
        ):
            result = cache.encode(
                "unused", {}, ("(;B[aa];W[bb];B[cc];W[dd])", "B")
            )
        recovered = np.unpackbits(result["spatial_packed"], axis=-1)[..., :19]
        np.testing.assert_array_equal(recovered[0], spatial[0])
        np.testing.assert_array_equal(recovered[7], spatial[1])
        np.testing.assert_array_equal(result["valid"], np.arange(8) < 2)
        self.assertEqual(
            result["points"].tolist(), [0, 40, 40, 40, 40, 40, 40, 40]
        )


if __name__ == "__main__":
    unittest.main()
