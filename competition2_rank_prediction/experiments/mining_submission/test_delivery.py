"""Checks joint side/time alignment and submission decoding invariants.

Run with unittest before releasing the alternate submission.
"""

import pathlib
import tempfile
import unittest

import numpy as np

from competition2_rank_prediction.experiments.mining_submission import (
    joint_features,
)
from competition2_rank_prediction.experiments.mining_submission import run


class DeliveryTest(unittest.TestCase):
    """Checks the representation and CSV behaviors used for submission."""

    def test_joint_side_and_time(self) -> None:
        """Side/time channels use actual moves and exclude passes."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "moves.npz"
            np.savez(
                path,
                game_key=np.asarray(["g"]),
                tokens=np.asarray([[0, 512, 361, 513, 2]]),
                length=np.asarray([5]),
                colour=np.asarray([0]),
            )
            masks = np.zeros((1, 96), dtype=bool)
            masks[:, :2] = True
            own = {
                "game_key": np.asarray(["g"]),
                "tokens": np.zeros((1, 96, 4, 3), dtype=np.float32),
                "masks": masks,
            }
            tokens, valid = joint_features.inputs(own, own, path)
            self.assertEqual(tokens.shape, (1, 192, 17))
            self.assertEqual(int(valid.sum()), 4)
            np.testing.assert_array_equal(tokens[0, :96, 15], np.zeros(96))
            np.testing.assert_array_equal(tokens[0, 96:, 15], np.ones(96))
            np.testing.assert_allclose(tokens[0, :2, 16], [0, 0.8])
            np.testing.assert_allclose(tokens[0, 96:98, 16], [0.2, 0.6])

    def test_official_reward_decode(self) -> None:
        """Writes the reward-optimal label even when the probability mode differs."""
        probabilities = np.zeros((1, 10))
        probabilities[0, 1:4] = [0.4, 0.35, 0.25]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "submission.csv"
            run.write_submission(path, ["pub_q_0001"], probabilities)
            self.assertEqual(
                path.read_text(), "question_id,pred_rank\npub_q_0001,B\n"
            )
            with self.assertRaises(ValueError):
                run.write_submission(path, [], probabilities)
            with self.assertRaises(ValueError):
                run.write_submission(
                    path, ["pub_q_0001"], np.full((1, 10), np.nan)
                )


if __name__ == "__main__":
    unittest.main()
