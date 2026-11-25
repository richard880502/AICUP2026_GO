"""Exercises coverage-pilot split isolation and cache rejection guards.

Run with python -m unittest followed by this module's full package path.
"""

import pathlib
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_data,
)


class CoverageTest(unittest.TestCase):
    """Checks that malformed pilot data cannot silently enter training."""

    def test_split_is_player_disjoint_and_order_invariant(self) -> None:
        """Reordering games cannot alter optimizer/inner membership."""
        rows = []
        for rank in range(10):
            for player in range(50):
                rows.append({"player_id": f"p{rank}_{player}", "rank": rank})
        frame = pd.DataFrame(rows)
        result = coverage_data.partitions(frame)
        self.assertEqual(len(result["optimizer"]), 400)
        self.assertEqual(len(result["inner"]), 100)
        self.assertFalse(result["optimizer"] & result["inner"])
        self.assertEqual(
            result,
            coverage_data.partitions(frame.sample(frac=1, random_state=1)),
        )
        frame.loc[len(frame)] = {"player_id": "p0_0", "rank": 1}
        with self.assertRaisesRegex(ValueError, "conflicting rank"):
            coverage_data.partitions(frame)

    def test_feature_rejection_guards(self) -> None:
        """Rejects incomplete, duplicated, misattributed and invalid tensors."""
        manifest = pd.DataFrame(
            {"game_key": ["g0"], "player_id": ["p0"], "rank": ["D"]}
        )
        payload: dict[str, Any] = {
            "profiles": np.array(["normal", "rank_5k", "rank_1d", "rank_5d"]),
            "game_key": np.array(["g0"]),
            "player_id": np.array(["p0"]),
            "rank": np.array(["D"]),
            "tokens": np.zeros((1, 96, 4, 3)),
            "masks": np.ones((1, 96), dtype=bool),
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            with self.assertRaisesRegex(ValueError, "Incomplete extraction"):
                coverage_data.validate_features(directory, manifest)
            path = directory / "feat_0000.npz"
            np.savez(path, **payload)
            coverage_data.validate_features(directory, manifest)
            np.savez(directory / "feat_0001.npz", **payload)
            with self.assertRaisesRegex(ValueError, "Duplicate cached"):
                coverage_data.validate_features(directory, manifest)
            (directory / "feat_0001.npz").unlink()
            payload["player_id"] = np.array(["wrong"])
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "Misaligned player_id"):
                coverage_data.validate_features(directory, manifest)
            payload["player_id"] = np.array(["p0"])
            payload["tokens"][0, 0, 0, 0] = np.nan
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                coverage_data.validate_features(directory, manifest)
            payload["tokens"][0, 0, 0, 0] = 0
            payload["profiles"] = payload["profiles"][::-1]
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "profile order"):
                coverage_data.validate_features(directory, manifest)


if __name__ == "__main__":
    unittest.main()
