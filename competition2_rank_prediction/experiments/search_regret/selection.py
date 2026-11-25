"""Selects surprising cached legal decisions without rank-label inputs.

The96decision index exactly follows production's original sampling rule.
"""

import pathlib
import sys

import numpy as np


# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def select_turns(
    moves: list, color: str, surprise: np.ndarray, masks: np.ndarray
) -> list[tuple[int, float]]:
    """Selects four surprising middle-game decisions with fixed fallback.

    Args:
        moves: Parsed SGF moves, in original order.
        color: Actual actor color to select.
        surprise: Normal-policy surprisals from the fixed96decision cache.
        masks: Matching cached legal/available-decision mask.

    Raises:
        ValueError: Color or cached array shapes are invalid.
    """
    if (
        color not in ("B", "W")
        or surprise.shape != (96,)
        or masks.shape != (96,)
    ):
        raise ValueError(
            "Selection requires actorB/W and matching96decision arrays"
        )
    turns = extract_rank_features.katago.sample_player_turns(moves, color, 96)
    candidates = []
    for index, turn in enumerate(turns):
        if masks[index] and np.isfinite(surprise[index]):
            candidates.append((turn, float(surprise[index])))
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    upper = min(250, len(moves) - 10)
    preferred = [pair for pair in candidates if 20 <= pair[0] <= upper]
    chosen = preferred[:4]
    for candidate in candidates:
        if len(chosen) >= 4:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return sorted(chosen)
