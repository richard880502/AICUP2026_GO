"""Computes matched raw/search regret features in each actor's own perspective.

These are finite-budget estimates, not exact game-theoretic move losses.
"""

import numpy as np


def source_values(before: dict, after: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns BLACK-perspective source values/visits for auditing."""
    first, second = before["rootInfo"], after["rootInfo"]
    if first["visits"] < 32 or second["visits"] < 32:
        raise ValueError(
            "Regret extraction requires32search visits at both positions"
        )
    if not before["moveInfos"]:
        raise ValueError("No searched best-child value")
    best = min(before["moveInfos"], key=lambda row: row["order"])
    trace = np.asarray(
        [
            first["rawLead"],
            first["rawWinrate"],
            best["scoreLead"],
            best["winrate"],
            second["rawLead"],
            second["rawWinrate"],
            second["scoreLead"],
            second["winrate"],
        ],
        dtype=np.float64,
    )
    visits = np.asarray(
        [first["visits"], second["visits"], best["visits"]], dtype=np.int32
    )
    if not np.isfinite(trace).all() or visits[2] < 1:
        raise ValueError(
            "Invalid searched source values or unvisited best child"
        )
    return trace, visits


def decision_values(
    before: dict, after: dict, color: str, surprise: float, fraction: float
) -> np.ndarray:
    """Returns raw/search score and winrate losses plus selection controls."""
    if color not in ("B", "W"):
        raise ValueError(f"Invalid actor color: {color!r}")
    trace, _ = source_values(before, after)
    sign = 1 if color == "B" else -1
    values = np.asarray(
        [
            sign * (trace[0] - trace[4]) / 20,
            sign * (trace[1] - trace[5]),
            sign * (trace[2] - trace[6]) / 20,
            sign * (trace[3] - trace[7]),
            surprise,
            fraction,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite searched/raw regret estimate")
    return values


def aggregate(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns seven raw and six incremental-search features per actor."""
    if not values:
        raw = np.full(7, np.nan)
        raw[-1] = 0
        return raw, np.full(6, np.nan)
    matrix = np.stack(values)
    if matrix.shape[1] != 6:
        raise ValueError(
            "Each decision must have six raw/search/context values"
        )
    raw_score = matrix[:, 0].clip(0, 2.5)
    raw_winrate = matrix[:, 1].clip(0, 1)
    search_score = matrix[:, 2].clip(0, 2.5)
    search_winrate = matrix[:, 3].clip(0, 1)
    difference = (matrix[:, 2] - matrix[:, 0]).clip(-2.5, 2.5)
    raw = np.asarray(
        [
            raw_score.mean(),
            raw_score.max(),
            raw_winrate.mean(),
            raw_winrate.max(),
            matrix[:, 4].mean(),
            matrix[:, 5].mean(),
            len(matrix) / 4,
        ]
    )
    search = np.asarray(
        [
            search_score.mean(),
            search_score.max(),
            search_winrate.mean(),
            search_winrate.max(),
            difference.mean(),
            np.abs(difference).mean(),
        ]
    )
    return raw, search
