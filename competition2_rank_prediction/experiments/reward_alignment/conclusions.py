"""Reports production-reference contrasts without changing experimental fits.

Example: python -m competition2_rank_prediction.experiments.reward_alignment.conclusions
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from competition2_rank_prediction.experiments.reward_alignment import train
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

matplotlib.use("Agg")
METHODS = ("original", "nll_nll", "nll_utility", "reward_utility")


def paired_interval(
    differences: np.ndarray, players: np.ndarray
) -> dict[str, float]:
    """Aggregates repeated queries before resampling independent players."""
    clustered = np.asarray(
        [differences[players == player].mean() for player in np.unique(players)]
    )
    return statistics.bootstrap_players(clustered, confidence=0.975)


def main() -> None:
    """Writes supplemental attribution and production comparisons once."""
    args = train.arguments()
    destination = args.output / "production_comparison.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite conclusions: {destination}"
        )
    with np.load(args.output / "evaluation.npz") as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    scores = {
        name: statistics.player_scores(arrays[name], arrays["y"])
        for name in METHODS
    }
    contrasts = {}
    for candidate, reference in (
        ("nll_utility", "original"),
        ("reward_utility", "original"),
        ("reward_utility", "nll_utility"),
    ):
        contrasts[f"{candidate}_minus_{reference}"] = paired_interval(
            scores[candidate] - scores[reference], arrays["player"]
        )
    report = {
        "utility": {
            name: float(values.mean()) for name, values in scores.items()
        },
        "paired_player_97_5ci": contrasts,
        "interpretation": (
            "Small-weight reward improves the matched retrained control; "
            "superiority over original production remains unconfirmed. "
            "Retain Public-best C1+C2 and seek fresh-player confirmation."
        ),
        "official_score": None,
        "new_submission_created": False,
    }
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(list(scores), [float(values.mean()) for values in scores.values()])
    axis.set_ylim(0.70, 0.72)
    axis.set_ylabel("Exploratory calibration-CV utility (truncated axis)")
    axis.set_title("Original C1+C2 versus matched own-branch replacements")
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png", dpi=160)
    plt.close(figure)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
