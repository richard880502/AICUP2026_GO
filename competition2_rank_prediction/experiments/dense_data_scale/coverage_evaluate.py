# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Scores completed coverage arms on identical historical validation questions.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Reports exploratory own-branch results, never official Public scores.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


def load(directory: pathlib.Path) -> tuple[dict, dict]:
    """Requires all six aligned probability arrays and complete histories."""
    predictions: dict = {2: [], 8: []}
    metadata: dict = {}
    figure, axis = plt.subplots(figsize=(8, 4))
    for seed in range(3):
        for cap in (2, 8):
            with np.load(
                directory / f"cap{cap}_seed{seed}_validation.npz"
            ) as payload:
                for name in ("game_key", "player", "y"):
                    values = np.asarray(payload[name])
                    if name in metadata and not np.array_equal(
                        metadata[name], values
                    ):
                        raise ValueError(
                            f"Mismatched prediction metadata: {name}"
                        )
                    metadata[name] = values
                probabilities = np.asarray(payload["probabilities"])
                if (
                    probabilities.shape != (8000, 10)
                    or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any()
                    or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
                ):
                    raise ValueError("Invalid prediction probabilities")
                predictions[cap].append(probabilities)
            history = json.loads(
                (directory / f"cap{cap}_seed{seed}_history.json").read_text()
            )
            if len(history) != 40 or not np.isfinite(history).all():
                raise ValueError("Incomplete or nonfinite training history")
            axis.plot(history, label=f"cap{cap} seed{seed}")
    axis.set_xlabel("Epoch index")
    axis.set_ylabel("Inner per-game NLL")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(figure)
    return predictions, metadata


def pooled(probabilities: np.ndarray, questions: list) -> np.ndarray:
    """Pools exactly the shared question game indices."""
    return np.stack([rank_lib.pool(probabilities[rows]) for rows in questions])


def summarize(predictions: dict, metadata: dict) -> dict:
    """Uses the preregistered ensemble score and paired-player bootstrap."""
    questions, labels = residual_shift.validation_questions(
        metadata["player"], metadata["y"]
    )
    players = np.asarray([metadata["player"][rows[0]] for rows in questions])
    rewards = {}
    seed_rewards = {}
    for cap in (2, 8):
        rewards[cap] = statistics.player_scores(
            pooled(np.mean(predictions[cap], axis=0), questions), labels
        )
        seed_rewards[cap] = np.stack(
            [
                statistics.player_scores(pooled(values, questions), labels)
                for values in predictions[cap]
            ]
        )
    delta = rewards[8] - rewards[2]
    by_player = np.asarray(
        [delta[players == player].mean() for player in np.unique(players)]
    )
    interval = statistics.bootstrap_players(by_player)
    rank_deltas = [float(delta[labels == rank].mean()) for rank in range(10)]
    seed_deltas = (seed_rewards[8] - seed_rewards[2]).mean(axis=1)
    counts = np.asarray([len(rows) for rows in questions])
    return {
        "scope": "exploratory original400-player validation; own branch only",
        "utility": {
            str(cap): float(values.mean()) for cap, values in rewards.items()
        },
        "paired_player_95ci": interval,
        "seed_deltas": seed_deltas.tolist(),
        "rank_deltas": rank_deltas,
        "game_count_deltas": {
            f"{low}-{high}": float(
                delta[(counts >= low) & (counts <= high)].mean()
            )
            for low, high in ((5, 9), (10, 14), (15, 20))
        },
        "screen_pass": bool(
            interval["delta"] >= 0.005
            and interval["low"] > 0
            and np.sum(seed_deltas > 0) >= 2
            and min(rank_deltas) >= -0.05
        ),
        "production_replaced": False,
    }


def main() -> None:
    """Writes one locked comparison report after all six runs complete."""
    directory = (
        pathlib.Path.cwd() / "production/game_coverage_pilot_20261109/training"
    )
    report = directory / "report.json"
    if report.exists():
        raise FileExistsError(f"Refusing existing report: {report}")
    predictions, metadata = load(directory)
    result = summarize(predictions, metadata)
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
