"""Reports the preregistered paired-player question-supervision comparison.

Example: python -m competition2_rank_prediction.experiments.question_supervision.evaluate
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from competition2_rank_prediction.experiments.question_supervision import run
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

matplotlib.use("Agg")


def load_results(
    directory,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Checks shared evaluation players and inner-only checkpoint selection."""
    predictions: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    metadata: dict[str, np.ndarray] = {}
    figure, axis = plt.subplots(figsize=(8, 4))
    for seed in range(3):
        for arm in ("A", "B"):
            parent = directory / f"seed_{seed}"
            with np.load(parent / f"{arm}_evaluation.npz") as payload:
                for name in ("labels", "players", "counts"):
                    if metadata and name in metadata:
                        if not np.array_equal(metadata[name], payload[name]):
                            raise ValueError(
                                f"Evaluation metadata changed: {name}"
                            )
                    metadata[name] = np.asarray(payload[name])
                predictions[arm].append(np.asarray(payload["probabilities"]))
            history = json.loads((parent / f"{arm}_history.json").read_text())
            checkpoint = torch.load(
                parent / f"{arm}.pt", weights_only=False, map_location="cpu"
            )
            if len(history) != run.EPOCHS:
                raise ValueError("Incomplete training budget")
            if (
                checkpoint["epoch"]
                != int(np.argmax([row["utility"] for row in history])) + 1
            ):
                raise ValueError(
                    "Checkpoint not selected by earliest inner utility maximum"
                )
            axis.plot(
                [row["epoch"] for row in history],
                [row["utility"] for row in history],
                label=f"{arm} seed{seed}",
            )
    axis.set_title("Inner question utility, not evaluation utility")
    axis.set_xlabel("Epoch")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(figure)
    return {
        arm: np.stack(values) for arm, values in predictions.items()
    }, metadata


def summarize(
    predictions: dict[str, np.ndarray], metadata: dict[str, np.ndarray]
) -> dict:
    """Bootstraps paired players after averaging training seeds within player."""
    rewards = {}
    for arm in ("A", "B"):
        rewards[arm] = np.stack(
            [
                statistics.player_scores(probabilities, metadata["labels"])
                for probabilities in predictions[arm]
            ]
        )
    per_player = (rewards["B"] - rewards["A"]).mean(axis=0)
    interval = statistics.bootstrap_players(per_player)
    rank_deltas = [
        float(per_player[metadata["labels"] == rank].mean())
        for rank in range(10)
    ]
    seed_deltas = (rewards["B"] - rewards["A"]).mean(axis=1)
    return {
        "scope": "exploratory, previously studied 300 players; own branch only",
        "mean_seed_utility": {
            arm: float(values.mean()) for arm, values in rewards.items()
        },
        "seed_deltas": seed_deltas.tolist(),
        "paired_player_95ci": interval,
        "rank_deltas": rank_deltas,
        "game_count_deltas": {
            f"{low}-{high}": float(
                per_player[
                    (metadata["counts"] >= low) & (metadata["counts"] <= high)
                ].mean()
            )
            for low, high in ((5, 9), (10, 14), (15, 20))
        },
        "mean_seed_nll": {
            arm: float(
                -np.log(
                    values[
                        :,
                        np.arange(len(metadata["labels"])),
                        metadata["labels"],
                    ].clip(1e-9)
                ).mean()
            )
            for arm, values in predictions.items()
        },
        "screen_pass": interval["delta"] >= 0.005
        and interval["low"] > 0
        and np.count_nonzero(seed_deltas > 0) >= 2
        and min(rank_deltas) >= -0.05,
    }


def main() -> None:
    """Evaluates completed artifacts only, without changing model choices."""
    args = common.parse_arguments(
        __doc__ or "", "question_supervision_20261109"
    )
    if not (args.output / "TRAINING_COMPLETE").exists():
        raise RuntimeError("All preregistered fits must finish first")
    if (args.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite completed report")
    predictions, metadata = load_results(args.output)
    report = summarize(predictions, metadata)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
