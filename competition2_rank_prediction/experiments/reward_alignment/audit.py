"""Audits completed matched training and renders its learning curves.

Example: python -m competition2_rank_prediction.experiments.reward_alignment.audit
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from competition2_rank_prediction.experiments.reward_alignment import train

matplotlib.use("Agg")


def check_split(args, seed: int) -> None:
    """Rejects any intersection between optimizer, inner and outer players."""
    with np.load(args.output / f"split_{seed}.npz") as split:
        groups = [set(split[key]) for key in split.files]
        for left, group in enumerate(groups):
            for right in range(left):
                if group & groups[right]:
                    raise ValueError(f"Player split overlap for seed {seed}")


def main() -> None:
    """Checks split separation, selected epochs, budgets and training artifacts."""
    args = train.arguments()
    records = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed in range(args.seeds):
        check_split(args, seed)
        for loss in ("nll", "reward"):
            directory = args.output / f"seed_{seed}" / loss
            history = json.loads((directory / "history.json").read_text())
            if len(history) != args.epochs:
                raise ValueError(f"Incomplete training budget: {directory}")
            for selection in ("nll", "utility"):
                values = [row[selection] for row in history]
                optimum = (
                    np.argmin(values)
                    if selection == "nll"
                    else np.argmax(values)
                )
                payload = torch.load(
                    directory / f"{selection}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if payload["epoch"] != int(optimum) + 1:
                    raise ValueError(
                        f"Incorrect checkpoint selection: {directory}"
                    )
                records.append(
                    {
                        "seed": seed,
                        "loss": loss,
                        "selection": selection,
                        "selected_epoch": payload["epoch"],
                        "weight": payload["weight"],
                    }
                )
            for axis, metric in zip(axes, ("nll", "utility")):
                axis.plot(
                    [row["epoch"] for row in history],
                    [row[metric] for row in history],
                    label=f"{loss} seed{seed}",
                    alpha=0.8,
                )
                axis.set_title(f"Inner {metric} (not outer performance)")
                axis.set_xlabel("Epoch")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "learning_curves.png", dpi=160)
    plt.close(figure)
    (args.output / "audit.json").write_text(json.dumps(records, indent=2))
    print(
        "PASS: player separation, equal budgets, checkpoint selection",
        flush=True,
    )


if __name__ == "__main__":
    main()
