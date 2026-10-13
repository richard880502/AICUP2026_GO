"""Trains a 5-20 game player model on frozen dense game embeddings.

The model is selected on a 300-player inner split and reported on the fixed
400-player external validation cohort. Holdout embeddings are not read here.

Example:
    c1/.venv/bin/python -m experiments.dense_player_bags.train_bags \
      --device cuda:1
"""

import argparse
import json
import pathlib

import numpy as np
import torch

import rank_lib
from experiments.hierarchical_move import train as bag_train


def load_split(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Loads one dense embedding split."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def player_data(split: dict[str, np.ndarray]) -> bag_train.PlayerData:
    """Builds player sampling tables from one embedding split."""
    players: np.ndarray = split["player_id"].astype(str)
    ranks: np.ndarray = rank_lib.rank_ids(split["rank"].astype(str))
    return bag_train.make_player_data(players, ranks, set(), 20)


def main() -> None:
    """Trains the frozen-encoder dense player-bag model."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=experiment / "embeddings",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=experiment / "results",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--architecture",
        choices=("transformer", "deepsets"),
        default="transformer",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--inner-players-per-rank", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    training_split = load_split(args.embeddings / "train.npz")
    validation_split = load_split(args.embeddings / "validation.npz")
    all_training = player_data(training_split)
    training, inner_validation = bag_train.split_inner_players(
        all_training,
        args.inner_players_per_rank,
        args.seed,
    )
    external_validation = player_data(validation_split)
    embeddings: np.ndarray = training_split["embeddings"].astype(np.float32)
    print(
        f"players train={len(training.rows)} inner={len(inner_validation.rows)} "
        f"external={len(external_validation.rows)}",
        flush=True,
    )
    model, history = bag_train.train_model(
        embeddings,
        training,
        inner_validation,
        args,
        device,
    )
    external_report = bag_train.evaluate(
        model,
        validation_split["embeddings"].astype(np.float32),
        external_validation,
        device,
        10,
        args.seed + 10000,
    )
    torch.save(
        {
            "state": model.state_dict(),
            "args": vars(args) | {
                "embeddings": str(args.embeddings),
                "output": str(args.output),
            },
        },
        args.output / "model.pt",
    )
    report = {
        "protocol": {
            "training_players": len(training.rows),
            "inner_validation_players": len(inner_validation.rows),
            "external_validation_players": len(external_validation.rows),
            "games_per_player": 20,
            "holdout_read": False,
            "encoder_note": (
                "frozen d96_h encoders saw two games from selected training "
                "players; external validation and holdout remain unseen"
            ),
        },
        "history": history,
        "external_validation": external_report,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(external_report, indent=2))


if __name__ == "__main__":
    main()
