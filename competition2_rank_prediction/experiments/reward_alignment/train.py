# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Runs matched original-architecture NLL and reward-alignment experiments.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.reward_alignment.train
"""

import argparse
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.reward_alignment import objective


@dataclasses.dataclass
class Inputs:
    """Carries training-only normalization and the untouched outer cohort.

    Attributes:
        tokens: Standardized training-library tokens on GPU.
        masks: Valid training-library decision masks on GPU.
        labels: Training-library rank labels on GPU.
        players: Training-library player identifiers on CPU.
        training: Optimizer row indices.
        inner: Checkpoint-selection row indices.
        outer: Original validation inputs/metadata and training normalization.
    """

    tokens: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    players: np.ndarray
    training: np.ndarray
    inner: np.ndarray
    outer: dict[str, np.ndarray]


def arguments() -> argparse.Namespace:
    """Returns reproducible input, budget and isolated output arguments."""
    root = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research",
        type=pathlib.Path,
        default=root / "runtime/competition2_research",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "production/reward_alignment_20261108",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace, seed: int) -> Inputs:
    """Checks player separation and fits normalization on optimizer rows."""
    train = rank_lib.load_features(args.research / "features/feat96/train")
    outer = rank_lib.load_features(args.research / "features/feat96/val")
    if set(train["player_id"]) & set(outer["player_id"]):
        raise ValueError("Neural training and outer validation players overlap")
    training, inner = transformer_rank.inner_split(train["player_id"], seed)
    train, inner = supplement_inner(args.research, train, inner)
    raw = transformer_rank.decision_tokens(train["tokens"])
    mean, std = transformer_rank.standardise(
        raw[training], train["masks"][training]
    )
    outer["inputs"] = transformer_rank.decision_tokens(outer["tokens"])
    outer["inputs"] = np.where(
        outer["masks"][..., None], (outer["inputs"] - mean) / std, 0
    ).astype(np.float32)
    raw = np.where(train["masks"][..., None], (raw - mean) / std, 0)
    outer["mean"], outer["std"] = mean, std
    return Inputs(
        tokens=torch.from_numpy(raw.astype(np.float32)).to(args.device),
        masks=torch.from_numpy(train["masks"]).to(args.device),
        labels=torch.from_numpy(rank_lib.rank_ids(train["rank"])).to(
            args.device
        ),
        players=train["player_id"],
        training=training,
        inner=inner,
        outer=outer,
    )


def supplement_inner(
    root: pathlib.Path, train: dict[str, np.ndarray], inner: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Adds only held-out players' dense games, leaving optimizer rows intact."""
    dense = rank_lib.load_features(
        root / "experiments/dense_player_bags/features"
    )
    eligible = np.isin(dense["player_id"], train["player_id"][inner])
    eligible &= ~np.isin(dense["game_key"], train["game_key"])
    for player in np.unique(dense["player_id"][eligible]):
        original = np.unique(train["rank"][train["player_id"] == player])
        extra = np.unique(dense["rank"][dense["player_id"] == player])
        if not np.array_equal(original, extra):
            raise ValueError(f"Conflicting rank for inner player {player!r}")
    added = np.arange(len(train["rank"]), len(train["rank"]) + eligible.sum())
    merged = {
        key: np.concatenate((value, dense[key][eligible]))
        for key, value in train.items()
    }
    return merged, np.concatenate((inner, added))


def fixed_questions(players: np.ndarray) -> list[np.ndarray]:
    """Samples repeatable 5/10/15/20-game questions from dense inner players."""
    generator = np.random.default_rng(20261109)
    questions = []
    for player in np.unique(players):
        rows = np.flatnonzero(players == player)
        if len(rows) < 20:
            continue
        for index in range(20):
            count = (5, 10, 15, 20)[index % 4]
            questions.append(generator.choice(rows, count, replace=False))
    if not questions:
        raise ValueError(
            "No dense inner players available for question selection"
        )
    return questions


def inner_metrics(model: torch.nn.Module, data: Inputs) -> dict[str, float]:
    """Scores fixed questions only on inner players, never outer players."""
    probabilities = transformer_rank.predict(
        model,
        data.tokens[data.inner].cpu().numpy(),
        data.masks[data.inner].cpu().numpy(),
        data.tokens.device,
        batch=256,
    )
    labels = data.labels[data.inner].cpu().numpy()
    questions = fixed_questions(data.players[data.inner])
    pooled = np.stack(
        [rank_lib.pool(probabilities[rows]) for rows in questions]
    )
    truth = np.asarray([labels[rows[0]] for rows in questions])
    return {
        "nll": float(
            -np.log(probabilities[np.arange(len(labels)), labels]).mean()
        ),
        "utility": rank_lib.competition_score(rank_lib.decode(pooled), truth),
    }


def train_epoch(
    model: torch.nn.Module,
    data: Inputs,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.OneCycleLR,
    weight: float,
) -> float:
    """Uses identical shuffled batches and augmentations in matched arms."""
    model.train()
    rows = torch.from_numpy(data.training).to(data.tokens.device)
    rows = rows[torch.randperm(len(rows), device=rows.device)]
    losses = []
    for start in range(0, len(rows), 256):
        selected = rows[start : start + 256]
        masks = data.masks[selected].clone()
        masks &= torch.rand_like(masks, dtype=torch.float32) > 0.15
        masks[:, 0] |= ~masks.any(dim=1)
        tokens = data.tokens[selected]
        tokens = tokens + 0.05 * torch.randn_like(tokens)
        loss = objective.mixed_loss(
            model(tokens, masks), data.labels[selected], weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def export_predictions(
    directory: pathlib.Path, model: torch.nn.Module, data: Inputs
) -> None:
    """Exports each selected checkpoint on the untouched outer games."""
    for selection in ("nll", "utility"):
        payload = torch.load(
            directory / f"{selection}.pt",
            weights_only=False,
            map_location=data.tokens.device,
        )
        model.load_state_dict(payload["state"])
        probabilities = transformer_rank.predict(
            model,
            data.outer["inputs"],
            data.outer["masks"],
            data.tokens.device,
            batch=256,
        )
        np.savez_compressed(
            directory / f"{selection}_outer.npz",
            probabilities=probabilities,
            game_key=data.outer["game_key"],
            player=data.outer["player_id"],
            y=rank_lib.rank_ids(data.outer["rank"]),
        )


def train_arm(
    args: argparse.Namespace, data: Inputs, seed: int, weight: float
) -> None:
    """Trains one loss arm and saves both selection rules without extra fits."""
    torch.manual_seed(seed)
    model = transformer_rank.RankTransformer(
        data.tokens.shape[2], data.tokens.shape[1]
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=args.epochs * (-(-len(data.training) // 256)),
        pct_start=0.1,
    )
    directory = (
        args.output / f"seed_{seed}" / ("nll" if weight == 0 else "reward")
    )
    directory.mkdir(parents=True)
    history = []
    best = {"nll": float("inf"), "utility": float("-inf")}
    for epoch in range(args.epochs):
        loss = train_epoch(model, data, optimizer, schedule, weight)
        metrics = inner_metrics(model, data)
        history.append({"epoch": epoch + 1, "loss": loss, **metrics})
        for selection in ("nll", "utility"):
            improved = metrics[selection] < best[selection]
            if selection == "utility":
                improved = metrics[selection] > best[selection]
            if improved:
                best[selection] = metrics[selection]
                torch.save(
                    {
                        "state": model.state_dict(),
                        "mean": data.outer["mean"],
                        "std": data.outer["std"],
                        "epoch": epoch + 1,
                        "seed": seed,
                        "weight": weight,
                        "selection": selection,
                        "inner": metrics,
                    },
                    directory / f"{selection}.pt",
                )
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"seed={seed} {directory.name} epoch={epoch + 1}/{args.epochs} "
            f"NLL={metrics['nll']:.5f} utility={metrics['utility']:.5f}",
            flush=True,
        )
    export_predictions(directory, model, data)


def main() -> None:
    """Runs all preregistered seeds with immutable split and source records."""
    args = arguments()
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite experiment: {args.output}"
        )
    if args.epochs < 1 or args.seeds < 1:
        raise ValueError("Epoch and seed counts must be positive")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    manifest = {"epochs": args.epochs, "seeds": args.seeds, "weight": 0.2}
    manifest["source_sha256"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for seed in range(args.seeds):
        data = load_inputs(args, seed)
        np.savez_compressed(
            args.output / f"split_{seed}.npz",
            train_players=np.unique(data.players[data.training]),
            inner_players=np.unique(data.players[data.inner]),
            outer_players=np.unique(data.outer["player_id"]),
        )
        counts = np.unique(data.players[data.inner], return_counts=True)[1]
        print(
            f"seed={seed} inner players={len(counts)} <5games={(counts < 5).sum()}",
            flush=True,
        )
        for weight in (0.0, 0.2):
            train_arm(args, data, seed, weight)
    (args.output / "TRAINING_COMPLETE").write_text(
        "All matched arms finished.\n"
    )


if __name__ == "__main__":
    main()
