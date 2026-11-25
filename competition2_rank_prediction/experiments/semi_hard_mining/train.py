"""Runs the fixed A/B ordinal pair-loss experiment on GPU.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.semi_hard_mining.train
"""

import dataclasses
import json
import pathlib
import time

import numpy as np
import sklearn.metrics  # type: ignore[import-untyped]
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import losses
from competition2_rank_prediction.experiments.semi_hard_mining import pair_miner
from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)

SEED = 20261108


@dataclasses.dataclass(frozen=True)
class TrainingRecipe:
    """Fixes the seed and training budget for one replication.

    Attributes:
        seed: Initialization and sampling seed.
        epochs: Number of complete training epochs.
        steps: Optimizer updates in each epoch.
    """

    seed: int = SEED
    epochs: int = 15
    steps: int = 64


@dataclasses.dataclass
class EpochSettings:
    """Carries optimizer and sampling state for one training epoch.

    Attributes:
        optimizer: AdamW optimizer.
        schedule: Per-update cosine learning-rate schedule.
        generator: Player-balanced NLL sampling RNG.
        pair_generator: Independent auxiliary bag sampling RNG.
        use_pairs: Whether this epoch uses auxiliary supervision.
        steps: Number of optimizer updates.
        miner: Training-only semi-hard sampler for arm C, when enabled.
    """

    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.CosineAnnealingLR
    generator: np.random.Generator
    pair_generator: np.random.Generator
    use_pairs: bool
    steps: int
    miner: pair_miner.SemiHardMiner | None = None


@dataclasses.dataclass
class TrainingContext:
    """Holds immutable data and device tensors shared by experimental arms.

    Attributes:
        data: Player-disjoint decision features and questions.
        tokens: All standardized game tensors on device.
        masks: All valid token masks on device.
        labels: All game labels on device.
        output: Isolated experimental artifact directory.
    """

    data: player_sampler.PlayerData
    tokens: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    output: pathlib.Path


def question_probabilities(
    model: torch.nn.Module,
    context: TrainingContext,
    questions: list[np.ndarray],
) -> np.ndarray:
    """Pools probabilities of fixed questions without changing the model."""
    model.eval()
    rows = np.unique(np.concatenate(questions))
    probabilities = np.zeros((len(context.labels), 10), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            indices = rows[start : start + 256]
            probabilities[indices] = (
                model(
                    context.tokens[indices],
                    context.masks[indices],
                )
                .exp()
                .cpu()
                .numpy()
            )
    return np.stack(
        [rank_lib.pool(probabilities[indices]) for indices in questions]
    )


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Summarizes all-rank utility and conditional 1D/2D discrimination."""
    decisions = rank_lib.decode(probabilities)
    errors = np.abs(decisions - labels)
    selected = np.isin(labels, (4, 5))
    odds = probabilities[selected, 5] / probabilities[selected, 4:6].sum(axis=1)
    return {
        "utility": rank_lib.competition_score(decisions, labels),
        "exact": float(np.mean(errors == 0)),
        "adjacent": float(np.mean(errors == 1)),
        "large_error": float(np.mean(errors >= 2)),
        "nll": float(
            -np.log(
                probabilities[np.arange(len(labels)), labels].clip(1e-9)
            ).mean()
        ),
        "pair_auc": float(
            sklearn.metrics.roc_auc_score(labels[selected] == 5, odds)
        ),
        "pair_accuracy": float(
            np.mean((odds >= 0.5) == (labels[selected] == 5))
        ),
    }


def _nll_rows(
    data: player_sampler.PlayerData, generator: np.random.Generator
) -> np.ndarray:
    players = generator.choice(data.splits["train"], 128)
    return np.asarray(
        [generator.choice(data.rows[str(player)]) for player in players]
    )


def _pair_forward(
    model: torch.nn.Module,
    context: TrainingContext,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    bags: list[np.ndarray] = []
    for low, high in pairs:
        bags.extend((low, high))
    rows = np.concatenate(bags)
    embeddings = model.encode(context.tokens[rows], context.masks[rows])
    strengths = model.score(embeddings).squeeze(-1)
    pooled = []
    start = 0
    for bag in bags:
        pooled.append(strengths[start : start + len(bag)].mean())
        start += len(bag)
    values = torch.stack(pooled)
    return losses.pairwise_loss(values[::2], values[1::2])


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    context: TrainingContext,
    rows: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    model.train()
    valid = context.masks[rows].clone()
    valid &= torch.rand_like(valid, dtype=torch.float32) > 0.15
    valid[:, 0] |= ~valid.any(dim=1)
    tokens = context.tokens[rows] + 0.05 * torch.randn_like(
        context.tokens[rows]
    )
    optimizer.zero_grad(set_to_none=True)
    nll = torch.nn.functional.nll_loss(
        model(tokens, valid), context.labels[rows]
    )
    pair_loss = (
        _pair_forward(model, context, pairs) if pairs else nll.new_zeros(())
    )
    losses.combined_loss(nll, pair_loss, 0.05 if pairs else 0).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(nll.detach()), float(pair_loss.detach()), float(norm)


def train_arm(
    context: TrainingContext,
    arm: str,
    recipe: TrainingRecipe | None = None,
) -> None:
    """Trains one fixed-budget arm and selects epochs using inner NLL."""
    if (context.output / f"{arm}.pt").exists():
        raise FileExistsError(f"Refusing to overwrite arm {arm}")
    recipe = recipe or TrainingRecipe()
    if recipe.epochs < 1 or recipe.steps < 1 or arm not in ("A", "B", "C"):
        raise ValueError("Invalid training budget or experimental arm")
    torch.manual_seed(recipe.seed)
    model = transformer_rank.RankTransformer(
        context.tokens.shape[2],
        context.tokens.shape[1],
        128,
        4,
        384,
        3,
        0.1,
        "clm",
    ).to(context.tokens.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=recipe.epochs * recipe.steps
    )
    settings = EpochSettings(
        optimizer=optimizer,
        schedule=schedule,
        generator=np.random.default_rng(recipe.seed),
        pair_generator=np.random.default_rng(recipe.seed + 2),
        use_pairs=False,
        steps=recipe.steps,
        miner=pair_miner.SemiHardMiner(context.data) if arm == "C" else None,
    )
    history = []
    best = float("inf")
    started = time.monotonic()
    for epoch in range(recipe.epochs):
        settings.use_pairs = arm in ("B", "C") and epoch >= 5
        if settings.miner is not None and epoch >= 5 and (epoch - 5) % 2 == 0:
            settings.miner.refresh(model, context.tokens, context.masks)
        values = _train_epoch(
            model,
            context,
            settings,
        )
        report = inner_metrics(model, context)
        history.append(
            {
                "epoch": epoch + 1,
                "inner": report,
                "train_nll_pair_gradnorm": np.mean(values, axis=0).tolist(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if settings.miner is not None:
            history[-1]["mining"] = dict(settings.miner.counts)
        if report["nll"] < best:
            best = report["nll"]
            torch.save(
                {
                    "state": model.state_dict(),
                    "epoch": epoch + 1,
                    "normalization": context.data.provenance.normalization,
                    "seed": recipe.seed,
                    "arm": arm,
                    "recipe": dataclasses.asdict(recipe),
                },
                context.output / f"{arm}.pt",
            )
        print(json.dumps({"arm": arm, **history[-1]}), flush=True)
        (context.output / f"{arm}_history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )


def inner_metrics(
    model: torch.nn.Module, context: TrainingContext
) -> dict[str, float]:
    """Reports the prelocked inner questions for epoch selection."""
    questions = context.data.questions["inner"]
    truth = np.asarray([context.data.labels[rows[0]] for rows in questions])
    return metrics(question_probabilities(model, context, questions), truth)


def _train_epoch(
    model: torch.nn.Module,
    context: TrainingContext,
    settings: EpochSettings,
) -> list[tuple[float, float, float]]:
    values = []
    for _ in range(settings.steps):
        rows = _nll_rows(context.data, settings.generator)
        if settings.use_pairs and settings.miner is not None:
            pairs = settings.miner.sample_pairs(settings.pair_generator)
        else:
            pairs = player_sampler.sample_pairs(
                context.data,
                settings.pair_generator,
            )
        if not settings.use_pairs:
            pairs = []
        values.append(
            _train_step(model, settings.optimizer, context, rows, pairs)
        )
        settings.schedule.step()
    return values


def main() -> None:
    """Audits data, saves partitions and runs matched-budget A/B arms."""
    arguments = common.parse_arguments(
        __doc__ or "", "semi_hard_mining_20261108", True
    )
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    torch.set_num_threads(4)
    data = player_sampler.load_data(arguments.research)
    arguments.output.mkdir(parents=True)
    (arguments.output / "data_audit.json").write_text(
        json.dumps(data.provenance.audit, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        arguments.output / "splits.npz",
        train=data.splits["train"],
        inner=data.splits["inner"],
        evaluation=data.splits["evaluation"],
    )
    device = torch.device(arguments.device)
    context = TrainingContext(
        data,
        torch.from_numpy(data.tokens).to(device),
        torch.from_numpy(data.masks).to(device),
        torch.from_numpy(data.labels).to(device),
        arguments.output,
    )
    print(json.dumps(data.provenance.audit), flush=True)
    for arm in ("A", "B"):
        train_arm(context, arm)


if __name__ == "__main__":
    main()
