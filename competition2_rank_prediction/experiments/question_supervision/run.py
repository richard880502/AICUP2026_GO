"""Runs matched per-game versus end-to-end question-supervised training.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.question_supervision.run
"""

import hashlib
import json
import pathlib

import numpy as np
import torch

import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.question_supervision import data
from competition2_rank_prediction.experiments.question_supervision import losses
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import train

EPOCHS = 40
STEPS = 64


def optimize(
    model: torch.nn.Module,
    context: train.TrainingContext,
    optimizer: torch.optim.Optimizer,
    bags: list[np.ndarray],
    arm: str,
) -> float:
    """Backpropagates one matched bag batch through the original encoder."""
    model.train()
    rows = np.concatenate(bags)
    masks = context.masks[rows].clone()
    masks &= torch.rand_like(masks, dtype=torch.float32) > 0.15
    masks[:, 0] |= ~masks.any(dim=1)
    tokens = context.tokens[rows]
    tokens = tokens + 0.05 * torch.randn_like(tokens)
    game, question = losses.question_losses(
        model(tokens, masks),
        context.labels[rows],
        tuple(len(bag) for bag in bags),
    )
    loss = game if arm == "A" else 0.5 * (game + question)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    return float(loss.item())


def train_arm(context: train.TrainingContext, seed: int, arm: str) -> None:
    """Selects each arm on identical fixed inner-player question utility."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = transformer_rank.RankTransformer(
        context.tokens.shape[2], context.tokens.shape[1]
    ).to(context.tokens.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * STEPS
    )
    history = []
    best = float("-inf")
    checkpoint = context.output / f"{arm}.pt"
    for epoch in range(EPOCHS):
        values = []
        for _ in range(STEPS):
            bags = data.sample_bags(context.data, generator)
            values.append(optimize(model, context, optimizer, bags, arm))
            schedule.step()
        metrics = train.inner_metrics(model, context)
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(values)), **metrics}
        )
        if metrics["utility"] > best:
            best = metrics["utility"]
            torch.save(
                {
                    "state": model.state_dict(),
                    "seed": seed,
                    "arm": arm,
                    "epoch": epoch + 1,
                    "normalization": context.data.provenance.normalization,
                },
                checkpoint,
            )
        (context.output / f"{arm}_history.json").write_text(
            json.dumps(history, indent=2)
        )
        print(
            f"seed={seed} arm={arm} epoch={epoch + 1} inner={best:.6f}",
            flush=True,
        )
    export_evaluation(context, model, arm)


def export_evaluation(
    context: train.TrainingContext, model: torch.nn.Module, arm: str
) -> None:
    """Exports predictions once from the inner-selected checkpoint."""
    model.load_state_dict(
        torch.load(
            context.output / f"{arm}.pt",
            map_location=context.tokens.device,
            weights_only=False,
        )["state"]
    )
    questions = context.data.questions["evaluation"]
    probabilities = train.question_probabilities(model, context, questions)
    np.savez_compressed(
        context.output / f"{arm}_evaluation.npz",
        probabilities=probabilities,
        labels=np.asarray([context.data.labels[rows[0]] for rows in questions]),
        players=context.data.splits["evaluation"],
        counts=np.asarray([len(rows) for rows in questions]),
    )


def main() -> None:
    """Audits input data and executes all predeclared matched arms."""
    args = common.parse_arguments(
        __doc__ or "", "question_supervision_20261109", True
    )
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite experiment: {args.output}"
        )
    torch.set_num_threads(4)
    inputs = data.load_dense(args.research)
    args.output.mkdir(parents=True)
    sources = pathlib.Path(__file__).resolve().parent
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources.glob("*.py")
    }
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "epochs": EPOCHS,
                "steps": STEPS,
                "seeds": [0, 1, 2],
                "sources_sha256": hashes,
                "device": args.device,
            },
            indent=2,
        )
    )
    (args.output / "data_audit.json").write_text(
        json.dumps(inputs.provenance.audit, indent=2)
    )
    np.savez_compressed(
        args.output / "splits.npz", allow_pickle=False, **inputs.splits
    )
    context = train.TrainingContext(
        inputs,
        torch.from_numpy(inputs.tokens).to(args.device),
        torch.from_numpy(inputs.masks).to(args.device),
        torch.from_numpy(inputs.labels).to(args.device),
        args.output,
    )
    for seed in range(3):
        context.output = args.output / f"seed_{seed}"
        context.output.mkdir()
        for arm in ("A", "B"):
            train_arm(context, seed, arm)
    (args.output / "TRAINING_COMPLETE").write_text("All six fits finished.\n")


if __name__ == "__main__":
    main()
