# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Runs the preregistered frozen versus last-two-block HumanSL rank pilot.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Only inner players select checkpoints; official test data is never loaded.
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_data,
)
from competition2_rank_prediction.experiments.human_finetune import data
from competition2_rank_prediction.experiments.human_trunk import bridge

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error

DEVICE = "cuda:1"


def prepare(root: pathlib.Path) -> dict:
    """Loads completed caches and audits all optimizer/inner/evaluation IDs."""
    source = root / "production/human_finetune_inputs_20261109"
    if not (source / "COMPLETE").exists():
        raise ValueError("Input cache is incomplete")
    training = data.load(source / "train")
    evaluation = data.load(source / "evaluation")
    split = coverage_data.partitions(
        coverage_data.read_manifest(
            root / "production/game_coverage_manifest_20261109/cap8"
        )
    )
    if set(training["player_id"]) != split["optimizer"] | split["inner"]:
        raise ValueError("Training input players differ from fixed manifest")
    if set(training["player_id"]) & set(evaluation["player_id"]):
        raise ValueError("Evaluation players overlap neural training/selection")
    if len(training["game_key"]) != 4000 or len(evaluation["game_key"]) != 8000:
        raise ValueError("Incorrect cache game counts")
    for payload in (training, evaluation):
        payload["y"] = rank_lib.rank_ids(payload["rank"])
    pools = np.asarray(
        [
            np.flatnonzero(training["player_id"] == player)
            for player in sorted(split["optimizer"])
        ]
    )
    if pools.shape != (400, 8):
        raise ValueError("Optimizer pools must have eight games per player")
    return {
        "training": training,
        "evaluation": evaluation,
        "pools": pools,
        "inner": np.flatnonzero(
            np.isin(training["player_id"], list(split["inner"]))
        ),
        "optimizer_players": sorted(split["optimizer"]),
        "inner_players": sorted(split["inner"]),
    }


def configure(modules, payload, run: tuple[int, bool]) -> dict:
    """Builds identical heads and optional adaptation with frozen running stats."""
    seed, adapt = run
    torch.manual_seed(seed)
    model = bridge.build_model(modules, payload, DEVICE, False)
    names = []
    backbone = []
    for name, parameter in model.named_parameters():
        enabled = adapt and name.startswith(
            ("blocks.16.", "blocks.17.", "norm_trunkfinal.")
        )
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
            backbone.append(parameter)
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(768), torch.nn.Linear(768, 10)
    ).to(DEVICE)
    groups = [{"params": list(head.parameters()), "lr": 0.001}]
    if adapt:
        groups.append({"params": backbone, "lr": 0.00001})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    return {
        "model": model,
        "head": head,
        "optimizer": optimizer,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40 * 64
        ),
        "parameters": list(head.parameters()) + backbone,
        "adapt_names": names,
    }


def step(context: dict, payload: dict, rows: np.ndarray) -> float:
    """Backpropagates real rank loss through the chosen trainable parameters."""
    context["optimizer"].zero_grad(set_to_none=True)
    features = data.representation(
        context["model"], data.batch(payload, rows, DEVICE)
    )
    loss = torch.nn.functional.cross_entropy(
        context["head"](features),
        torch.as_tensor(payload["y"][rows], device=DEVICE),
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite rank loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(context["parameters"], 1.0)
    context["optimizer"].step()
    context["scheduler"].step()
    return float(loss.detach().cpu())


def predict(context: dict, payload: dict, rows: np.ndarray) -> np.ndarray:
    """Predicts selected rows without model or normalization updates."""
    context["head"].eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            features = data.representation(
                context["model"],
                data.batch(payload, rows[start : start + 16], DEVICE),
            )
            predictions.append(
                context["head"](features).softmax(1).cpu().numpy()
            )
    return np.concatenate(predictions)


def selected_state(
    context: dict, epoch: int, dataset: dict, run: tuple
) -> dict:
    """Stores only adapted backbone parameters plus the full classification head."""
    return {
        "backbone_partial": {
            name: value.detach().cpu().clone()
            for name, value in context["model"].state_dict().items()
            if name in context["adapt_names"]
        },
        "head": {
            name: value.detach().cpu().clone()
            for name, value in context["head"].state_dict().items()
        },
        "epoch": epoch,
        "seed": run[0],
        "adapt": run[1],
        "optimizer_players": dataset["optimizer_players"],
        "inner_players": dataset["inner_players"],
    }


def fit(
    context: dict, dataset: dict, run: tuple, directory: pathlib.Path
) -> dict:
    """Selects solely among the eight fixed inner-NLL checkpoint epochs."""
    generator = np.random.default_rng(20261109 + run[0])
    history = []
    best_nll = float("inf")
    best = {}
    tag = f"adapt{int(run[1])}_seed{run[0]}"
    for epoch in range(1, 41):
        context["head"].train()
        losses = []
        for _ in range(64):
            rows = dataset["pools"][
                generator.integers(400, size=4), generator.integers(8, size=4)
            ]
            losses.append(
                step(
                    context,
                    dataset["training"],
                    rows,
                )
            )
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses))})
        if epoch % 5 == 0:
            probability = predict(
                context, dataset["training"], dataset["inner"]
            )
            labels = dataset["training"]["y"][dataset["inner"]]
            nll = float(
                -np.log(
                    probability[np.arange(len(labels)), labels].clip(1e-9)
                ).mean()
            )
            history[-1]["inner_nll"] = nll
            if nll < best_nll:
                best_nll = nll
                best = selected_state(context, epoch, dataset, run)
                torch.save(best, directory / f"{tag}.pt")
            print(f"{tag} epoch={epoch} inner_nll={nll:.5f}", flush=True)
        (directory / f"{tag}_history.json").write_text(json.dumps(history))
    context["model"].load_state_dict(best["backbone_partial"], strict=False)
    context["head"].load_state_dict(best["head"])
    return best


def main() -> None:
    """Runs all six matched fits in an isolated, non-overwriting output directory."""
    root = pathlib.Path.cwd()
    directory = root / "production/human_finetune_20261109"
    if directory.exists():
        raise FileExistsError(f"Refusing existing training: {directory}")
    dataset = prepare(root)
    modules, payload = bridge.load_assets(
        root / "runtime/competition2_research"
    )
    directory.mkdir(parents=True)
    paths = list(
        (root / "production/human_finetune_inputs_20261109").rglob("*.npz")
    )
    paths.extend(pathlib.Path(__file__).parent.glob("*.py"))
    paths.append(
        root
        / "runtime/competition2_research/models/human_trunk/b18c384nbt-humanv0.ckpt"
    )
    (directory / "input_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in paths
            },
            indent=2,
        )
    )
    for seed in range(3):
        for adapt in (False, True):
            context = configure(modules, payload, (seed, adapt))
            fit(context, dataset, (seed, adapt), directory)
            evaluation = dataset["evaluation"]
            probabilities = predict(
                context, evaluation, np.arange(len(evaluation["y"]))
            )
            np.savez_compressed(
                directory / f"adapt{int(adapt)}_seed{seed}_validation.npz",
                probabilities=probabilities,
                game_key=evaluation["game_key"],
                player=evaluation["player_id"],
                y=evaluation["y"],
            )
            del context


if __name__ == "__main__":
    main()
