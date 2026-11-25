# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Runs matched legal candidate-policy adaptation with a fixed inner-only cost gate.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Never uses evaluation utility to continue, stop or select a checkpoint.
"""

import hashlib
import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_finetune import (
    train as input_train,
)
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.policy_adaptation import features
from competition2_rank_prediction.experiments.policy_adaptation import probe

DEVICE = "cuda:1"


def prepare(root: pathlib.Path) -> dict:
    """Reuses fixed player partitions only after complete frozen-score alignment."""
    source = root / "production/policy_frozen_scores_20261109"
    if not (source / "COMPLETE").exists():
        raise ValueError("Frozen score cache has not completed")
    for path, expected in json.loads(
        (source / "input_hashes.json").read_text()
    ).items():
        if pathlib.Path(path).suffix in (".npz", ".ckpt"):
            if (
                hashlib.sha256((root / path).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(
                    f"Frozen input/model provenance changed: {path}"
                )
    dataset = input_train.prepare(root)
    for split, name in (("train", "training"), ("evaluation", "evaluation")):
        payload = features.load(root, split)
        if not np.array_equal(payload["game_key"], dataset[name]["game_key"]):
            raise ValueError(
                "Legal-score data ordering differs from fixed pools"
            )
        payload["y"] = dataset[name]["y"]
        with np.load(source / f"{split}.npz") as cache:
            for field in ("game_key", "player_id", "rank"):
                if not np.array_equal(np.asarray(cache[field]), payload[field]):
                    raise ValueError(f"Frozen score alignment differs: {field}")
            if not np.array_equal(cache["inverse_ranks"], probe.INVERSE_RANKS):
                raise ValueError("Candidate profile order differs")
            payload["frozen_scores"] = np.asarray(cache["scores"])
        if payload["frozen_scores"].shape != (len(payload["y"]), 10):
            raise ValueError("Invalid frozen score dimensions")
        dataset[name] = payload
    optimizer = dataset["training"]["frozen_scores"][dataset["pools"].ravel()]
    dataset["mean"] = optimizer.mean(0).astype(np.float32)
    dataset["std"] = (optimizer.std(0) + 1e-6).astype(np.float32)
    return dataset


def configure(modules, payload, run: tuple, dataset: dict) -> dict:
    """Initializes identical heads before any optional backbone construction."""
    torch.manual_seed(run[0])
    head = torch.nn.Linear(10, 10).to(DEVICE)
    model = None
    names = []
    backbone = []
    if run[1]:
        model = bridge.build_model(modules, payload, DEVICE, False)
        for name, parameter in model.named_parameters():
            enabled = name.startswith(
                ("blocks.16.", "blocks.17.", "norm_trunkfinal.")
            )
            parameter.requires_grad_(enabled)
            if enabled:
                names.append(name)
                backbone.append(parameter)
    groups = [{"params": list(head.parameters()), "lr": 0.001}]
    if backbone:
        groups.append({"params": backbone, "lr": 0.00001})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    return {
        "model": model,
        "head": head,
        "optimizer": optimizer,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=2560
        ),
        "parameters": list(head.parameters()) + backbone,
        "adapt_names": names,
        "mean": torch.as_tensor(dataset["mean"], device=DEVICE),
        "std": torch.as_tensor(dataset["std"], device=DEVICE),
        "profiles": probe.candidate_metadata(modules, 1, DEVICE),
        "device": DEVICE,
    }


def logits(context: dict, payload: dict, rows: np.ndarray) -> torch.Tensor:
    """Reads all ten profile likelihoods, never a true-rank-selected profile."""
    if context["model"] is None:
        scores = torch.as_tensor(payload["frozen_scores"][rows], device=DEVICE)
    else:
        scores = features.game_scores(context, payload, rows)
    return context["head"]((scores - context["mean"]) / context["std"])


def update(context: dict, payload: dict, rows: np.ndarray) -> float:
    """Performs one rank-CE update while preserving backbone eval normalization."""
    context["optimizer"].zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(
        logits(context, payload, rows),
        torch.as_tensor(payload["y"][rows], device=DEVICE),
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite candidate-policy rank loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(context["parameters"], 1.0)
    context["optimizer"].step()
    context["scheduler"].step()
    return float(loss.detach().cpu())


def predict(context: dict, payload: dict, rows: np.ndarray) -> np.ndarray:
    """Predicts complete supplied rows without targets or updates."""
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            probabilities.append(
                logits(context, payload, rows[start : start + 4])
                .softmax(1)
                .cpu()
                .numpy()
            )
    return np.concatenate(probabilities)


def checkpoint(context: dict, dataset: dict, epoch: int, run: tuple) -> dict:
    """Stores head, changed backbone parameters, scale and player exclusions."""
    backbone = (
        {}
        if context["model"] is None
        else {
            name: value.detach().cpu().clone()
            for name, value in context["model"].state_dict().items()
            if name in context["adapt_names"]
        }
    )
    return {
        "head": {
            name: value.detach().cpu().clone()
            for name, value in context["head"].state_dict().items()
        },
        "backbone_partial": backbone,
        "mean": context["mean"].cpu(),
        "std": context["std"].cpu(),
        "epoch": epoch,
        "steps": epoch * 64,
        "seed": run[0],
        "adapt": run[1],
        "optimizer_players": dataset["optimizer_players"],
        "inner_players": dataset["inner_players"],
    }


def cost_gate(directory: pathlib.Path, history: list) -> bool:
    """Uses only seed0 inner epochs5/10 in both arms, never outer predictions."""
    frozen = json.loads((directory / "adapt0_seed0_history.json").read_text())
    baseline = min(
        row["inner_nll"] for row in frozen if row["epoch"] in (5, 10)
    )
    adapted = min(row["inner_nll"] for row in history if "inner_nll" in row)
    result = {
        "frozen_inner_nll": baseline,
        "adapted_inner_nll": adapted,
        "required_improvement": 0.01,
        "passes": adapted <= baseline - 0.01,
    }
    (directory / "inner_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result["passes"]


def fit(
    context: dict, dataset: dict, run: tuple, directory: pathlib.Path
) -> bool:
    """Continues the same optimizer/RNG only if the fixed seed0 cost gate passes."""
    generator = np.random.default_rng(20261109 + run[0])
    history = []
    best_nll = float("inf")
    selected = {}
    tag = f"adapt{int(run[1])}_seed{run[0]}"
    for epoch in range(1, 41):
        losses = []
        for _ in range(64):
            rows = dataset["pools"][
                generator.integers(400, size=4), generator.integers(8, size=4)
            ]
            losses.append(update(context, dataset["training"], rows))
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses))})
        if epoch % 5 == 0:
            probability = predict(
                context, dataset["training"], dataset["inner"]
            )
            targets = dataset["training"]["y"][dataset["inner"]]
            nll = float(
                -np.log(
                    probability[np.arange(len(targets)), targets].clip(1e-9)
                ).mean()
            )
            history[-1]["inner_nll"] = nll
            if nll < best_nll:
                best_nll = nll
                selected = checkpoint(context, dataset, epoch, run)
                torch.save(selected, directory / f"{tag}.pt")
            print(f"{tag} epoch={epoch} inner_nll={nll:.5f}", flush=True)
        (directory / f"{tag}_history.json").write_text(json.dumps(history))
        if (
            run == (0, True)
            and epoch == 10
            and not cost_gate(directory, history)
        ):
            return False
    context["head"].load_state_dict(selected["head"])
    if context["model"] is not None:
        context["model"].load_state_dict(
            selected["backbone_partial"], strict=False
        )
    return True


def main() -> None:
    """Runs only the predeclared staged pilot, with no automatic submission."""
    root = pathlib.Path.cwd()
    directory = root / "production/policy_adaptation_20261109"
    if directory.exists():
        raise FileExistsError(
            f"Refusing existing policy adaptation run: {directory}"
        )
    dataset = prepare(root)
    modules, payload = bridge.load_assets(
        root / "runtime/competition2_research"
    )
    directory.mkdir(parents=True)
    paths = list(pathlib.Path(__file__).parent.glob("*.py"))
    paths.extend(
        (root / "production/policy_frozen_scores_20261109").glob("*.npz")
    )
    paths.extend(
        (root / "production/human_finetune_inputs_20261109").rglob("*.npz")
    )
    paths.extend(
        (root / "production/policy_legal_inputs_20261109").rglob("*.npz")
    )
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
            context = configure(modules, payload, (seed, adapt), dataset)
            if not fit(context, dataset, (seed, adapt), directory):
                (directory / "STOPPED_AT_INNER_GATE").touch()
                return
            evaluation = dataset["evaluation"]
            probability = predict(
                context, evaluation, np.arange(len(evaluation["y"]))
            )
            np.savez_compressed(
                directory / f"adapt{int(adapt)}_seed{seed}_validation.npz",
                probabilities=probability,
                game_key=evaluation["game_key"],
                player=evaluation["player_id"],
                y=evaluation["y"],
            )
            del context
    (directory / "COMPLETE").touch()


if __name__ == "__main__":
    main()
