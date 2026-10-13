"""Augments frozen dense game embeddings with per-seed ordinal logits.

Each 384D dense embedding is three concatenated 128D d96 encoder outputs.
This script applies the corresponding frozen ordinal heads to append thirty
per-game log-probability values, letting a player-bag model see both style
representations and calibrated rank evidence without rerunning KataGo.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.attach_game_logits --device cpu
"""

import argparse
import dataclasses
import pathlib

import numpy as np
import torch

from experiments.dense_player_bags import extract_game_embeddings

SPLITS = ("train", "validation", "holdout")


@dataclasses.dataclass(frozen=True)
class AugmentContext:
    """Contains frozen heads and input/output locations for every split."""

    source: pathlib.Path
    output: pathlib.Path
    models: list[tuple[torch.nn.Module, dict[str, np.ndarray]]]
    device: torch.device
    batch_size: int


def classify_embeddings(
    embeddings: np.ndarray,
    models: list[
        tuple[torch.nn.Module, dict[str, np.ndarray]]
    ],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Returns concatenated per-seed log probabilities for cached embeddings."""
    expected_dimensions = 128 * len(models)
    if embeddings.shape[1] != expected_dimensions:
        raise ValueError(
            f"Expected {expected_dimensions} embedding dimensions, found "
            f"{embeddings.shape[1]}",
        )
    output = []
    with torch.no_grad(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for model_index, (model, _) in enumerate(models):
            start = model_index * 128
            end = start + 128
            parts = []
            for row in range(0, len(embeddings), batch_size):
                game = torch.from_numpy(
                    embeddings[row:row + batch_size, start:end].astype(
                        np.float32,
                    ),
                ).to(device)
                parts.append(model.classify(game).cpu().numpy())
            output.append(np.concatenate(parts, axis=0))
    return np.concatenate(output, axis=1)


def augment_split(
    name: str,
    context: AugmentContext,
) -> None:
    """Writes one embedding split with its thirty additional logit features."""
    with np.load(context.source / f"{name}.npz", allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    embeddings = payload["embeddings"].astype(np.float32)
    logits = classify_embeddings(
        embeddings,
        context.models,
        context.device,
        context.batch_size,
    )
    payload["embeddings"] = np.concatenate((embeddings, logits), axis=1).astype(
        np.float16,
    )
    np.savez_compressed(context.output / f"{name}.npz", **payload)
    print(f"saved {name}: {payload['embeddings'].shape}", flush=True)


def main() -> None:
    """Builds 414D style-plus-rank-evidence dense embeddings."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=experiment / "embeddings",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=experiment / "embeddings_logits",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "results",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_paths = sorted(args.results.glob("d96_h_seed*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError("No d96_h seed checkpoints were found")
    checkpoint_path = checkpoint_paths[0]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    models = extract_game_embeddings.load_models(
        args.results,
        len(checkpoint["mean"]),
        96,
        device,
    )
    context = AugmentContext(
        args.source,
        args.output,
        models,
        device,
        args.batch,
    )
    for name in SPLITS:
        augment_split(name, context)


if __name__ == "__main__":
    main()
