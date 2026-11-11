"""Fits candidate-only ArcFace encoders for official retrieval pools.

The production runner uses frozen original epoch settings for 100-game pools
and candidate-only checkpoint selection for the new ten-game pools.
"""

import copy
import dataclasses
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification import train_profile_metric
from competition1_player_identification.production_pipeline import data


@dataclasses.dataclass
class FitSpec:
    """Stores one deterministic encoder fit configuration."""

    dimension: int
    seed: int
    epochs: int
    player_count: int
    device: torch.device


@dataclasses.dataclass
class Evaluation:
    """Stores candidate-only reference and held-out query tensors."""

    references: torch.Tensor
    classes: torch.Tensor
    queries: torch.Tensor
    counts: tuple[int, ...]


@dataclasses.dataclass
class Context:
    """Stores one pool and its candidate-only partition."""

    pool: data.Pool
    output: pathlib.Path
    device: torch.device
    reference_positions: np.ndarray
    holdout_positions: np.ndarray
    counts: tuple[int, ...]


@dataclasses.dataclass
class Branch:
    """Stores three-seed embeddings and checkpoint-selection records."""

    development: np.ndarray
    final: np.ndarray
    queries: np.ndarray
    epochs: list[int]


def standardized(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Applies candidate-only centering and variance normalization."""
    return np.nan_to_num(
        (values - reference.mean(axis=0))
        / reference.std(axis=0).clip(min=1e-6),
    ).astype(np.float32)


def encode(model: torch.nn.Module, profiles: np.ndarray) -> np.ndarray:
    """Encodes profiles in bounded batches on the model's current device."""
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(profiles), 1024):
            predictions.append(model(torch.from_numpy(
                profiles[start:start + 1024],
            ).to(device)).cpu().numpy())
    return np.concatenate(predictions)


def evaluation_score(model: torch.nn.Module, evaluation: Evaluation) -> float:
    """Measures exponential Top-5 score using candidate-held-out games only."""
    model.eval()
    with torch.no_grad():
        centroids = train_profile_metric.make_centroids(
            model(evaluation.references),
            evaluation.classes,
            evaluation.queries.shape[0],
        )
        queries = model(evaluation.queries.flatten(0, 1)).reshape(
            *evaluation.queries.shape[:2], -1,
        )
        scores = []
        for count in evaluation.counts:
            cosine = torch.nn.functional.normalize(
                queries[:, :count].mean(dim=1), dim=1,
            ) @ centroids.T
            scores.append(train_profile_metric.make_metrics(
                cosine.cpu().numpy(),
            )["top5_exponential_score"])
    return float(np.mean(scores))


def fit_model(
    inputs: tuple[torch.Tensor, torch.Tensor],
    spec: FitSpec,
    evaluation: Evaluation | None = None,
) -> tuple[train_profile_metric.ArcFaceModel, int]:
    """Fits a deterministic ArcFace model with optional candidate selection."""
    torch.manual_seed(spec.seed)
    model = train_profile_metric.ArcFaceModel(
        inputs[0].shape[1], spec.dimension, spec.player_count,
    ).to(spec.device)
    config = train_profile_metric.MetricConfig(
        spec.epochs, 1024, 0.001, 0.0001, spec.dimension, 0.15, 24.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.0001)
    best: tuple[float, int, dict[str, torch.Tensor] | None] = (
        -1.0, spec.epochs, None,
    )
    for epoch in range(1, spec.epochs + 1):
        train_profile_metric.run_epoch(model, optimizer, *inputs, config)
        if evaluation is not None:
            score = evaluation_score(model, evaluation)
            if score > best[0]:
                best = (score, epoch, copy.deepcopy(model.state_dict()))
        if epoch % 10 == 0 or epoch == spec.epochs:
            print(f"ArcFace d={spec.dimension} seed={spec.seed} "
                  f"epoch={epoch}/{spec.epochs}", flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    return model, best[1]


def make_context(pool: data.Pool, output: pathlib.Path,
                 device: torch.device) -> Context:
    """Creates the same deterministic candidate partition for every seed."""
    game_count = pool.candidates.profiles.shape[1]
    held_count = 20 if game_count >= 100 else 5
    generator = np.random.default_rng(20260927)
    positions = np.stack([
        generator.permutation(game_count)
        for _ in pool.candidates.players
    ])
    return Context(pool, output, device, positions[:, held_count:],
                   positions[:, :held_count],
                   (5, 10, 15, 20) if game_count >= 100 else (1, 2, 3, 5))


def partition(context: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns standardized development profiles and reference class IDs."""
    profiles = context.pool.candidates.profiles
    references = np.stack([
        profiles[index, positions]
        for index, positions in enumerate(context.reference_positions)
    ])
    flattened = references.reshape(-1, profiles.shape[-1])
    held = np.stack([
        profiles[index, positions]
        for index, positions in enumerate(context.holdout_positions)
    ])
    return (
        standardized(flattened, flattened),
        standardized(held, flattened),
        np.repeat(np.arange(len(profiles)), references.shape[1]),
    )


def development_fit(context: Context, spec: FitSpec) -> tuple:
    """Fits the reference-only encoder and encodes all candidate games."""
    references, held, classes = partition(context)
    inputs = (torch.from_numpy(references).to(context.device),
              torch.from_numpy(classes).to(context.device))
    evaluation = Evaluation(inputs[0], inputs[1],
                            torch.from_numpy(held).to(context.device),
                            context.counts)
    model, epoch = fit_model(inputs, spec,
                             evaluation if held.shape[1] == 5 else None)
    profiles = context.pool.candidates.profiles
    raw_reference = np.stack([
        profiles[index, positions]
        for index, positions in enumerate(context.reference_positions)
    ]).reshape(-1, profiles.shape[-1])
    return encode(model, standardized(profiles.reshape(
        -1, profiles.shape[-1],
    ), raw_reference)).reshape(*profiles.shape[:2], spec.dimension), epoch


def final_fit(context: Context, spec: FitSpec) -> tuple[np.ndarray, np.ndarray]:
    """Refits on all candidates, saves weights, and encodes official queries."""
    profiles = context.pool.candidates.profiles
    flattened = profiles.reshape(-1, profiles.shape[-1])
    normalized = standardized(flattened, flattened)
    classes = np.repeat(np.arange(len(profiles)), profiles.shape[1])
    model, _ = fit_model((torch.from_numpy(normalized).to(context.device),
                          torch.from_numpy(classes).to(context.device)), spec)
    torch.save({
        "state": model.state_dict(), "mean": flattened.mean(axis=0),
        "scale": flattened.std(axis=0).clip(min=1e-6),
        "spec": dataclasses.asdict(spec),
        "players": context.pool.candidates.players,
    }, context.output / f"arc{spec.dimension}_seed{spec.seed}.pt")
    return (
        encode(model, normalized).reshape(*profiles.shape[:2], spec.dimension),
        encode(model, standardized(context.pool.queries.profiles, flattened)),
    )


def fit_branch(context: Context, dimension: int) -> Branch:
    """Builds the fixed three-seed branch without using test answers."""
    cache = context.output / f"embeddings_{dimension}.npz"
    if cache.exists():
        with np.load(cache) as values:
            return Branch(values["development"], values["final"],
                          values["queries"], np.asarray(values["epochs"]).tolist())
    development, final, queries, epochs = [], [], [], []
    for index, seed in enumerate((20260927, 20260928, 20260929)):
        spec = FitSpec(dimension, seed,
                       60 if context.counts[0] == 1 else (
                           4 if dimension == 1024 and index == 1 else 3
                       ),
                       len(context.pool.candidates.players), context.device)
        embeddings, epoch = development_fit(context, spec)
        result = final_fit(context, dataclasses.replace(
            spec, epochs=epoch,
        ))
        development.append(embeddings)
        final.append(result[0])
        queries.append(result[1])
        epochs.append(epoch)
    branch = Branch(np.stack(development), np.stack(final),
                    np.stack(queries), epochs)
    np.savez_compressed(cache, **dataclasses.asdict(branch))
    (context.output / f"selection_{dimension}.json").write_text(
        json.dumps({"epochs": epochs, "dimension": dimension}, indent=2),
        encoding="utf-8",
    )
    return branch
