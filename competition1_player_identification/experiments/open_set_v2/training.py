"""Provides player-disjoint episodic sampling and ArcFace/SupCon losses.

Small training libraries use all available distinct games. Singleton players
contribute ArcFace only; query and reference games are never duplicated.
"""

import dataclasses

import numpy as np
import torch
import torch.nn.functional

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import models


@dataclasses.dataclass
class GameData:
    """Holds game values, training groups, and validation-only row indices."""

    values: np.ndarray
    colors: np.ndarray
    groups: list[np.ndarray]
    labels: np.ndarray
    validation: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


@dataclasses.dataclass
class Episode:
    """Holds distinct row positions and local reference/query membership."""

    indices: np.ndarray
    references: list[slice]
    queries: list[slice]


@dataclasses.dataclass
class Runtime:
    """Stores mutable training state for one experiment configuration."""

    model: models.Encoder
    optimizer: torch.optim.Optimizer
    data: GameData
    device: str
    generator: np.random.Generator
    kind: str


def sample_episode(
    groups: list[np.ndarray], generator: np.random.Generator
) -> Episode:
    """Samples sixteen players and 1/2/3/5 query games against up to ten refs."""
    rows: list[int] = []
    references = []
    queries = []
    requested = int(
        generator.choice(
            (
                1,
                2,
                3,
                5,
            )
        )
    )
    for owner in generator.choice(
        len(groups),
        16,
        replace=False,
    ):
        count = min(requested, len(groups[owner]) - 1)
        reference_count = min(10, len(groups[owner]) - count)
        chosen = generator.choice(
            groups[owner],
            count + reference_count,
            replace=False,
        )
        start = len(rows)
        rows.extend(chosen.tolist())
        references.append(slice(start, start + reference_count))
        queries.append(slice(start + reference_count, len(rows)))
    return Episode(
        np.asarray(rows),
        references,
        queries,
    )


def episodic_loss(values: torch.Tensor, episode: Episode) -> torch.Tensor:
    """Learns the same few-query, few-reference centroid matching as evaluation."""
    references = torch.stack(
        [values[part].mean(0) for part in episode.references]
    )
    owners = []
    queries = []
    for owner, part in enumerate(episode.queries):
        if part.stop > part.start:
            owners.append(owner)
            queries.append(values[part].mean(0))
    if not queries:
        return values.sum() * 0
    queries_tensor = torch.nn.functional.normalize(torch.stack(queries), dim=-1)
    references = torch.nn.functional.normalize(references, dim=-1)
    return torch.nn.functional.cross_entropy(
        queries_tensor @ references.T / 0.1,
        torch.tensor(owners, device=values.device),
    )


def contrastive_loss(
    values: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Contrasts different games by identity, excluding singleton positives."""
    identity = torch.eye(
        len(values),
        device=values.device,
        dtype=torch.bool,
    )
    positives = (labels[:, None] == labels[None]) & ~identity
    similarity = (values @ values.T / 0.1).masked_fill(identity, -10000)
    log_probability = similarity - torch.logsumexp(
        similarity,
        dim=1,
        keepdim=True,
    )
    counts = positives.sum(1)
    losses = -(log_probability * positives).sum(1) / counts.clamp_min(1)
    return (losses * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)


def augment_moves(
    tokens: torch.Tensor, generator: np.random.Generator
) -> torch.Tensor:
    """Applies an independent board symmetry per game, preserving move colors."""
    position = (tokens - 1).clamp_min(0) % 362
    column, row = position // 19, position % 19
    transforms = torch.tensor(
        generator.integers(
            0,
            8,
            len(tokens),
        ),
        device=tokens.device,
    )[:, None]
    column, row = (
        torch.where(
            transforms >= 4,
            row,
            column,
        ),
        torch.where(
            transforms >= 4,
            column,
            row,
        ),
    )
    column = torch.where(
        transforms % 4 >= 2,
        18 - column,
        column,
    )
    row = torch.where(
        transforms % 2 == 1,
        18 - row,
        row,
    )
    moved = column * 19 + row + 1 + ((tokens - 1).clamp_min(0) // 362) * 362
    return torch.where(
        (tokens > 0) & (position < 361),
        moved,
        tokens,
    )


def train_step(runtime: Runtime) -> dict:
    """Fits one mixed-color episode with class, set-matching, and game losses."""
    episode = sample_episode(runtime.data.groups, runtime.generator)
    indices = episode.indices
    values = torch.from_numpy(runtime.data.values[indices]).to(runtime.device)
    colors = (
        torch.from_numpy(runtime.data.colors[indices]).long().to(runtime.device)
    )
    labels = (
        torch.from_numpy(runtime.data.labels[indices]).long().to(runtime.device)
    )
    if runtime.kind == "sequence":
        values = augment_moves(values.long(), runtime.generator)
    runtime.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=runtime.device.startswith("cuda"),
    ):
        encoded = runtime.model(values, colors).float()
        angular = runtime.model.arcface_loss(encoded, labels)
        episodic = episodic_loss(encoded, episode)
        contrastive = contrastive_loss(encoded, labels)
        loss = angular + episodic * 0.5 + contrastive * 0.1
    loss.backward()
    torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 2.0)
    runtime.optimizer.step()
    return {
        "loss": float(loss.detach()),
        "arcface": float(angular.detach()),
        "episodic": float(episodic.detach()),
        "supcon": float(contrastive.detach()),
    }


def encode(runtime: Runtime, indices: np.ndarray) -> np.ndarray:
    """Encodes requested rows without updating weights or normalization."""
    outputs = []
    runtime.model.eval()
    with torch.no_grad():
        for chosen in artifacts.batch_indices(indices):
            values = torch.from_numpy(runtime.data.values[chosen]).to(
                runtime.device
            )
            colors = (
                torch.from_numpy(runtime.data.colors[chosen])
                .long()
                .to(runtime.device)
            )
            if runtime.kind == "sequence":
                values = values.long()
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=runtime.device.startswith("cuda"),
            ):
                outputs.append(
                    runtime.model(values, colors).float().cpu().numpy()
                )
    return np.concatenate(outputs)
