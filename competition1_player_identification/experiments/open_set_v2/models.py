"""Defines normalized profile and both-player move encoders for open-set fits.

Encoder accepts game values and target-player colors. Class weights are used
only by the training loss, never by retrieval on unseen players.
"""

import dataclasses

import torch
import torch.nn.functional


@dataclasses.dataclass
class Specification:
    """Stores reproducible architecture choices for a training checkpoint."""

    kind: str
    input_dimension: int
    classes: int
    dimension: int = 256
    width: int = 128
    depth: int = 3
    length: int = 192


class MoveBackbone(torch.nn.Module):
    """Reads both players' moves, absolute color, perspective, and move index."""

    def __init__(self, specification: Specification) -> None:
        super().__init__()
        width = specification.width
        self.embeddings = torch.nn.ModuleDict(
            {
                "position": torch.nn.Embedding(
                    363,
                    width,
                    padding_idx=0,
                ),
                "color": torch.nn.Embedding(2, width),
                "own": torch.nn.Embedding(2, width),
                "step": torch.nn.Embedding(specification.length, width),
            }
        )
        block = torch.nn.TransformerEncoderLayer(
            width,
            4,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            block,
            specification.depth,
            enable_nested_tensor=False,
        )
        self.norm = torch.nn.LayerNorm(width)
        self.pool = torch.nn.Linear(width, 1)
        self.output = torch.nn.Linear(width, specification.dimension)

    def forward(
        self,
        tokens: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Returns attention-pooled game vectors with padded moves excluded."""
        valid = tokens > 0
        values = (tokens - 1).clamp_min(0)
        position = torch.where(
            valid,
            values % 362 + 1,
            0,
        )
        actor = values // 362
        steps = torch.arange(tokens.shape[1], device=tokens.device)
        encoded = (
            self.embeddings["position"](position)
            + self.embeddings["color"](actor)
            + self.embeddings["own"]((actor == colors[:, None]).long())
            + self.embeddings["step"](steps)[None]
        )
        encoded = self.norm(self.encoder(encoded, src_key_padding_mask=~valid))
        weights = self.pool(encoded).squeeze(-1).float()
        weights = weights.masked_fill(~valid, float("-inf")).softmax(1)
        return self.output((weights[..., None] * encoded.float()).sum(1))


class ProfileBackbone(torch.nn.Module):
    """Embeds standardized engine statistics or cheap opening profiles."""

    def __init__(self, specification: Specification) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(specification.input_dimension, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, specification.dimension),
        )

    def forward(
        self,
        values: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Returns game vectors; perspective is already present in profiles."""
        del colors
        return self.encoder(values)


class Encoder(torch.nn.Module):
    """Combines a game backbone with a training-only ArcFace class table."""

    def __init__(self, specification: Specification) -> None:
        super().__init__()
        self.backbone = (
            MoveBackbone(specification)
            if specification.kind == "sequence"
            else ProfileBackbone(specification)
        )
        self.class_weights = torch.nn.Parameter(
            torch.empty(
                specification.classes,
                specification.dimension,
            )
        )
        torch.nn.init.xavier_uniform_(self.class_weights)

    def forward(
        self,
        values: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Returns unit-length embeddings for cosine retrieval."""
        return torch.nn.functional.normalize(
            self.backbone(values, colors), dim=-1
        )

    def arcface_loss(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Applies an angular margin to the true training identity only."""
        cosine = (
            values.float()
            @ torch.nn.functional.normalize(
                self.class_weights.float(),
                dim=-1,
            ).T
        ).float()
        cosine = cosine.clamp(-1 + 1e-5, 1 - 1e-5)
        true = cosine[torch.arange(len(labels), device=labels.device), labels]
        margin = torch.cos(torch.acos(true) + 0.2)
        logits = (
            cosine.scatter(
                1,
                labels[:, None],
                margin[:, None],
            )
            * 24
        )
        return torch.nn.functional.cross_entropy(logits, labels)
