"""Shares Competition 2 ordinal heads and clean rank-training operations.

RankBase retains the old raw-move checkpoint key names while allowing the
same cumulative-link model to consume engine-decision tokens.
"""

import abc
import pathlib

import numpy as np
import torch
import torch.nn.functional

from competition1_player_identification.experiments.open_set_v2 import (
    optimization,
)

from competition1_player_identification.experiments.open_set_v2 import training


def log_probabilities(
    score: torch.Tensor,
    gaps_parameter: torch.Tensor,
    first: torch.Tensor,
) -> torch.Tensor:
    """Computes the original ten-class ordinal probabilities in float32."""
    gaps = torch.nn.Softplus()(gaps_parameter + 0.5)
    thresholds = first + torch.cat(
        (
            torch.zeros(1, device=score.device),
            gaps.cumsum(0),
        )
    )
    cumulative = torch.sigmoid(score.float() - thresholds)
    upper = torch.cat((torch.ones_like(cumulative[:, :1]), cumulative), dim=1)
    lower = torch.cat((cumulative, torch.zeros_like(cumulative[:, :1])), dim=1)
    return (upper - lower).clamp_min(1e-7).log()


def encoder_layer(
    width: int, heads: int, feedforward: int
) -> torch.nn.TransformerEncoderLayer:
    """Creates the shared C2 pre-norm Transformer block."""
    return torch.nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=feedforward,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class RankBase(torch.nn.Module, abc.ABC):
    """Shares attention pooling and ordinal rank classification.

    Attributes:
        layers: Child-specific projection, encoder, pool and score modules.
        parameters_by_name: Learned increasing ordinal cut points.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.parameters_by_name = torch.nn.ParameterDict(
            {
                "gaps": torch.nn.Parameter(torch.zeros(8)),
                "first": torch.nn.Parameter(torch.tensor(-2.0)),
            }
        )

    def pool_game(
        self, encoded: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Applies masked temporal encoding and float32 attention pooling."""
        encoded = self.layers["norm"](
            self.layers["encoder"](
                encoded,
                src_key_padding_mask=~valid,
            )
        )
        weights = self.layers["pool"](encoded).squeeze(-1).float()
        weights = weights.masked_fill(~valid, float("-inf")).softmax(1)
        return (weights[..., None] * encoded.float()).sum(1)

    def rank_log_probabilities(self, pooled: torch.Tensor) -> torch.Tensor:
        """Applies the original cumulative-link rank head to game vectors."""
        return log_probabilities(
            self.layers["score"](pooled).float(),
            self.parameters_by_name["gaps"],
            self.parameters_by_name["first"],
        )


def probability_array(logits: torch.Tensor) -> np.ndarray:
    """Returns detached rank probabilities in the input game order."""
    return logits.detach().float().exp().cpu().numpy()


def train_epoch(context: tuple) -> float:
    """Fits a clean rank-training table, with raw-move augmentation if needed."""
    model, optimizer, table, generator, raw_moves, batches = context
    model.train()
    losses = []
    for indices in np.array_split(generator.permutation(table[3]), batches):
        values = torch.from_numpy(table[0][indices]).cuda(0)
        if raw_moves:
            values = training.augment_moves(values.long(), generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(values, torch.from_numpy(table[1][indices]).cuda(0))
            loss = torch.nn.functional.nll_loss(
                logits,
                torch.from_numpy(table[2][indices]).cuda(0),
            )
        losses.append(optimization.step(model, optimizer, loss))
    return float(np.mean(losses))


def save_selected(context: tuple) -> float:
    """Keeps a lower-inner-NLL checkpoint without consulting style outcomes."""
    model, destination, nll, best, row = context
    if nll < best:
        torch.save(
            {"state": model.state_dict(), "selection": row},
            pathlib.Path(destination),
        )
    return min(nll, best)
