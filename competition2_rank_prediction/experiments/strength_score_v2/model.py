"""Small continuous-strength network over frozen HumanSL rich trunk tokens."""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F


class StrengthScoreNetwork(torch.nn.Module):
    """Maps move-level HumanSL trunk descriptors to one scalar player strength.

    Input shape is [batch, games, moves, features]. The same move head is
    applied to every decision, valid moves are mean-pooled to game strength,
    then games are mean-pooled to player strength.
    """

    def __init__(
        self,
        input_dimensions: int = 960,
        hidden_dimensions: int = 256,
        bottleneck_dimensions: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(input_dimensions)
        self.move_head = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, hidden_dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dimensions, bottleneck_dimensions),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck_dimensions, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns player, game, and move strength scores."""
        if tokens.ndim != 4 or masks.shape != tokens.shape[:-1]:
            raise ValueError(
                f"expected tokens [B,G,M,D] and masks [B,G,M], got "
                f"{tuple(tokens.shape)} and {tuple(masks.shape)}"
            )
        move_scores = self.move_head(self.input_norm(tokens)).squeeze(-1)
        valid = masks.to(move_scores.dtype)
        counts = valid.sum(dim=2).clamp(min=1.0)
        game_scores = (move_scores * valid).sum(dim=2) / counts
        player_scores = game_scores.mean(dim=1)
        return player_scores, game_scores, move_scores


@dataclasses.dataclass(frozen=True)
class LossTerms:
    """Detached-friendly components of the relative-strength objective."""

    total: torch.Tensor
    pairwise: torch.Tensor
    consistency: torch.Tensor
    scale: torch.Tensor


DISTANCE_WEIGHTINGS = ("forward", "uniform", "inverse", "adjacent")


def relative_strength_loss(
    player_scores: torch.Tensor,
    game_scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    consistency_weight: float = 0.10,
    scale_weight: float = 1e-3,
    distance_weighting: str = "forward",
) -> LossTerms:
    """Bradley-Terry style ranking loss plus within-player consistency.

    Larger rank indices are stronger. Under the default "forward" weighting,
    pairs farther apart in registered rank receive more weight, so ambiguous
    adjacent boundaries (notably A/1D) are not forced apart as aggressively as
    distant ranks.

    The remaining holdout3 error is almost entirely adjacent-rank, so the other
    weightings exist to test whether that default trains the scalar away from
    the only boundaries still worth winning: "uniform" weights every unequal
    pair alike, "inverse" prefers near pairs, and "adjacent" supervises only
    one-rank pairs.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if distance_weighting not in DISTANCE_WEIGHTINGS:
        raise ValueError(f"unknown distance weighting: {distance_weighting}")
    gap = labels[:, None] - labels[None, :]
    pair_mask = (gap != 0) & torch.triu(
        torch.ones_like(gap, dtype=torch.bool), diagonal=1
    )
    if not bool(pair_mask.any()):
        raise ValueError("batch contains no unequal-rank pairs")
    direction = gap.sign().to(player_scores.dtype)
    margin = direction * (player_scores[:, None] - player_scores[None, :])
    distance = gap.abs().clamp(max=3).to(player_scores.dtype)
    if distance_weighting == "forward":
        distance_weight = distance / 3.0
    elif distance_weighting == "uniform":
        distance_weight = torch.ones_like(distance)
    elif distance_weighting == "inverse":
        distance_weight = 1.0 / distance
    else:
        distance_weight = (gap.abs() == 1).to(player_scores.dtype)
    per_pair = F.softplus(-margin / temperature)
    weights = distance_weight[pair_mask]
    pairwise = (
        (per_pair[pair_mask] * weights).sum()
        / weights.sum().clamp(min=1e-8)
    )
    consistency = game_scores.var(dim=1, correction=0).mean()
    scale = player_scores.square().mean()
    total = pairwise + consistency_weight * consistency + scale_weight * scale
    return LossTerms(total, pairwise, consistency, scale)
