"""Provides the differentiable ordinal pair-ranking auxiliary objective.

The caller pools scalar per-game strengths within each player's bag.
"""

import torch


def pairwise_loss(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Returns softplus ranking loss for adjacent low/high player strengths."""
    if low.shape != high.shape or low.numel() == 0:
        raise ValueError("Pair strengths must have equal nonempty shapes")
    return torch.nn.Softplus()(-(high - low)).mean()


def combined_loss(
    nll: torch.Tensor,
    pair_loss: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Adds auxiliary supervision, with exact zero-weight identity."""
    if weight == 0:
        return nll
    return nll + weight * pair_loss
