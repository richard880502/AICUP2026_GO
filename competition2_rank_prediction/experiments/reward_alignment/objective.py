"""Provides ordinal NLL plus a small official-reward objective.

Example: loss = objective.mixed_loss(log_probabilities, labels, 0.2).
"""

import math

import torch


def reward_matrix(reference: torch.Tensor) -> torch.Tensor:
    """Returns the official ten-rank reward table on the input device."""
    indices = torch.arange(10, device=reference.device)
    distance = (indices[:, None] - indices[None, :]).abs()
    return (distance == 0).to(reference.dtype) + math.exp(-1) * (
        distance == 1
    ).to(reference.dtype)


def mixed_loss(
    log_probabilities: torch.Tensor,
    labels: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Computes NLL plus weighted one-minus-expected-official-reward.

    Args:
        log_probabilities: Normalized log probabilities shaped [batch, 10].
        labels: True rank indices shaped [batch].
        weight: Nonnegative auxiliary reward coefficient.

    Raises:
        ValueError: Input dimensions or the coefficient are invalid.
    """
    if weight < 0 or not math.isfinite(weight):
        raise ValueError(
            f"Reward weight must be finite/nonnegative: {weight!r}"
        )
    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 10:
        raise ValueError("Expected log probabilities shaped [batch, 10]")
    if labels.shape != log_probabilities.shape[:1] or not labels.numel():
        raise ValueError("Expected one label per nonempty probability row")
    nll = torch.nn.functional.nll_loss(log_probabilities, labels)
    if weight == 0:
        return nll
    rewards = reward_matrix(log_probabilities)[labels]
    expected = (log_probabilities.exp() * rewards).sum(dim=1).mean()
    return nll + weight * (1 - expected)
