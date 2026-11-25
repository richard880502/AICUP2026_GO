"""Computes differentiable played-move likelihood on legal policy support.

Use played_log_probability with logits, played indices and a Japanese-rule
legal mask from replay; illegal moves contribute neither mass nor gradient.
"""

import torch


def played_log_probability(
    logits: torch.Tensor, points: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Normalizes legal logits and returns each observed move's log probability.

    Args:
        logits: Position-by-move policy logits including the pass location.
        points: One observed move index per position.
        legal: Boolean support mask matching logits.

    Returns:
        One differentiable log probability per observed move.

    Raises:
        ValueError: Shapes, indices or observed move legality are invalid.
    """
    if (
        logits.ndim != 2
        or legal.shape != logits.shape
        or legal.dtype != torch.bool
    ):
        raise ValueError(
            "Policy requires matching two-dimensional logits/boolean mask"
        )
    if points.shape != logits.shape[:1] or points.dtype != torch.long:
        raise ValueError("Played points must be one long index per policy row")
    if ((points < 0) | (points >= logits.shape[1])).any():
        raise ValueError("Observed move index is outside policy support")
    if not legal.gather(1, points[:, None]).all():
        raise ValueError("Observed move is illegal under the supplied mask")
    normalized = logits.masked_fill(~legal, float("-inf")).log_softmax(1)
    return normalized.gather(1, points[:, None]).squeeze(1)
