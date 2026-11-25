"""Provides differentiable question pooling and matched per-game controls.

Example: per_game, per_question = losses.question_losses(logp, labels, sizes).
The inputs retain their autograd graph all the way back to decision tokens.
"""

import math

import torch


def question_losses(
    log_probabilities: torch.Tensor,
    labels: torch.Tensor,
    sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns bag-balanced game NLL and normalized log-mean question NLL.

    Args:
        log_probabilities: Per-game log probabilities shaped [games, 10].
        labels: One true rank per game, shaped [games].
        sizes: Number of games in each consecutive single-player bag.

    Raises:
        ValueError: Dimensions, bag sizes or within-bag labels are invalid.
    """
    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 10:
        raise ValueError(
            "Expected per-game log probabilities shaped [games, 10]"
        )
    if labels.shape != log_probabilities.shape[:1]:
        raise ValueError("Expected one rank label per game")
    if not sizes or min(sizes) < 1 or sum(sizes) != len(labels):
        raise ValueError(f"Bag sizes do not cover all games: {sizes!r}")
    per_game = []
    per_question = []
    offset = 0
    for size in sizes:
        ranks = labels[offset : offset + size]
        if not torch.all(ranks == ranks[0]):
            raise ValueError("Question bag mixes different true ranks")
        logp = log_probabilities[offset : offset + size]
        per_game.append(torch.nn.functional.nll_loss(logp, ranks))
        pooled = logp.clamp(min=math.log(1e-6), max=0).mean(dim=0)
        pooled = pooled.log_softmax(dim=0)
        per_question.append(-pooled[ranks[0]])
        offset += size
    return torch.stack(per_game).mean(), torch.stack(per_question).mean()
