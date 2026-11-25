# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Synthetic forward/backward smoke test for Rank Context v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from competition2_rank_prediction.experiments.rank_context.model import (
    RankContextModel,
)


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def main() -> None:
    """Checks all four model configurations without external artifacts."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    classes = 32
    games = 18
    questions = 6
    context_dimensions = 15

    player_rank = torch.arange(classes) % 10
    # pylint false positive: torch callables are not inferred.
    # pylint: disable-next=not-callable
    ranks = F.one_hot(player_rank, num_classes=10).float().to(device)
    scores = torch.randn(games, classes, device=device)
    kind = torch.tensor(
        [0, 1, 2, 3, 0, 3, 0, 1, 3, 2, 0, 0, 3, 1, 0, 2, 3, 0],
        device=device,
    )
    game_question = torch.arange(games, device=device) % questions
    base = torch.rand(questions, 10, device=device)
    base = base / base.sum(1, keepdim=True)
    handicap = torch.tensor(
        [0, 1, 1, 2, 0, 3, 0, 1, 4, 1, 0, 0, 2, 1, 0, 1, 5, 0],
        device=device,
    )
    own_black = torch.tensor(
        [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        dtype=torch.bool,
        device=device,
    )
    context = torch.rand(
        questions, context_dimensions, device=device
    )
    truth = torch.arange(questions, device=device) % 10
    arguments = (
        scores,
        kind,
        game_question,
        base,
        handicap,
        own_black,
        context,
    )

    for use_handicap, use_context in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        model = RankContextModel(
            ranks,
            context_dimensions,
            use_handicap,
            use_context,
        ).to(device)
        logits = model(*arguments)
        assert logits.shape == (questions, 10)
        assert torch.isfinite(logits).all()
        loss = F.cross_entropy(logits, truth) + model.regularization(0.01)
        loss.backward()
        assert torch.isfinite(loss)
        print(
            "ok",
            f"handicap={use_handicap}",
            f"context={use_context}",
            f"loss={float(loss):.4f}",
        )


if __name__ == "__main__":
    main()
