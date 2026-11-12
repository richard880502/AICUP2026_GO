"""Shares bounded-gradient optimizer updates across experiment learners."""

import torch


def step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
) -> float:
    """Backpropagates one loss and returns its detached logging value."""
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return float(loss.detach())
