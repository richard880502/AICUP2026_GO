"""Fits bounded relative member weights with fixed question corrections.

Call fit with game-by-member-by-class log probabilities, question row masks
and frozen C1 offsets. Evaluation labels must never enter the optimizer.
"""

import numpy as np
import torch


def weights(raw: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Preserves nonnegativity and total weight under bounded log shifts."""
    proposed = baseline * (0.2 * raw.tanh()).exp()
    return proposed / proposed.sum() * baseline.sum()


def posterior(
    logs: torch.Tensor,
    offset: torch.Tensor,
    values: torch.Tensor,
    groups: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Applies frozen C1 correction after differentiable relative fusion."""
    per_game = (logs * values[None, :, None]).sum(1).log_softmax(1)
    per_game = per_game.clamp(min=float(np.log(1e-6)), max=0)
    rows, mask = groups
    pooled = (per_game[rows] * mask[..., None]).sum(1) / mask.sum(
        1, keepdim=True
    )
    return (pooled + offset).softmax(1)


def fit(
    logs: dict,
    offset: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    method: str,
) -> np.ndarray:
    """Uses exactly300full-batch updates and the fixed objective regularizer."""
    if method not in ("nll", "utility"):
        raise ValueError(f"Unknown fusion objective: {method!r}")
    tensors = (
        torch.as_tensor(logs["members"], dtype=torch.float64),
        torch.as_tensor(offset, dtype=torch.float64),
        torch.as_tensor(baseline, dtype=torch.float64),
    )
    target = torch.as_tensor(labels, dtype=torch.long)
    groups = (
        torch.as_tensor(logs["rows"], dtype=torch.long),
        torch.as_tensor(logs["mask"], dtype=torch.float64),
    )
    raw = torch.nn.Parameter(torch.zeros(len(baseline), dtype=torch.float64))
    optimizer = torch.optim.Adam([raw], lr=0.02)
    reward = torch.eye(10, dtype=torch.float64)
    reward += np.exp(-1) * (
        torch.eye(10, 10, dtype=torch.float64).roll(1, 0)
        + torch.eye(10, 10, dtype=torch.float64).roll(-1, 0)
    )
    reward[0, 9] = 0
    reward[9, 0] = 0
    for _ in range(300):
        optimizer.zero_grad()
        probability = posterior(
            tensors[0], tensors[1], weights(raw, tensors[2]), groups
        )
        if method == "nll":
            loss = -probability[torch.arange(len(target)), target].log().mean()
        else:
            decision = (probability @ reward / 0.1).softmax(1)
            loss = -(decision * reward[target]).sum(1).mean()
        loss = loss + 0.1 * (0.2 * raw.tanh()).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite fusion fitting objective")
        loss.backward()
        optimizer.step()
    return weights(raw, tensors[2]).detach().numpy()


def predict(logs: dict, offset: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Evaluates fixed relative weights without accessing evaluation labels."""
    with torch.inference_mode():
        return posterior(
            torch.as_tensor(logs["members"], dtype=torch.float64),
            torch.as_tensor(offset, dtype=torch.float64),
            torch.as_tensor(values, dtype=torch.float64),
            (
                torch.as_tensor(logs["rows"], dtype=torch.long),
                torch.as_tensor(logs["mask"], dtype=torch.float64),
            ),
        ).numpy()
