"""Fits a fixed low-capacity residual MLP on player-disjoint OOF queries."""

import dataclasses

import numpy as np
import torch
from torch.nn import functional

HIDDEN_WIDTH = 32
DROPOUT = 0.1
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.05
MAX_EPOCHS = 100
PATIENCE = 10
VALIDATION_FOLD = 4


@dataclasses.dataclass
class Examples:
    """Stores query inputs for meta-calibration.

    Attributes:
        features: C1 question summaries and baseline logits.
        logits: Fitted C1+C2 logits before the meta residual.
        labels: Integer rank targets.
        players: Player IDs used to maintain group isolation.
        folds: Player folds used for meta-level early stopping.
    """

    features: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    players: np.ndarray
    folds: np.ndarray


@dataclasses.dataclass
class FittedModel:
    """Stores a standardized residual MLP and its selected epoch.

    Attributes:
        network: Zero-initialized residual network trained on OOF queries.
        mean: OOF feature means.
        std: OOF feature standard deviations.
        best_epoch: Epoch selected on held-player OOF queries.
        validation_nll: Best held-player NLL during epoch selection.
    """

    network: torch.nn.Sequential
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    validation_nll: float


def build(input_width: int) -> torch.nn.Sequential:
    """Creates an identity-initialized10-class logit residual network."""
    model = torch.nn.Sequential(
        torch.nn.Linear(input_width, HIDDEN_WIDTH),
        torch.nn.GELU(),
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN_WIDTH, 10),
    )
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def _tensors(
    examples: Examples, mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts one player partition to standardized device-independent tensors."""
    features = torch.as_tensor(
        (examples.features[mask] - mean) / std, dtype=torch.float32
    )
    logits = torch.as_tensor(examples.logits[mask], dtype=torch.float32)
    labels = torch.as_tensor(examples.labels[mask], dtype=torch.long)
    return features, logits, labels


def _select_epoch(
    input_width: int,
    training: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[int, float]:
    """Selects the earliest minimum-NLL epoch on held-player OOF rows."""
    network = build(input_width).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train = [value.to(device) for value in training]
    valid = [value.to(device) for value in validation]
    best_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        network.train()
        optimizer.zero_grad()
        loss = functional.cross_entropy(
            train[1] + network(train[0]), train[2]
        )
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.no_grad():
            valid_output = valid[1] + network(valid[0])
            valid_nll = float(
                functional.cross_entropy(valid_output, valid[2])
            )
        if valid_nll < best_nll:
            best_nll = valid_nll
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_epoch < 1:
        raise ValueError("Meta-MLP did not select a valid training epoch")
    return best_epoch, best_nll


def _refit(
    examples: Examples,
    mean: np.ndarray,
    std: np.ndarray,
    best_epoch: int,
    device: torch.device,
) -> torch.nn.Sequential:
    """Refits the selected architecture and epoch using every OOF player."""
    network = build(examples.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    all_features, all_logits, all_labels = _tensors(
        examples, np.ones(len(examples.labels), dtype=bool), mean, std
    )
    all_features = all_features.to(device)
    all_logits = all_logits.to(device)
    all_labels = all_labels.to(device)
    for _ in range(best_epoch):
        network.train()
        optimizer.zero_grad()
        loss = functional.cross_entropy(
            all_logits + network(all_features), all_labels
        )
        loss.backward()
        optimizer.step()
    network.eval()
    return network


def fit(
    examples: Examples, player_folds: dict[str, int], device: torch.device
) -> FittedModel:
    """Selects epoch on fold4 and refits the fixed model on all OOF rows."""
    validation_mask = np.asarray(
        [
            player_folds[str(player)] == VALIDATION_FOLD
            for player in examples.players
        ]
    )
    training_mask = ~validation_mask
    if not validation_mask.any() or not training_mask.any():
        raise ValueError("Meta-training split must contain both player groups")
    mean = examples.features[training_mask].mean(0)
    std = examples.features[training_mask].std(0) + 1e-6
    training = _tensors(examples, training_mask, mean, std)
    validation = _tensors(examples, validation_mask, mean, std)
    best_epoch, validation_nll = _select_epoch(
        examples.features.shape[1], training, validation, device
    )
    network = _refit(examples, mean, std, best_epoch, device)
    return FittedModel(network, mean, std, best_epoch, validation_nll)


def predict_logits(
    fitted: FittedModel,
    features: np.ndarray,
    base_logits: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Applies the fitted residual network to held-out query logits."""
    input_tensor = torch.as_tensor(
        (features - fitted.mean) / fitted.std,
        dtype=torch.float32,
        device=device,
    )
    logits_tensor = torch.as_tensor(
        base_logits, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        output = logits_tensor + fitted.network(input_tensor)
    return output.cpu().numpy()
