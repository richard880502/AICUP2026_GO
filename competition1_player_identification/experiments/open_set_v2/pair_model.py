"""Fits a small group-softmax reranker on frozen, training-only game pairs.

Validation chooses epochs and blending strength. Stage-one misses remain
misses, and a zero-strength blend is always considered.
"""

import dataclasses

import numpy as np
import torch
import torch.nn.functional

from competition1_player_identification.experiments.open_set_v2 import (
    optimization,
)

from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import pair_data
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)

BLENDS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
CHECKPOINT_EPOCHS = (10, 20, 40, 80)


class PairReranker(torch.nn.Module):
    """Scores candidate pairs independently before query-group normalization."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one logit per candidate without depending on its player ID."""
        return self.network(features).squeeze(-1)


@dataclasses.dataclass
class PairFit:
    """Retains a selected model and training-only feature scaling.

    Attributes:
        state: Selected CPU parameter tensors.
        mean: Training pair-feature means.
        scale: Training pair-feature standard deviations.
        selection: Selected epoch, blend, and validation score.
        history: Training losses and held-out selection trials.
        logits: Selected model logits in original validation-query order.
    """

    state: dict
    mean: np.ndarray
    scale: np.ndarray
    selection: dict
    history: list
    logits: np.ndarray


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Uses fixed train-only scaling and bounds outlying validation features."""
    return np.clip((features - mean) / scale, -10, 10).astype(np.float32)


def predict(model: PairReranker, values: np.ndarray, device: str) -> np.ndarray:
    """Returns grouped logits with no optimizer or feature fitting."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for indices in np.array_split(np.arange(len(values)), 20):
            if len(indices):
                outputs.append(
                    model(torch.from_numpy(values[indices]).to(device))
                    .cpu()
                    .numpy()
                )
    return np.concatenate(outputs)


def fit_epoch(context: tuple) -> float:
    """Fits only query groups whose natural Top-20 contains the true player."""
    model, optimizer, values, targets, generator = context
    model.train()
    losses = []
    for indices in np.array_split(generator.permutation(len(values)), 32):
        if len(indices):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(values[indices]), targets[indices]
            )
            losses.append(optimization.step(model, optimizer, loss))
    return float(np.mean(losses))


def choose_blend(context: tuple) -> tuple[dict, np.ndarray]:
    """Selects an epoch-local blend by the exact exponential Top-5 utility."""
    stage, candidate_ids, logits, epoch = context
    trials = []
    for weight in BLENDS:
        reordered = pair_features.reorder_scores(
            stage, candidate_ids, logits, weight
        )
        ranks = evaluation.correct_ranks(reordered)
        reward = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0).mean()
        trials.append({"epoch": epoch, "blend": weight, "score": float(reward)})
    selection = max(trials, key=lambda row: (row["score"], -row["blend"]))
    return selection, pair_features.reorder_scores(
        stage, candidate_ids, logits, selection["blend"]
    )


@dataclasses.dataclass
class Runtime:
    """Holds a small reranker fit independently of frozen game encoders.

    Attributes:
        model: Pair-scoring network.
        optimizer: Optimizer for the pair network only.
        values: Accepted training groups after training-only scaling.
        targets: Correct hard-candidate slots for training groups.
        validation: Scaled features for every style-validation question.
        pairs: Original validation scores and candidate order.
        generator: Deterministic training batch generator.
    """

    model: PairReranker
    optimizer: torch.optim.Optimizer
    values: torch.Tensor
    targets: torch.Tensor
    validation: np.ndarray
    pairs: pair_data.PairDataset
    generator: np.random.Generator


def prepare_fit(context: tuple) -> tuple:
    """Fits pair-feature scaling using only natural training Top-20 hits."""
    train_pairs, validation_pairs, columns, device = context
    accepted = train_pairs.targets >= 0
    if accepted.sum() < 32:
        raise ValueError("Too few natural Top-20 positive query groups to fit")
    features = train_pairs.features[accepted][:, :, columns]
    mean = features.reshape(-1, len(columns)).mean(0)
    scale = features.reshape(-1, len(columns)).std(0).clip(min=1e-4)
    values = torch.from_numpy(normalize_features(features, mean, scale)).to(
        device
    )
    targets = torch.from_numpy(train_pairs.targets[accepted]).long().to(device)
    validation = normalize_features(
        validation_pairs.features[:, :, columns], mean, scale
    )
    torch.manual_seed(20261112)
    model = PairReranker(len(columns)).to(device)
    return (
        Runtime(
            model,
            torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01),
            values,
            targets,
            validation,
            validation_pairs,
            np.random.default_rng(20261112),
        ),
        mean,
        scale,
    )


def fit(context: tuple) -> PairFit:
    """Fits one feature-family ablation without updating either game encoder."""
    runtime, mean, scale = prepare_fit(context)
    selected = PairFit({}, mean, scale, {"score": -1.0}, [], np.zeros(0))
    for epoch in range(1, 81):
        loss = fit_epoch(
            (
                runtime.model,
                runtime.optimizer,
                runtime.values,
                runtime.targets,
                runtime.generator,
            )
        )
        row: dict = {"epoch": epoch, "train_loss": loss}
        if epoch in CHECKPOINT_EPOCHS:
            logits = predict(
                runtime.model, runtime.validation, str(runtime.values.device)
            )
            selection, _ = choose_blend(
                (
                    runtime.pairs.scores.reshape(4, 600, 600),
                    runtime.pairs.top_ids,
                    logits,
                    epoch,
                )
            )
            row["validation"] = selection
            if selection["score"] > selected.selection["score"]:
                selected.state = {
                    name: value.detach().cpu().clone()
                    for name, value in runtime.model.state_dict().items()
                }
                selected.selection = selection
                selected.logits = logits
        selected.history.append(row)
    return selected
