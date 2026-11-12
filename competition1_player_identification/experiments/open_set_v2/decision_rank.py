"""Retrains the Competition 2 decision-Transformer architecture as a rank prior.

Uses the new 64-by-41 KataGo/HumanSL/ownership cache, rather than assuming its
schema matches old C2 checkpoints. Three fresh ordinal models exclude both
style-evaluation cohorts; 300 inner training players select/calibrate them.
"""

import dataclasses
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import ordinal
from competition1_player_identification.experiments.open_set_v2 import (
    rank_prior,
)

SEEDS = (20261111, 20261112, 20261113)


class DecisionRanker(ordinal.RankBase):
    """Adapts C2's 128D / three-layer decision model to the 41 input channels."""

    def __init__(self) -> None:
        super().__init__()
        block = ordinal.encoder_layer(128, 4, 384)
        self.layers = torch.nn.ModuleDict(
            {
                "project": torch.nn.Sequential(
                    torch.nn.Linear(41, 128), torch.nn.LayerNorm(128)
                ),
                "encoder": torch.nn.TransformerEncoder(
                    block, 3, enable_nested_tensor=False
                ),
                "norm": torch.nn.LayerNorm(128),
                "pool": torch.nn.Linear(128, 1),
                "score": torch.nn.Linear(128, 1),
            }
        )
        self.parameters_by_name["position"] = torch.nn.Parameter(
            torch.randn(1, 64, 128) * 0.02,
        )

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Returns ordered rank probabilities from masked decision tokens."""
        encoded = (
            self.layers["project"](values) + self.parameters_by_name["position"]
        )
        return self.rank_log_probabilities(self.pool_game(encoded, valid))


@dataclasses.dataclass
class RankData:
    """Holds normalized decisions and clean training-only rank partitions.

    Attributes:
        values: Decision feature tensor in selected-game order.
        masks: Valid decision positions.
        labels: True ranks, used only at train and inner positions.
        train: Training rows excluding the internal calibration players.
        inner: Rows of 300 internal training players.
        validation: Style-validation query/reference rows, used without grades.
    """

    values: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    inner: np.ndarray
    validation: np.ndarray


def load_tokens(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads complete engine decisions in the immutable extraction row order."""
    tokens, masks = [], []
    for path in sorted(
        (root / "engine_features/candidates").glob("shard_*.npz")
    ):
        with np.load(path) as arrays:
            tokens.append(np.asarray(arrays["tokens"]))
            masks.append(np.asarray(arrays["masks"]))
    values = np.concatenate(tokens)
    valid = np.concatenate(masks)
    if len(values) != len(np.load(root / "engine_inputs/global_indices.npy")):
        raise ValueError("Incomplete engine cache for the rank prior")
    if values.shape[1:] != (64, 41) or not valid.any(1).all():
        raise ValueError("Invalid decision shape or empty engine game")
    return values, valid


def fit_scaling(
    values: np.ndarray, masks: np.ndarray, rows: np.ndarray
) -> tuple:
    """Fits masked mean/variance on rank-training rows only, in bounded chunks."""
    sums = np.zeros(41, np.float64)
    squares = np.zeros(41, np.float64)
    count = 0
    for indices in np.array_split(rows, 100):
        selected = values[indices][masks[indices]].astype(np.float64)
        sums += selected.sum(0)
        squares += np.square(selected).sum(0)
        count += len(selected)
    mean = sums / count
    scale = np.sqrt(np.maximum(squares / count - np.square(mean), 1e-8))
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize(
    values: np.ndarray, masks: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Applies fixed rank-training statistics and zeros padded decisions."""
    return np.where(
        masks[..., None], np.clip((values - mean) / scale, -10, 10), 0
    ).astype(np.float32)


def prepare(root: pathlib.Path) -> RankData:
    """Uses the same 300 inner players as the clean C2 raw-move control."""
    values, valid = load_tokens(root)
    split = rank_prior.load_data()
    mean, scale = fit_scaling(values, valid, split.train)
    with (root / "rank_engine/preprocessing.npz").open("wb") as output:
        np.savez_compressed(output, mean=mean, scale=scale)
    return RankData(
        normalize(values, valid, mean, scale),
        valid,
        split.labels,
        split.train,
        split.inner,
        split.validation,
    )


def predict(
    model: DecisionRanker, data: RankData, rows: np.ndarray
) -> np.ndarray:
    """Predicts requested games without reading ranks or changing model state."""
    model.eval()
    outputs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for chosen in artifacts.batch_indices(rows):
            logits = model(
                torch.from_numpy(data.values[chosen]).cuda(0),
                torch.from_numpy(data.masks[chosen]).cuda(0),
            )
            outputs.append(ordinal.probability_array(logits))
    return np.concatenate(outputs)


def fit_seed(root: pathlib.Path, data: RankData, seed: int) -> None:
    """Selects a seed checkpoint on inner rank NLL, never style validation."""
    torch.manual_seed(seed)
    model = DecisionRanker().cuda(0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 16, eta_min=0.00003
    )
    generator = np.random.default_rng(seed)
    history = []
    best = float("inf")
    for epoch in range(16):
        loss = ordinal.train_epoch(
            (
                model,
                optimizer,
                (data.values, data.masks, data.labels, data.train),
                generator,
                False,
                512,
            )
        )
        scheduler.step()
        probabilities = predict(model, data, data.inner)
        nll = float(
            -np.log(
                probabilities[
                    np.arange(len(data.inner)), data.labels[data.inner]
                ].clip(min=1e-7)
            ).mean()
        )
        row = {"epoch": epoch + 1, "loss": loss, "inner_nll": nll}
        history.append(row)
        best = ordinal.save_selected(
            (model, root / f"seed_{seed}.pt", nll, best, row)
        )
        artifacts.write_report(
            root / f"history_{seed}.json", {"epochs": history}
        )
        print({"seed": seed, **row}, flush=True)


def ensemble(
    root: pathlib.Path, data: RankData, rows: np.ndarray
) -> np.ndarray:
    """Averages the three selected C2 decision-rank probability predictions."""
    predictions = []
    for seed in SEEDS:
        checkpoint = torch.load(
            root / f"seed_{seed}.pt", map_location="cpu", weights_only=False
        )
        model = DecisionRanker().cuda(0)
        model.load_state_dict(checkpoint["state"])
        predictions.append(predict(model, data, rows))
    return np.mean(predictions, axis=0)


def main() -> None:
    """Trains three clean engine-assisted rank priors and their soft affinity."""
    root = dataset.ROOT
    output = root / "rank_engine"
    if (output / "completed.json").exists():
        raise FileExistsError(
            "Clean engine-rank experiment is already complete"
        )
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    data = prepare(root)
    for seed in SEEDS:
        fit_seed(output, data, seed)
    inner = ensemble(output, data, data.inner)
    nll, temperature = rank_prior.calibrate(inner, data.labels[data.inner])
    probabilities = ensemble(output, data, data.validation.ravel()).clip(
        min=1e-7
    ) ** (1 / temperature)
    probabilities /= probabilities.sum(1, keepdims=True)
    probabilities = np.reshape(probabilities, (600, 15, 10))
    np.save(output / "validation_probabilities.npy", probabilities)
    scores = evaluation.rank_affinity(probabilities)
    np.save(output / "validation_scores.npy", scores)
    artifacts.write_report(
        output / "completed.json",
        {
            "status": "complete",
            "branch": "C2 decision ordinal Transformer; 64x41 inputs; clean three-seed retraining",
            "inner_rank_nll": nll,
            "temperature": temperature,
            "style_validation_rank_only": evaluation.summarize(scores),
            "rank_training_games": len(data.train),
            "rank_calibration_games": len(data.inner),
            "hard_candidate_pruning": False,
        },
        emit=True,
    )


if __name__ == "__main__":
    main()
