"""Retrains Competition 2's raw-move ordinal branch as a clean rank prior.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.rank_prior

Architecture and ordinal equations follow competition2_rank_prediction's
move_transformer.py. This is its raw branch, not the full C1+C2 ensemble.
Existing checkpoints are deliberately not reused: their training-player sets
may include the new validation identities. Inner training players select the
rank checkpoint and temperature; style validation only selects fusion later.
"""

import dataclasses

import numpy as np
import torch
import torch.nn.functional

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import training


class C2MoveRanker(torch.nn.Module):
    """Uses the Competition 2 192D, four-layer ordinal move Transformer."""

    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.TransformerEncoderLayer(
            192,
            6,
            dim_feedforward=768,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = torch.nn.ModuleDict(
            {
                "position": torch.nn.Embedding(362, 192),
                "color": torch.nn.Embedding(2, 192),
                "own": torch.nn.Embedding(2, 192),
                "step": torch.nn.Embedding(256, 192),
                "encoder": torch.nn.TransformerEncoder(
                    block,
                    4,
                    enable_nested_tensor=False,
                ),
                "norm": torch.nn.LayerNorm(192),
                "pool": torch.nn.Linear(192, 1),
                "score": torch.nn.Linear(192, 1),
            }
        )
        self.parameters_by_name = torch.nn.ParameterDict(
            {
                "gaps": torch.nn.Parameter(torch.zeros(8)),
                "first": torch.nn.Parameter(torch.tensor(-2.0)),
            }
        )
        self.softplus = torch.nn.Softplus()

    def forward(
        self,
        tokens: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ten ordinal log probabilities, masking padding explicitly."""
        valid = tokens > 0
        values = (tokens - 1).clamp_min(0)
        position, actor = values % 362, values // 362
        steps = torch.arange(tokens.shape[1], device=tokens.device)
        encoded = (
            self.layers["position"](position)
            + self.layers["color"](actor)
            + self.layers["own"]((actor == colors[:, None]).long())
            + self.layers["step"](steps)[None]
        )
        encoded = self.layers["norm"](
            self.layers["encoder"](
                encoded,
                src_key_padding_mask=~valid,
            )
        )
        weights = self.layers["pool"](encoded).squeeze(-1).float()
        weights = weights.masked_fill(~valid, float("-inf")).softmax(1)
        pooled = (weights[..., None] * encoded.float()).sum(1)
        return self.rank_log_probabilities(pooled)

    def rank_log_probabilities(self, pooled: torch.Tensor) -> torch.Tensor:
        """Applies the original cumulative-link rank head to game vectors."""
        gaps = self.softplus(self.parameters_by_name["gaps"] + 0.5)
        thresholds = self.parameters_by_name["first"] + torch.cat(
            (
                torch.zeros(1, device=pooled.device),
                gaps.cumsum(0),
            )
        )
        cumulative = torch.sigmoid(
            self.layers["score"](pooled).float() - thresholds
        )
        upper = torch.cat(
            (torch.ones_like(cumulative[:, :1]), cumulative), dim=1
        )
        lower = torch.cat(
            (cumulative, torch.zeros_like(cumulative[:, :1])), dim=1
        )
        return (upper - lower).clamp_min(1e-7).log()


@dataclasses.dataclass
class RankData:
    """Holds training rows and a player-disjoint, internal rank calibration set."""

    tokens: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    inner: np.ndarray
    validation: np.ndarray


def load_data() -> RankData:
    """Builds a 300-player inner split entirely inside the 6,807 train players."""
    root = dataset.ROOT
    arrays = dataset.load_arrays(
        root / "raw_cache",
        (
            "tokens",
            "players",
            "colors",
            "rank",
        ),
    )
    selected = np.load(root / "engine_inputs/global_indices.npy")
    with np.load(root / "clean_splits.npz") as splits:
        train_rows = np.flatnonzero(np.isin(selected, splits["train"]))
        validation = np.searchsorted(selected, splits["validation"])
    players = arrays["players"][selected]
    generator = np.random.default_rng(20261111)
    inner_players = generator.choice(
        np.unique(players[train_rows]),
        300,
        replace=False,
    )
    inner = np.isin(players[train_rows], inner_players)
    return RankData(
        arrays["tokens"][selected, :256],
        arrays["colors"][selected].astype(np.int64),
        arrays["rank"][selected].astype(np.int64),
        train_rows[~inner],
        train_rows[inner],
        validation,
    )


def predict(
    model: C2MoveRanker,
    data: RankData,
    indices: np.ndarray,
) -> np.ndarray:
    """Returns uncalibrated rank probabilities for exactly the requested rows."""
    outputs = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for chosen in artifacts.batch_indices(indices):
            outputs.append(
                model(
                    torch.from_numpy(data.tokens[chosen]).long().cuda(0),
                    torch.from_numpy(data.colors[chosen]).cuda(0),
                )
                .float()
                .exp()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def train_epoch(context: tuple) -> float:
    """Trains one shuffled pass using only non-inner training identities."""
    model, optimizer, data, generator = context
    model.train()
    losses = []
    for chosen in np.array_split(generator.permutation(data.train), 1024):
        tokens = torch.from_numpy(data.tokens[chosen]).long().cuda(0)
        tokens = training.augment_moves(tokens, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                tokens, torch.from_numpy(data.colors[chosen]).cuda(0)
            )
            loss = torch.nn.functional.nll_loss(
                logits,
                torch.from_numpy(data.labels[chosen]).cuda(0),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def calibrate(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Chooses a scalar temperature using only internal rank-validation labels."""
    candidates = []
    for temperature in (
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    ):
        calibrated = probabilities.clip(min=1e-7) ** (1 / temperature)
        calibrated /= calibrated.sum(1, keepdims=True)
        nll = -np.log(calibrated[np.arange(len(labels)), labels]).mean()
        candidates.append((float(nll), temperature))
    return min(candidates)


def run() -> None:
    """Fits the rank branch, calibrates internally, and saves style-val priors."""
    root = dataset.ROOT / "rank_prior"
    root.mkdir(exist_ok=True)
    data = load_data()
    model = C2MoveRanker().cuda(0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
    )
    generator = np.random.default_rng(20261111)
    best = float("inf")
    history = []
    for epoch in range(8):
        loss = train_epoch(
            (
                model,
                optimizer,
                data,
                generator,
            )
        )
        probabilities = predict(
            model,
            data,
            data.inner,
        )
        nll, temperature = calibrate(probabilities, data.labels[data.inner])
        row = {
            "epoch": epoch + 1,
            "loss": loss,
            "inner_nll": nll,
            "temperature": temperature,
        }
        history.append(row)
        if nll < best:
            best = nll
            torch.save(
                {"state": model.state_dict(), "selection": row},
                root / "checkpoint.pt",
            )
        print(row, flush=True)
        artifacts.write_report(root / "history.json", {"epochs": history})
    checkpoint = torch.load(
        root / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state"])
    probabilities = predict(
        model,
        data,
        data.validation.ravel(),
    )
    probabilities = probabilities.clip(min=1e-7) ** (
        1 / checkpoint["selection"]["temperature"]
    )
    probabilities /= probabilities.sum(1, keepdims=True)
    probabilities = probabilities.reshape(
        600,
        15,
        10,
    )
    np.save(root / "validation_probabilities.npy", probabilities)
    scores = np.stack(
        [
            np.log(
                (
                    probabilities[:, 10 : 10 + count].mean(1)
                    @ probabilities[:, :10].mean(1).T
                ).clip(min=1e-7)
            )
            for count in evaluation.COUNTS
        ]
    )
    np.save(root / "validation_scores.npy", scores)
    artifacts.write_report(
        root / "completed.json",
        {
            "rank_checkpoint_selection": checkpoint["selection"],
            "style_validation_rank_only": evaluation.summarize(scores),
            "branch": (
                "C2 raw-move ordinal; fresh clean training; " "not full C1+C2"
            ),
            "rank_training_games": len(data.train),
            "rank_calibration_games": len(data.inner),
        },
        emit=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    run()
