"""Trains open-set game embeddings with candidate/query episodic supervision.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.train \
      --kind sequence --device cuda:0

Validation identities never enter gradient updates or normalization fitting.
The final-holdout scores are never computed by this command.
"""

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import models
from competition1_player_identification.experiments.open_set_v2 import training


def load_data(arguments: argparse.Namespace) -> training.GameData:
    """Loads training-only standardization and maps immutable global indices."""
    root = arguments.root
    if (
        json.loads((root / "split_audit.json").read_text())["status"]
        != "passed"
    ):
        raise ValueError("Content-disjointness gate has not passed")
    selected = np.load(root / "engine_inputs/global_indices.npy")
    arrays = dataset.load_arrays(root / "raw_cache", ("players", "colors"))
    if arguments.kind == "sequence":
        values = dataset.load_arrays(root / "raw_cache", ("tokens",))["tokens"]
        values = values[selected, : arguments.length]
    else:
        values = np.load(root / f"{arguments.kind}_profiles.npy")
    with np.load(root / "clean_splits.npz") as splits:
        train_rows = np.flatnonzero(np.isin(selected, splits["train"]))
        validation = np.searchsorted(selected, splits["validation"])
    _, groups = dataset.player_groups(arrays["players"][selected[train_rows]])
    groups = [train_rows[group] for group in groups]
    labels = np.full(
        len(selected),
        -1,
        np.int64,
    )
    for owner, indices in enumerate(groups):
        labels[indices] = owner
    mean, scale = np.zeros(0, np.float32), np.ones(0, np.float32)
    if arguments.kind != "sequence":
        mean = values[train_rows].mean(0)
        scale = values[train_rows].std(0).clip(min=1e-4)
        values = np.clip(
            (values - mean) / scale,
            -10,
            10,
        ).astype(np.float32)
    return training.GameData(
        values,
        arrays["colors"][selected].astype(np.int64),
        groups,
        labels,
        validation,
        mean,
        scale,
    )


def make_runtime(arguments: argparse.Namespace) -> tuple:
    """Creates the architecture and optimizer after confirming available data."""
    data = load_data(arguments)
    specification = models.Specification(
        arguments.kind,
        data.values.shape[-1],
        len(data.groups),
        arguments.dimension,
        arguments.width,
        arguments.depth,
        arguments.length,
    )
    model = models.Encoder(specification).to(arguments.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=0.01,
    )
    return (
        training.Runtime(
            model,
            optimizer,
            data,
            arguments.device,
            np.random.default_rng(arguments.seed),
            arguments.kind,
        ),
        specification,
    )


def run_epoch(runtime: training.Runtime, steps: int) -> dict:
    """Runs a fixed number of P-by-K episodes and reports distinct-game losses."""
    runtime.model.train()
    history = []
    started = time.monotonic()
    for step in range(steps):
        history.append(training.train_step(runtime))
        if (step + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "steps": steps,
                        "loss": history[-1]["loss"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    return {
        key: float(np.mean([row[key] for row in history])) for key in history[0]
    }


def validate(runtime: training.Runtime) -> tuple[dict, np.ndarray]:
    """Encodes held-out identities and measures all fixed reference aggregators."""
    values = np.reshape(
        training.encode(runtime, runtime.data.validation.ravel()),
        (
            600,
            15,
            -1,
        ),
    )
    scores = evaluation.similarity_scores(values, runtime.device)
    return scores, values


def save_best(
    context: tuple,
    scores: dict,
    values: np.ndarray,
) -> None:
    """Saves sufficient state to reproduce validation and later blind inference."""
    runtime, specification, arguments, selection = context
    torch.save(
        {
            "state": runtime.model.state_dict(),
            "specification": dataclasses.asdict(specification),
            "mean": runtime.data.mean,
            "scale": runtime.data.scale,
            "selection": selection,
            "arguments": vars(arguments),
        },
        arguments.output / "checkpoint.pt",
    )
    np.save(arguments.output / "validation_embeddings.npy", values)
    evaluation.save_scores(arguments.output / "validation_scores.npz", scores)


def fit(arguments: argparse.Namespace) -> None:
    """Selects checkpoints using only the fixed 600-player validation cohort."""
    if (arguments.output / "completed.json").exists():
        raise FileExistsError(
            f"Completed experiment exists: {arguments.output}"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    runtime, specification = make_runtime(arguments)
    history = []
    best = {
        "score": -1.0,
        "epoch": 0,
        "aggregation": "centroid",
    }
    for epoch in range(arguments.epochs):
        row = {"epoch": epoch + 1, **run_epoch(runtime, arguments.steps)}
        scores, values = validate(runtime)
        reports = {
            name: evaluation.summarize(matrix)
            for name, matrix in scores.items()
        }
        row["validation"] = reports
        aggregation = max(
            (report["score"], name) for name, report in reports.items()
        )[1]
        if reports[aggregation]["score"] > best["score"]:
            best = {
                "score": reports[aggregation]["score"],
                "epoch": epoch + 1,
                "aggregation": aggregation,
            }
            save_best(
                (
                    runtime,
                    specification,
                    arguments,
                    best,
                ),
                scores,
                values,
            )
        history.append(row)
        (arguments.output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": row["loss"],
                    "best": best,
                }
            ),
            flush=True,
        )
    (arguments.output / "completed.json").write_text(
        json.dumps(best, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parses experiment settings and bounds CUDA memory use around other jobs."""
    parser = artifacts.root_parser(__doc__, dataset.ROOT)
    parser.add_argument(
        "--kind",
        choices=(
            "opening",
            "engine",
            "sequence",
        ),
        required=True,
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )
    artifacts.add_seed(parser)
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = arguments.root / f"encoder_{arguments.kind}"
    torch.set_num_threads(4)
    torch.manual_seed(arguments.seed)
    if arguments.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.14, arguments.device)
    fit(arguments)


if __name__ == "__main__":
    main()
