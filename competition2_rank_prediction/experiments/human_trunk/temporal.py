"""Trains a fixed original-architecture temporal probe on frozen full trunk tokens.

Example: python -m competition2_rank_prediction.experiments.human_trunk.temporal
This follow-up changes the probe, never the pretrained HumanSL backbone.
"""

import argparse
import json

import numpy as np
import sklearn.preprocessing  # type: ignore[import-untyped]
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common


def recipe() -> argparse.Namespace:
    """Returns the original model/budget settings fixed before follow-up outcomes."""
    return argparse.Namespace(
        d_model=128,
        heads=4,
        ffn=384,
        layers=3,
        dropout=0.1,
        head="clm",
        lr=5e-4,
        weight_decay=0.05,
        epochs=40,
        batch=256,
        train_only="all",
        token_dropout=0.15,
        noise=0.05,
        loss="nll",
        select="nll",
        device="cuda:1",
    )


def train_seed(data: tuple, settings: argparse.Namespace, seed: int, directory):
    """Uses original training helpers, retaining every inner metric for auditing."""
    torch.manual_seed(seed)
    rows = transformer_rank.inner_split(data[3], seed)
    model = transformer_rank.build_model(
        data[0].shape[2],
        data[0].shape[1],
        settings,
        torch.device(settings.device),
        None,
    )
    optimizer = transformer_rank.make_optimiser(model, settings, len(rows[0]))
    tensors = tuple(
        torch.from_numpy(part[rows[0]]).to(settings.device) for part in data[:3]
    )
    inner_data = tuple(part[rows[1]] for part in data[:3])
    history: list[float] = []
    best_state = {}
    for epoch in range(settings.epochs):
        transformer_rank.train_epoch(model, optimizer, tensors, settings)
        metric = transformer_rank.inner_metric(
            model, inner_data, settings.select, torch.device(settings.device)
        )
        if not history or metric < min(history):
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        history.append(metric)
        (directory / f"seed_{seed}_history.json").write_text(
            json.dumps(history)
        )
        if (epoch + 1) % 5 == 0:
            print(
                f"seed={seed} epoch={epoch + 1}/{settings.epochs} inner_nll={metric:.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, {
        "best_inner_nll": min(history),
        "best_epoch": int(np.argmin(history)) + 1,
    }


def normalize(
    training: dict, validation: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fits stable channel moments on optimizer players only."""
    rows, inner = transformer_rank.inner_split(training["player_id"], seed)
    if set(training["player_id"][rows]) & set(training["player_id"][inner]):
        raise ValueError("Temporal optimizer/inner player overlap")
    scaler = sklearn.preprocessing.StandardScaler().fit(
        training["tokens"][rows][training["masks"][rows]].astype(np.float32)
    )
    mean, std = scaler.mean_.astype(np.float32), scaler.scale_.astype(
        np.float32
    )
    transformed = []
    for payload in (training, validation):
        transformed.append(
            np.where(
                payload["masks"][..., None],
                (payload["tokens"].astype(np.float32) - mean) / std,
                0,
            ).astype(np.float32)
        )
    return (
        transformed[0],
        transformed[1],
        {
            "mean": mean,
            "std": std,
            "optimizer_players": np.unique(training["player_id"][rows]),
            "inner_players": np.unique(training["player_id"][inner]),
        },
    )


def main() -> None:
    """Trains three seeds selected solely on the inner cohort and exports the mean."""
    args = common.parse_arguments(
        __doc__ or "", "human_trunk_temporal_20261109", True
    )
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporal probe: {args.output}"
        )
    if args.device != "cuda:1":
        raise ValueError("Temporal training uses GPU1 only")
    source = args.research.parents[1] / "production/human_trunk_20261109"
    if not (source / "EXTRACTION_COMPLETE").exists():
        raise RuntimeError("Frozen feature extraction must finish first")
    training = rank_lib.load_features(source / "features/train")
    validation = rank_lib.load_features(source / "features/val")
    if set(training["player_id"]) & set(validation["player_id"]):
        raise ValueError("Temporal training/validation player overlap")
    if (
        not training["masks"].any(axis=1).all()
        or not validation["masks"].any(axis=1).all()
    ):
        raise ValueError(
            "Temporal probe requires usable decisions in every game"
        )
    args.output.mkdir(parents=True)
    (args.output / "recipe.json").write_text(
        json.dumps(vars(recipe()), indent=2)
    )
    torch.set_num_threads(4)
    predictions = []
    records = []
    for seed in range(3):
        train_tokens, val_tokens, normalization = normalize(
            training, validation, seed
        )
        model, info = train_seed(
            (
                train_tokens,
                training["masks"],
                rank_lib.rank_ids(training["rank"]),
                training["player_id"],
            ),
            recipe(),
            seed,
            args.output,
        )
        predictions.append(
            transformer_rank.predict(
                model,
                val_tokens,
                validation["masks"],
                torch.device(args.device),
                batch=256,
            )
        )
        torch.save(
            {
                "state": model.state_dict(),
                "seed": seed,
                "info": info,
                **normalization,
            },
            args.output / f"seed_{seed}.pt",
        )
        records.append({"seed": seed, **info})
        print(json.dumps(records[-1]), flush=True)
    np.savez_compressed(
        args.output / "probe_validation.npz",
        probabilities=np.mean(predictions, axis=0),
        seed_probabilities=np.stack(predictions),
        game_key=validation["game_key"],
        player=validation["player_id"],
        y=rank_lib.rank_ids(validation["rank"]),
    )
    (args.output / "temporal_fit.json").write_text(
        json.dumps(records, indent=2)
    )


if __name__ == "__main__":
    main()
