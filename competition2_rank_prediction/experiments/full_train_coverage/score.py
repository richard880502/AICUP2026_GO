"""Scores the expanded rank encoder on validation and cross-fit queries.

Example:
    python -m competition2_rank_prediction.experiments.full_train_coverage.score \
        --fold 0 --device cuda:0
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error


def project_root() -> pathlib.Path:
    """Returns the repository root for this experiment module."""
    return pathlib.Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    """Parses the fold and inference-device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag-prefix", default="d96_fullcov")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--probability-dir",
        type=pathlib.Path,
        default=None,
    )
    return parser.parse_args()


def checkpoint_probabilities(
    features: dict[str, np.ndarray],
    checkpoints: list[pathlib.Path],
    device: torch.device,
) -> np.ndarray:
    """Averages three trained seeds and assigns empty games uniform output."""
    tokens = transformer_rank.decision_tokens(features["tokens"])
    valid = features["masks"].copy()
    empty = ~valid.any(axis=1)
    valid[empty, 0] = True
    predictions = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        settings = checkpoint["args"]
        model = transformer_rank.RankTransformer(
            tokens.shape[2],
            tokens.shape[1],
            settings["d_model"],
            settings["heads"],
            settings["ffn"],
            settings["layers"],
            settings["dropout"],
            settings.get("head", "clm"),
        ).to(device)
        model.load_state_dict(checkpoint["state"])
        scaled = np.where(
            valid[..., None],
            (tokens - checkpoint["mean"]) / checkpoint["std"],
            0,
        ).astype(np.float32)
        predictions.append(transformer_rank.predict(model, scaled, valid, device))
    probabilities = np.mean(predictions, axis=0)
    probabilities[empty] = 0.1
    return probabilities


def score_split(
    args: argparse.Namespace,
    name: str,
    feature_directory: pathlib.Path,
    device: torch.device,
) -> None:
    """Writes aligned per-game predictions for one data split."""
    features = rank_lib.load_features(feature_directory)
    tag = f"{args.tag_prefix}_c{args.fold}"
    checkpoints = [
        args.checkpoint_dir / f"{tag}_seed{seed}.pt"
        for seed in range(3)
    ]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Three-seed ensemble is incomplete: {missing!r}")
    probabilities = checkpoint_probabilities(
        features,
        checkpoints,
        device,
    )
    payload = {
        "game_key": features["game_key"],
        "player_id": features["player_id"],
        "rank": features["rank"],
        "probabilities": probabilities.astype(np.float32),
    }
    output_path = args.probability_dir / f"c{args.fold}_{name}.npz"
    np.savez_compressed(output_path, **payload)  # type: ignore[arg-type]
    if name == "validation":
        labels = rank_lib.rank_ids(features["rank"])
        accuracy = float((probabilities.argmax(axis=1) == labels).mean())
        print(
            f"fold {args.fold} validation per-game top1 {accuracy:.4f}",
            flush=True,
        )
    print(
        f"saved fold {args.fold} {name}: {len(probabilities)} games",
        flush=True,
    )


def main() -> None:
    """Scores validation and holdout3 with one fold-excluded checkpoint set."""
    args = parse_args()
    root = project_root()
    research = root / "runtime/competition2_research"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = research / "results"
    if args.probability_dir is None:
        args.probability_dir = (
            root / "production/full_train_coverage_20261109/probabilities"
        )
    args.probability_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(8)
    splits = (
        ("validation", research / "features/feat96/val"),
        ("holdout3", research / "features/feat96/holdout3"),
    )
    for name, feature_directory in splits:
        score_split(args, name, feature_directory, device)


if __name__ == "__main__":
    main()
