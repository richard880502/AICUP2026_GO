"""Extracts per-seed opponent-model probabilities on validation and holdout3."""

import importlib
import pathlib
from typing import Any

import numpy as np
import torch

# Imported by name at run time (same modules as before) so that type checking does not follow the
# legacy research copy, which has its own untyped debt.
legacy_rank_lib: Any = importlib.import_module("runtime.competition2_research.rank_lib")
transformer_rank: Any = importlib.import_module("runtime.competition2_research.transformer_rank")

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def checkpoint_probabilities(
    checkpoint_path: pathlib.Path,
    features: dict[str, np.ndarray],
    keys: np.ndarray,
) -> np.ndarray:
    """Returns one checkpoint's probabilities aligned to requested keys."""
    lookup = {str(key): row for row, key in enumerate(features["game_key"])}
    if set(keys) - set(lookup):
        raise ValueError("Requested game keys are missing from feature cache")
    rows = np.asarray([lookup[str(key)] for key in keys])
    tokens = features["tokens"][rows]
    valid = features["masks"][rows].copy()
    empty = ~valid.any(1)
    valid[empty, 0] = True
    values = transformer_rank.decision_tokens(tokens)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    arguments = checkpoint["args"]
    model = transformer_rank.RankTransformer(
        values.shape[2],
        values.shape[1],
        arguments["d_model"],
        arguments["heads"],
        arguments["ffn"],
        arguments["layers"],
        arguments["dropout"],
        arguments.get("head", "clm"),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    normalized = np.where(
        valid[..., None],
        (values - checkpoint["mean"]) / checkpoint["std"],
        0,
    ).astype(np.float32)
    probabilities = transformer_rank.predict(model, normalized, valid, DEVICE)
    probabilities[empty] = 0.1
    return probabilities


def extract_split(
    root: pathlib.Path,
    fold: int,
    features: dict[str, np.ndarray],
    keys: np.ndarray,
) -> np.ndarray:
    """Returns all three checkpoints for one fixed split and model fold."""
    probabilities = []
    for seed in range(3):
        checkpoint = (
            root
            / "runtime/competition2_research/results"
            / f"opp_scratch_c{fold}_seed{seed}.pt"
        )
        probabilities.append(
            checkpoint_probabilities(checkpoint, features, keys)
        )
        print(
            f"fold={fold} seed={seed} games={len(keys)}",
            flush=True,
        )
    return np.stack(probabilities)


def main() -> None:
    """Extracts all per-seed validation and holdout3 predictions."""
    root = ROOT
    output_root = root / "production/seed_ensemble_20261109"
    if output_root.exists():
        raise FileExistsError(f"Refusing existing output: {output_root}")
    output_root.mkdir(parents=True)
    validation_features = legacy_rank_lib.load_features(
        root / "runtime/competition2_research/features/feat96opp/val"
    )
    holdout_features = legacy_rank_lib.load_features(
        root / "runtime/competition2_research/features/feat96opp/holdout3"
    )
    for fold in (0, 1):
        with np.load(
            root
            / "runtime/competition2_research/results"
            / f"val_probs_d96_c{fold}.npz",
            allow_pickle=True,
        ) as cache:
            validation_keys = np.asarray(cache["game_key"]).astype(str)
        with np.load(
            root
            / "runtime/competition2_research/experiments/crossfit_cohort/results"
            / f"probs_c{fold}.npz",
            allow_pickle=True,
        ) as cache:
            holdout_keys = np.asarray(cache["game_key"]).astype(str)
        validation = extract_split(
            root, fold, validation_features, validation_keys
        )
        holdout = extract_split(root, fold, holdout_features, holdout_keys)
        for split, keys, probabilities in (
            ("val", validation_keys, validation),
            ("holdout3", holdout_keys, holdout),
        ):
            np.savez_compressed(
                output_root / f"{split}_c{fold}.npz",
                game_key=keys,
                probabilities=probabilities,
            )


if __name__ == "__main__":
    main()
