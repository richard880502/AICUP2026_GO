"""Encodes dense decision features with the three holdout-safe d96 models.

The three 128-dimensional game representations are concatenated into one
384-dimensional vector. Training combines the existing two-game subset with
the newly extracted games and deduplicates by game key.

Example:
    c1/.venv/bin/python -m \
      experiments.dense_player_bags.extract_game_embeddings --device cuda:0
"""

import argparse
import concurrent.futures
import dataclasses
import json
import pathlib

import numpy as np
import torch

import rank_lib
import transformer_rank


@dataclasses.dataclass(frozen=True)
class EncodingContext:
    """Contains shared encoder state for writing multiple data splits."""

    models: list[
        tuple[transformer_rank.RankTransformer, dict[str, np.ndarray]]
    ]
    output: pathlib.Path
    device: torch.device
    batch_size: int


def select_players(
    features: dict[str, np.ndarray],
    players: set[str],
) -> dict[str, np.ndarray]:
    """Filters feature arrays to selected player IDs."""
    selected = np.asarray([
        player in players
        for player in features["player_id"]
    ])
    return {key: values[selected] for key, values in features.items()}


def merge_training_features(
    base_directory: pathlib.Path,
    extra_directory: pathlib.Path,
    players_path: pathlib.Path,
) -> dict[str, np.ndarray]:
    """Combines old and new shards, filters players, and removes duplicates."""
    selected_players = set(
        players_path.read_text(encoding="utf-8").split(),
    )
    base = select_players(
        rank_lib.load_features(base_directory),
        selected_players,
    )
    extra = select_players(
        rank_lib.load_features(extra_directory),
        selected_players,
    )
    combined = {
        key: np.concatenate((base[key], extra[key]))
        for key in base
    }
    _, unique_rows = np.unique(combined["game_key"], return_index=True)
    unique_rows.sort()
    result = {key: values[unique_rows] for key, values in combined.items()}
    counts = np.unique(result["player_id"], return_counts=True)[1]
    if len(counts) != len(selected_players) or set(counts.tolist()) != {20}:
        raise ValueError(
            "Merged dense features do not contain exactly 20 games per player",
        )
    return result


def load_models(
    results_directory: pathlib.Path,
    input_dimensions: int,
    tokens_per_game: int,
    device: torch.device,
) -> list[tuple[transformer_rank.RankTransformer, dict[str, np.ndarray]]]:
    """Loads the three d96_h encoders and their normalization statistics."""
    models = []
    for path in sorted(results_directory.glob("d96_h_seed*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        arguments = checkpoint["args"]
        model = transformer_rank.RankTransformer(
            input_dimensions,
            tokens_per_game,
            arguments["d_model"],
            arguments["heads"],
            arguments["ffn"],
            arguments["layers"],
            arguments["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        models.append(
            (
                model,
                {"mean": checkpoint["mean"], "std": checkpoint["std"]},
            ),
        )
    if len(models) != 3:
        raise ValueError(f"Expected three d96_h encoders, found {len(models)}")
    return models


def encode_features(
    features: dict[str, np.ndarray],
    models: list[tuple[transformer_rank.RankTransformer, dict[str, np.ndarray]]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Returns concatenated frozen game embeddings for one feature split."""
    tokens = transformer_rank.decision_tokens(features["tokens"])
    valid = features["masks"]
    if device.type == "cpu" and len(models) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(models),
        ) as executor:
            futures = [
                executor.submit(
                    encode_one_model,
                    tokens,
                    valid,
                    model_spec,
                    device,
                    batch_size,
                )
                for model_spec in models
            ]
            model_embeddings = [future.result() for future in futures]
    else:
        model_embeddings = [
            encode_one_model(
                tokens,
                valid,
                model_spec,
                device,
                batch_size,
            )
            for model_spec in models
        ]
    return np.concatenate(model_embeddings, axis=1).astype(np.float16)


def encode_one_model(
    tokens: np.ndarray,
    valid: np.ndarray,
    model_spec: tuple[
        transformer_rank.RankTransformer,
        dict[str, np.ndarray],
    ],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encodes one split with one independently frozen game encoder."""
    model, normalization = model_spec
    standardized = np.where(
        valid[..., None],
        (tokens - normalization["mean"]) / normalization["std"],
        0,
    ).astype(np.float32)
    parts = []
    with torch.no_grad(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for start in range(0, len(tokens), batch_size):
            token_tensor = torch.from_numpy(
                standardized[start:start + batch_size],
            ).to(device)
            valid_tensor = torch.from_numpy(
                valid[start:start + batch_size],
            ).to(device)
            parts.append(
                model.encode(token_tensor, valid_tensor).cpu().numpy(),
            )
    return np.concatenate(parts)


def save_split(
    name: str,
    features: dict[str, np.ndarray],
    context: EncodingContext,
) -> dict[str, int]:
    """Encodes and saves one aligned feature split."""
    embeddings = encode_features(
        features,
        context.models,
        context.device,
        context.batch_size,
    )
    np.savez_compressed(
        context.output / f"{name}.npz",
        embeddings=embeddings,
        player_id=features["player_id"],
        rank=features["rank"],
        game_key=features["game_key"],
    )
    print(f"saved {name}: {embeddings.shape}", flush=True)
    return {"games": len(embeddings), "dimensions": embeddings.shape[1]}


def main() -> None:
    """Builds train, validation, and holdout dense game embeddings."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--base-train",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "train",
    )
    parser.add_argument(
        "--extra-train",
        type=pathlib.Path,
        default=experiment / "features",
    )
    parser.add_argument(
        "--players",
        type=pathlib.Path,
        default=experiment / "shards" / "players.txt",
    )
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "val",
    )
    parser.add_argument(
        "--holdout",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat96" / "holdout",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=rank_lib.ROOT / "results",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--output", type=pathlib.Path, default=experiment / "embeddings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    training = merge_training_features(
        args.base_train,
        args.extra_train,
        args.players,
    )
    validation = rank_lib.load_features(args.validation)
    holdout = rank_lib.load_features(args.holdout)
    input_dimensions = transformer_rank.decision_tokens(
        training["tokens"][:1],
    ).shape[2]
    models = load_models(
        args.results,
        input_dimensions,
        training["tokens"].shape[1],
        device,
    )
    context = EncodingContext(models, args.output, device, args.batch)
    manifest = {
        "train": save_split(
            "train",
            training,
            context,
        ),
        "validation": save_split(
            "validation",
            validation,
            context,
        ),
        "holdout": save_split(
            "holdout",
            holdout,
            context,
        ),
        "encoders": 3,
        "holdout_safe": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
