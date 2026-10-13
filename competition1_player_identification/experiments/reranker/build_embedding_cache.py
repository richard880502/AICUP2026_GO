"""Rebuilds and caches frozen ArcFace game embeddings for reranking.

The strongest cross-dimensional result stores deterministic seeds and selected
epoch counts rather than model checkpoints. This command rebuilds both the
80-game development refits and the final 100-game refits, then materializes
per-game embeddings. Independent query embeddings are cached for one final
evaluation, but their labels are never used for fitting or model selection.

Example:
    .venv/bin/python -m experiments.reranker.build_embedding_cache \
      --device cuda:1
"""

import argparse
import dataclasses
import json
import pathlib
import typing

import numpy as np
import torch

import train_profile_metric


DEFAULT_PROFILES = pathlib.Path(
    "features/best/humansl_multirank_s64/profiles.npz",
)
DEFAULT_SELECTION = pathlib.Path(
    "results/best/arcface_cross_dimension_ensemble.json",
)
DEFAULT_OUTPUT = pathlib.Path(
    "experiments/reranker/artifacts/frozen_game_embeddings.npz",
)


@dataclasses.dataclass(frozen=True)
class FitData:
    """Stores tensors used by one deterministic ArcFace refit."""

    training_candidates: torch.Tensor
    inference_candidates: torch.Tensor
    queries: torch.Tensor
    training_classes: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FitConfig:
    """Stores model settings for one deterministic ArcFace refit."""

    class_count: int
    embedding_dimension: int
    seed: int
    epochs: int


@dataclasses.dataclass(frozen=True)
class PreparedData:
    """Stores development and final tensors plus cache metadata."""

    development_fit: FitData
    final_fit: FitData
    metadata: dict[str, np.ndarray]


def parse_arguments() -> argparse.Namespace:
    """Parses source, output, and device settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=pathlib.Path, default=DEFAULT_PROFILES)
    parser.add_argument("--selection", type=pathlib.Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def _load_selection(path: pathlib.Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if payload.get("method") != "candidate_only_cross_dimension_arcface_ensemble":
        raise ValueError(f"Unexpected selection method in {path}")
    return payload


def _group_queries(data: train_profile_metric.ProfileData) -> np.ndarray:
    groups = []
    for player in data.players:
        profiles = data.queries[data.query_labels == player]
        if len(profiles) < 20:
            raise ValueError(f"Player {player!r} has fewer than 20 query games")
        groups.append(profiles[:20])
    return np.stack(groups)


def _fit_embeddings(
    fit_data: FitData,
    fit_config: FitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(fit_config.seed)
    config = train_profile_metric.MetricConfig(
        epochs=fit_config.epochs,
        batch_size=1024,
        learning_rate=0.001,
        weight_decay=0.0001,
        embedding_dimension=fit_config.embedding_dimension,
        margin=0.15,
        scale=24.0,
    )
    model = train_profile_metric.ArcFaceModel(
        fit_data.training_candidates.shape[1],
        fit_config.embedding_dimension,
        fit_config.class_count,
    ).to(fit_data.training_candidates.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(fit_config.epochs):
        train_profile_metric.run_epoch(
            model,
            optimizer,
            fit_data.training_candidates,
            fit_data.training_classes,
            config,
        )
        print(
            f"dimension={fit_config.embedding_dimension} "
            f"seed={fit_config.seed} "
            f"epoch={epoch + 1}/{fit_config.epochs}",
            flush=True,
        )
    model.eval()
    with torch.no_grad():
        candidate_embeddings = model.embeddings(
            fit_data.inference_candidates,
        ).cpu().numpy()
        flat_queries = fit_data.queries.reshape(-1, fit_data.queries.shape[-1])
        query_embeddings = model.embeddings(flat_queries).reshape(
            fit_data.queries.shape[0],
            fit_data.queries.shape[1],
            fit_config.embedding_dimension,
        ).cpu().numpy()
    return candidate_embeddings, query_embeddings


def _local_positions(data: train_profile_metric.ProfileData) -> np.ndarray:
    positions = np.empty(len(data.candidate_labels), dtype=np.int64)
    for player in data.players:
        player_indices = np.flatnonzero(data.candidate_labels == player)
        positions[player_indices] = np.arange(len(player_indices))
    return positions


def _cache_metadata(
    data: train_profile_metric.ProfileData,
    split: train_profile_metric.CandidateSplit,
) -> dict[str, np.ndarray]:
    positions = _local_positions(data)
    return {
        "candidate_labels": data.candidate_labels,
        "query_labels": np.repeat(data.players, 20),
        "players": data.players,
        "development_reference_positions": positions[
            split.train_indices
        ].reshape(len(data.players), -1),
        "development_holdout_positions": positions[
            split.validation_indices
        ],
    }


def _prepare_data(
    data: train_profile_metric.ProfileData,
    split_seed: int,
    device: torch.device,
) -> PreparedData:
    split = train_profile_metric.make_candidate_split(
        data,
        validation_games=20,
        seed=split_seed,
    )
    development_candidates, _, _ = train_profile_metric.standardize(
        data.candidates[split.train_indices],
        data.candidates,
    )
    final_candidates, final_means, final_scales = train_profile_metric.standardize(
        data.candidates,
        data.candidates,
    )
    final_queries = np.nan_to_num(
        (_group_queries(data) - final_means) / final_scales,
    ).astype(np.float32)
    all_classes = train_profile_metric.make_class_ids(
        data.candidate_labels,
        data.players,
    )
    development_tensor = torch.from_numpy(development_candidates).to(device)
    final_tensor = torch.from_numpy(final_candidates).to(device)
    development_fit = FitData(
        development_tensor[split.train_indices],
        development_tensor,
        development_tensor[:1, None, :],
        torch.from_numpy(all_classes[split.train_indices]).to(device),
    )
    final_fit = FitData(
        final_tensor,
        final_tensor,
        torch.from_numpy(final_queries).to(device),
        torch.from_numpy(all_classes).to(device),
    )
    return PreparedData(
        development_fit,
        final_fit,
        _cache_metadata(data, split),
    )


def _fit_dimension(
    dimension_payload: dict[str, object],
    prepared: PreparedData,
    player_count: int,
) -> dict[str, np.ndarray]:
    dimension = typing.cast(int, dimension_payload["embedding_dimension"])
    development_models = []
    final_candidate_models = []
    final_query_models = []
    configs = []
    selected_models = typing.cast(
        list[dict[str, object]],
        dimension_payload["selected_models"],
    )
    for model_payload in selected_models:
        config = FitConfig(
            player_count,
            dimension,
            typing.cast(int, model_payload["seed"]),
            typing.cast(int, model_payload["selected_epoch"]),
        )
        development_embedding, _ = _fit_embeddings(
            prepared.development_fit,
            config,
        )
        print(
            f"refitting final dimension={dimension} seed={config.seed}",
            flush=True,
        )
        final_candidate_embedding, final_query_embedding = _fit_embeddings(
            prepared.final_fit,
            config,
        )
        development_models.append(development_embedding)
        final_candidate_models.append(final_candidate_embedding)
        final_query_models.append(final_query_embedding)
        configs.append(config)
    return {
        f"development_candidate_{dimension}": np.stack(development_models),
        f"final_candidate_{dimension}": np.stack(final_candidate_models),
        f"final_query_{dimension}": np.stack(final_query_models),
        f"seeds_{dimension}": np.asarray(
            [config.seed for config in configs],
            dtype=np.int64,
        ),
        f"epochs_{dimension}": np.asarray(
            [config.epochs for config in configs],
            dtype=np.int64,
        ),
    }


def _build_output(
    selection: dict[str, object],
    prepared: PreparedData,
    player_count: int,
) -> dict[str, np.ndarray]:
    output = dict(prepared.metadata)
    dimensions = selection.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("Selection JSON has no dimensions list")
    for dimension_payload in dimensions:
        output.update(_fit_dimension(
            typing.cast(dict[str, object], dimension_payload),
            prepared,
            player_count,
        ))
    return output


def main() -> None:
    """Refits selected models and saves all per-game embeddings."""
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selection = _load_selection(arguments.selection)
    data = train_profile_metric.load_profiles(arguments.profiles)
    prepared = _prepare_data(
        data,
        typing.cast(int, selection["split_seed"]),
        device,
    )
    output = _build_output(selection, prepared, len(data.players))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # NumPy's overload incorrectly treats arbitrary named arrays as bools.
    np.savez(arguments.output, **output)  # type: ignore[arg-type]
    print(f"saved {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
