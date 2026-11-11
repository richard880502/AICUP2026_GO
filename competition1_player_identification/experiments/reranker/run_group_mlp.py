"""Trains a group-softmax neural reranker on frozen pair features.

The ArcFace encoders and Stage-1 retrieval remain frozen.  Synthetic queries
come only from the candidate library, and query owners are player-disjoint
between training and validation.  Model selection and blend-weight selection
use only candidate-only validation; the independent cohort is constructed
and evaluated once after all choices are frozen.

Example:
    .venv/bin/python -m experiments.reranker.run_group_mlp
"""

import argparse
import copy
import dataclasses
import json
import pathlib
import typing

import numpy as np
import torch

from competition1_player_identification.experiments.reranker import run_experiment


DEFAULT_OUTPUT = pathlib.Path(
    "experiments/reranker/results/group_softmax_mlp.json",
)


class GroupReranker(torch.nn.Module):
    """Scores every candidate row with a small shared MLP."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_count, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns one ranking logit for each query-candidate row."""
        return self.network(features).squeeze(-1)


@dataclasses.dataclass(frozen=True)
class GroupTensors:
    """Stores normalized, fixed-size query groups."""

    features: torch.Tensor
    targets: torch.Tensor
    group_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class Selection:
    """Stores the candidate-only selected model and settings."""

    state: dict[str, torch.Tensor]
    epoch: int
    weight: float
    score: float
    history: list[dict[str, float]]


@dataclasses.dataclass(frozen=True)
class PreparedData:
    """Stores frozen data, development pairs, and normalization state."""

    embeddings: run_experiment.EmbeddingData
    blend_weights: dict[int, float]
    development: run_experiment.DevelopmentData
    mean: np.ndarray
    standard_deviation: np.ndarray
    train_groups: GroupTensors
    validation_groups: GroupTensors


def parse_arguments() -> argparse.Namespace:
    """Parses frozen-cache, development, and optimization settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=run_experiment.DEFAULT_EMBEDDINGS,
    )
    parser.add_argument(
        "--selection",
        type=pathlib.Path,
        default=run_experiment.DEFAULT_SELECTION,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--samples-per-count", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-groups", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def _group_arrays(
    dataset: run_experiment.PairDataset,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_ids = np.unique(dataset.group_ids)
    features = np.empty(
        (len(group_ids), top_k, dataset.features.shape[1]),
        dtype=np.float32,
    )
    targets: np.ndarray = np.full(len(group_ids), -1, dtype=np.int64)
    for output_index, group_id in enumerate(group_ids):
        row_indices = np.flatnonzero(dataset.group_ids == group_id)
        if len(row_indices) != top_k:
            raise ValueError(
                f"Group {group_id} has {len(row_indices)} rows, expected "
                f"{top_k}",
            )
        features[output_index] = dataset.features[row_indices]
        positives = np.flatnonzero(dataset.labels[row_indices] == 1)
        if len(positives) == 1:
            targets[output_index] = int(positives[0])
    return features, targets, group_ids


def _normalization(
    train_pairs: run_experiment.PairDataset,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_pairs.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = train_pairs.features.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    return mean, np.maximum(standard_deviation, 1e-5)


def _group_tensors(
    dataset: run_experiment.PairDataset,
    top_k: int,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    device: torch.device,
) -> GroupTensors:
    features, targets, group_ids = _group_arrays(dataset, top_k)
    normalized = (features - mean[None, None, :]) / standard_deviation[
        None,
        None,
        :
    ]
    return GroupTensors(
        torch.from_numpy(normalized).to(device),
        torch.from_numpy(targets).to(device),
        group_ids,
    )


def _score_rows(
    model: GroupReranker,
    groups: GroupTensors,
    row_count: int,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        group_scores = model(groups.features).cpu().numpy()
    scores: np.ndarray = np.zeros(row_count, dtype=np.float32)
    for group_index, group_id in enumerate(groups.group_ids):
        start = int(group_id) * group_scores.shape[1]
        stop = start + group_scores.shape[1]
        scores[start:stop] = group_scores[group_index]
    return scores


def _reranked_ranks(
    row_scores: np.ndarray,
    dataset: run_experiment.PairDataset,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    stage1_ranks = _stage1_ranks(dataset)
    reranked = stage1_ranks.copy()
    owners = _task_owners(dataset)
    for group_id, owner in enumerate(owners):
        row_indices = np.flatnonzero(dataset.group_ids == group_id)
        candidates = dataset.candidate_ids[row_indices]
        if owner not in candidates:
            continue
        neural_scores = _standardize_scores(
            row_scores[row_indices],
        )
        stage1_scores = _standardize_scores(
            dataset.stage1_scores[group_id][candidates],
        )
        blended = weight * neural_scores + (1.0 - weight) * stage1_scores
        position = np.flatnonzero(
            candidates[np.argsort(-blended, kind="stable")] == owner,
        )
        reranked[group_id] = int(position[0]) + 1
    return stage1_ranks, reranked


def _evaluate_scores(
    row_scores: np.ndarray,
    dataset: run_experiment.PairDataset,
    weight: float,
) -> dict[str, object]:
    stage1_ranks, reranked = _reranked_ranks(row_scores, dataset, weight)
    reranker_metrics = _metrics_by_count(reranked, dataset)
    stage1_metrics = _metrics_by_count(stage1_ranks, dataset)
    return {
        "reranker": reranker_metrics,
        "stage1": stage1_metrics,
        "reranker_proxy": float(np.mean([
            reranker_metrics[str(count)]["top5_exponential_score"]
            for count in reranker_metrics
        ])),
        "stage1_proxy": float(np.mean([
            stage1_metrics[str(count)]["top5_exponential_score"]
            for count in stage1_metrics
        ])),
    }


def _standardize_scores(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std().clip(min=1e-6)


def _task_owners(dataset: run_experiment.PairDataset) -> np.ndarray:
    _, first_indices = np.unique(dataset.group_ids, return_index=True)
    return dataset.owners[first_indices]


def _stage1_ranks(dataset: run_experiment.PairDataset) -> np.ndarray:
    return np.asarray([
        int(np.flatnonzero(order == owner)[0]) + 1
        for order, owner in zip(dataset.stage1_orders, _task_owners(dataset))
    ])


def _competition_metrics(ranks: np.ndarray) -> dict[str, float]:
    scores = np.where(ranks <= 5, np.exp(-(ranks - 1)), 0.0)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top5_exponential_score": float(np.mean(scores)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def _metrics_by_count(
    ranks: np.ndarray,
    dataset: run_experiment.PairDataset,
) -> dict[str, dict[str, float]]:
    _, first_indices = np.unique(dataset.group_ids, return_index=True)
    task_counts = dataset.counts[first_indices]
    return {
        str(game_count): _competition_metrics(ranks[task_counts == game_count])
        for game_count in np.unique(task_counts)
    }


def _select_weight(
    row_scores: np.ndarray,
    validation_pairs: run_experiment.PairDataset,
) -> tuple[float, float]:
    best_weight = 0.0
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        results = _evaluate_scores(row_scores, validation_pairs, float(weight))
        score = typing.cast(float, results["reranker_proxy"])
        if score > best_score:
            best_weight = float(weight)
            best_score = score
    return best_weight, best_score


def _train_epoch(
    model: GroupReranker,
    optimizer: torch.optim.Optimizer,
    groups: GroupTensors,
    batch_groups: int,
    generator: torch.Generator,
) -> float:
    model.train()
    valid_indices = torch.nonzero(
        groups.targets >= 0,
        as_tuple=False,
    ).flatten()
    order = valid_indices[
        torch.randperm(len(valid_indices), generator=generator, device="cpu").to(
            valid_indices.device,
        )
    ]
    losses = []
    for start in range(0, len(order), batch_groups):
        indices = order[start:start + batch_groups]
        logits = model(groups.features.index_select(0, indices))
        loss = torch.nn.functional.cross_entropy(
            logits,
            groups.targets.index_select(0, indices),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _fit_and_select(
    arguments: argparse.Namespace,
    train_groups: GroupTensors,
    validation_groups: GroupTensors,
    validation_pairs: run_experiment.PairDataset,
) -> tuple[GroupReranker, Selection]:
    model = GroupReranker(train_groups.features.shape[2]).to(
        train_groups.features.device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(arguments.seed)
    best: Selection | None = None
    stale_epochs = 0
    history = []
    for epoch in range(1, arguments.epochs + 1):
        loss = _train_epoch(
            model,
            optimizer,
            train_groups,
            arguments.batch_groups,
            generator,
        )
        validation_scores = _score_rows(
            model,
            validation_groups,
            len(validation_pairs.features),
        )
        weight, score = _select_weight(validation_scores, validation_pairs)
        history.append({
            "epoch": float(epoch),
            "loss": loss,
            "validation_score": score,
            "weight": weight,
        })
        print(
            f"epoch {epoch}/{arguments.epochs} loss={loss:.5f} "
            f"validation={score:.7f} weight={weight:.2f}",
            flush=True,
        )
        if best is None or score > best.score:
            best = Selection(
                copy.deepcopy(model.state_dict()),
                epoch,
                weight,
                score,
                history.copy(),
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= arguments.patience:
            break
    if best is None:
        raise RuntimeError("No valid group-softmax checkpoint was selected")
    best = dataclasses.replace(best, history=history)
    model.load_state_dict(best.state)
    return model, best


@dataclasses.dataclass
class ProductionFit:
    """Stores a fitted reranker and its candidate-only normalization."""

    model: GroupReranker
    selection: Selection
    mean: np.ndarray
    standard_deviation: np.ndarray


def fit_production_pairs(
    training: run_experiment.PairDataset,
    validation: run_experiment.PairDataset,
    device: torch.device,
) -> ProductionFit:
    """Fits synthetic candidate tasks with the established group-softmax recipe."""
    torch.manual_seed(20261004)
    mean, deviation = _normalization(training)
    arguments = argparse.Namespace(
        learning_rate=1e-3,
        weight_decay=1e-3,
        seed=20261004,
        epochs=120,
        batch_groups=256,
        patience=20,
    )
    model, selection = _fit_and_select(
        arguments,
        _group_tensors(training, 20, mean, deviation, device),
        _group_tensors(validation, 20, mean, deviation, device),
        validation,
    )
    return ProductionFit(model, selection, mean, deviation)


def predict_production_pairs(
    fitted: ProductionFit,
    dataset: run_experiment.PairDataset,
    device: torch.device,
) -> np.ndarray:
    """Predicts unlabeled candidate-pair logits without computing label metrics."""
    return _score_rows(
        fitted.model,
        _group_tensors(dataset, 20, fitted.mean,
                       fitted.standard_deviation, device),
        len(dataset.features),
    )


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.top_k < 5 or arguments.top_k > 50:
        raise ValueError("top-k must be between 5 and 50")
    if arguments.samples_per_count < 1:
        raise ValueError("samples-per-count must be positive")
    if arguments.epochs < 1 or arguments.patience < 1:
        raise ValueError("epochs and patience must be positive")


def _load_blend_weights(path: pathlib.Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    weights = payload["selected_second_dimension_weights"]
    return {int(key): float(value) for key, value in weights.items()}


def _pair_config(
    arguments: argparse.Namespace,
    blend_weights: dict[int, float],
    independent: bool,
) -> run_experiment.PairBuildConfig:
    return run_experiment.PairBuildConfig(
        arguments.top_k,
        blend_weights,
        independent,
        torch.device(arguments.device),
    )


def _development_data(
    arguments: argparse.Namespace,
    data: run_experiment.EmbeddingData,
    blend_weights: dict[int, float],
) -> run_experiment.DevelopmentData:
    generator = np.random.default_rng(arguments.seed)
    player_order = generator.permutation(len(data.players))
    validation_count = round(len(player_order) * arguments.validation_fraction)
    validation_owners = np.sort(player_order[:validation_count])
    train_owners = np.sort(player_order[validation_count:])
    train_tasks = run_experiment.make_tasks(
        train_owners,
        arguments.samples_per_count,
        data.development_holdout_positions,
        generator,
    )
    validation_tasks = run_experiment.make_tasks(
        validation_owners,
        arguments.samples_per_count,
        data.development_holdout_positions,
        generator,
    )
    config = _pair_config(arguments, blend_weights, False)
    return run_experiment.DevelopmentData(
        train_owners,
        validation_owners,
        run_experiment.build_pair_dataset(data, train_tasks, config),
        run_experiment.build_pair_dataset(data, validation_tasks, config),
    )


def _prepare(arguments: argparse.Namespace) -> PreparedData:
    data = run_experiment.load_embeddings(arguments.embeddings)
    blend_weights = _load_blend_weights(arguments.selection)
    development = _development_data(arguments, data, blend_weights)
    mean, standard_deviation = _normalization(development.train_pairs)
    device = torch.device(arguments.device)
    return PreparedData(
        data,
        blend_weights,
        development,
        mean,
        standard_deviation,
        _group_tensors(
            development.train_pairs,
            arguments.top_k,
            mean,
            standard_deviation,
            device,
        ),
        _group_tensors(
            development.validation_pairs,
            arguments.top_k,
            mean,
            standard_deviation,
            device,
        ),
    )


def _independent_tasks(player_count: int) -> list[run_experiment.Task]:
    tasks = []
    for owner in range(player_count):
        for game_count in run_experiment.QUERY_COUNTS:
            tasks.append(run_experiment.Task(
                owner,
                game_count,
                np.arange(game_count),
            ))
    return tasks


def _independent_results(
    arguments: argparse.Namespace,
    prepared: PreparedData,
    model: GroupReranker,
    weight: float,
) -> dict[str, object]:
    independent_pairs = run_experiment.build_pair_dataset(
        prepared.embeddings,
        _independent_tasks(len(prepared.embeddings.players)),
        _pair_config(arguments, prepared.blend_weights, True),
    )
    independent_groups = _group_tensors(
        independent_pairs,
        arguments.top_k,
        prepared.mean,
        prepared.standard_deviation,
        torch.device(arguments.device),
    )
    independent_scores = _score_rows(
        model,
        independent_groups,
        len(independent_pairs.features),
    )
    return _evaluate_scores(independent_scores, independent_pairs, weight)


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    prepared = _prepare(arguments)
    model, selection = _fit_and_select(
        arguments,
        prepared.train_groups,
        prepared.validation_groups,
        prepared.development.validation_pairs,
    )
    validation_scores = _score_rows(
        model,
        prepared.validation_groups,
        len(prepared.development.validation_pairs.features),
    )
    validation_results = _evaluate_scores(
        validation_scores,
        prepared.development.validation_pairs,
        selection.weight,
    )
    return {
        "method": "candidate_only_group_softmax_mlp",
        "uses_independent_query_labels_for_training_or_selection": False,
        "independent_evaluations": 1,
        "seed": arguments.seed,
        "top_k": arguments.top_k,
        "samples_per_count": arguments.samples_per_count,
        "selected_epoch": selection.epoch,
        "selected_weight": selection.weight,
        "validation": validation_results,
        "independent": _independent_results(
            arguments,
            prepared,
            model,
            selection.weight,
        ),
        "history": selection.history,
        "feature_names": run_experiment.FEATURE_NAMES,
        "normalization_mean": prepared.mean.tolist(),
        "normalization_standard_deviation": (
            prepared.standard_deviation.tolist()
        ),
    }


def main() -> None:
    """Runs candidate-only model selection and one independent evaluation."""
    arguments = parse_arguments()
    _validate_arguments(arguments)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    payload = _run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
