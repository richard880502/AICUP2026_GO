"""Runs a cached, within-half cross-validation pilot of frozen rank experts.

Uses existing player-disjoint C1+C2 predictions without rerunning KataGo.
Example, from the repository root:
    runtime/competition1_research/.venv/bin/python -m \
        competition2_rank_prediction.experiments.rank_experts.run
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import pathlib
import time

import numpy as np
import torch


LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
REGIONS = ((0, 1, 2, 3), (2, 3, 4, 5, 6), (5, 6, 7, 8, 9))
BOUNDARIES = ((3, 4), (4, 5), (7, 8))
KINDS = ("bias", "linear", "region", "boundary")
SEED = 20261106


@dataclasses.dataclass(frozen=True)
class PilotData:
    """Stores aligned, audited question-level arrays."""

    probabilities: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    halves: np.ndarray
    counts: np.ndarray
    players: np.ndarray
    audit: dict[str, object]


@dataclasses.dataclass(frozen=True)
class FoldData:
    """Stores training-standardized tensors for one expert fold."""

    train_features: torch.Tensor
    test_features: torch.Tensor
    train_logits: torch.Tensor
    test_logits: torch.Tensor
    train_labels: torch.Tensor


class ExpertModel(torch.nn.Module):
    """Produces bounded corrections to a frozen ten-rank distribution."""

    def __init__(self, dimension: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "bias":
            self.networks = torch.nn.ModuleList([
                torch.nn.Linear(1, 10),
            ])
        elif kind == "linear":
            self.networks = torch.nn.ModuleList([
                torch.nn.Linear(dimension, 10),
            ])
        else:
            outputs = 1 if kind == "boundary" else 10
            self.networks = torch.nn.ModuleList([
                _network(dimension, outputs),
                _network(dimension, outputs),
                _network(dimension, outputs),
            ])
        for network in self.networks:
            final_layer = network[-1] if isinstance(
                network, torch.nn.Sequential,
            ) else network
            torch.nn.init.zeros_(final_layer.weight)
            torch.nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits, with routing depending only on frozen probabilities."""
        if self.kind in ("bias", "linear"):
            inputs = features
            if self.kind == "bias":
                inputs = torch.ones_like(features[:, :1])
            return logits + 0.5 * torch.tanh(self.networks[0](inputs))
        probabilities = torch.softmax(logits, dim=1)
        if self.kind == "boundary":
            return self._boundary(logits, features, probabilities)
        masses = torch.stack([
            probabilities[:, region].sum(dim=1)
            for region in REGIONS
        ], dim=1)
        gates = masses / masses.sum(dim=1, keepdim=True)
        correction = torch.zeros_like(logits)
        for index, region in enumerate(REGIONS):
            mask = torch.zeros(10, device=logits.device)
            mask[list(region)] = 1.0
            residual = 0.5 * torch.tanh(self.networks[index](features))
            correction = correction + gates[:, index:index + 1] * mask * residual
        return logits + correction

    def _boundary(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        output = logits
        for index, pair in enumerate(BOUNDARIES):
            positions = torch.tensor(pair, device=logits.device)
            pair_logits = output[:, positions]
            mass = torch.logsumexp(pair_logits, dim=1, keepdim=True)
            gate = probabilities[:, positions].sum(dim=1, keepdim=True)
            delta = 0.5 * torch.tanh(self.networks[index](features)) * gate
            adjustment = torch.cat((-delta / 2, delta / 2), dim=1)
            updated = torch.log_softmax(pair_logits + adjustment, dim=1) + mass
            selector = torch.eye(10, device=logits.device)[positions]
            output = output + (updated - pair_logits).matmul(selector)
        return output


def _network(dimension: int, outputs: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(dimension, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, outputs),
    )


def parse_arguments() -> argparse.Namespace:
    """Parses immutable cache paths and runtime settings."""
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research",
        type=pathlib.Path,
        default=repository / "runtime/competition2_research",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=repository / "production/rank_experts_20261106",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def _load_final(root: pathlib.Path) -> dict[str, np.ndarray]:
    rows = {}
    for path in sorted((root / "features/final_result/holdout3").glob("*.npz")):
        with np.load(path) as payload:
            for key, values in zip(payload["game_key"], payload["values"]):
                rows[str(key)] = values
    return rows


def _question_features(
    member_rows: dict[str, np.ndarray],
    final_values: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    parts = [logits - logits.mean()]
    for name in ("d96", "resnet", "opp"):
        probabilities = member_rows[name]
        parts.extend((probabilities.mean(axis=0), probabilities.std(axis=0)))
    own = member_rows["d96"] @ np.arange(10)
    opponent = member_rows["opp"] @ np.arange(10)
    gap = own - opponent
    parts.append(np.asarray([
        own.mean(), own.std(), opponent.mean(), opponent.std(),
        gap.mean(), gap.std(), np.log(len(own)),
    ]))
    # Scaling constants are fixed; fold-specific standardization is fitted later.
    final_scaled = final_values / np.asarray((1, 60, 300, 1, 1, 9))
    parts.extend((final_scaled.mean(axis=0), final_scaled.std(axis=0)))
    return np.nan_to_num(np.concatenate(parts)).astype(np.float32)


def load_data(root: pathlib.Path) -> PilotData:
    """Loads OOF caches and checks player, half, and game alignment."""
    source = root / "experiments/sgf_structure/results/base_holdout3.npz"
    with np.load(source) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    with (root / "features/holdout3/question_folds.csv").open(
        "r", encoding="utf-8", newline="",
    ) as input_file:
        mappings = {row["question_id"]: row for row in csv.DictReader(input_file)}
    players = np.asarray([
        mappings[str(question)]["player"]
        for question in arrays["question_id"]
    ])
    if len(np.unique(players)) != len(players):
        raise ValueError("Pilot requires exactly one question per player")
    for half in (0, 1):
        excluded = set((root / f"features/holdout3/exclude_c{half}.txt").read_text(
            encoding="utf-8",
        ).split())
        if not set(players[arrays["fold"] == half]).issubset(excluded):
            raise ValueError(f"Half {half} has players missing from exclusion list")
    features = _aligned_features(root, arrays)
    audit = {
        "questions": len(players),
        "unique_players": len(np.unique(players)),
        "label_counts": np.bincount(arrays["y"], minlength=10).tolist(),
        "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocol": "stratified expert CV within each upstream-excluded half",
        "fresh_outer_cohort": False,
    }
    return PilotData(
        arrays["qprob"], features, arrays["y"], arrays["fold"],
        arrays["k"], players, audit,
    )


def _aligned_features(
    root: pathlib.Path,
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    final = _load_final(root)
    features = []
    cache = root / "experiments/sgf_structure/results/game_members_holdout3.npz"
    with np.load(cache) as payload:
        games = {name: np.asarray(payload[name]) for name in payload.files}
    for index, question in enumerate(arrays["question_id"]):
        positions = np.flatnonzero(games["question_id"] == question)
        if len(positions) != arrays["k"][index]:
            raise ValueError(f"Game count mismatch for {question!r}")
        members = {}
        for name in ("d96", "resnet", "opp"):
            members[name] = games[name][positions]
        values = np.stack([
            final[str(key)] for key in games["game_key"][positions]
        ])
        features.append(_question_features(
            members, values, np.log(arrays["qprob"][index].clip(1e-9)),
        ))
    return np.stack(features)


def _fold_ids(data: PilotData) -> np.ndarray:
    generator = np.random.default_rng(SEED)
    folds = np.empty(len(data.labels), dtype=np.int64)
    for half in (0, 1):
        for label in range(10):
            indices = np.flatnonzero((data.halves == half) & (data.labels == label))
            folds[generator.permutation(indices)] = np.arange(len(indices)) % 5
    return folds


def _fold_data(
    data: PilotData,
    training: np.ndarray,
    testing: np.ndarray,
    device: torch.device,
) -> FoldData:
    mean = data.features[training].mean(axis=0)
    scale = data.features[training].std(axis=0).clip(1e-5)
    normalized = (data.features - mean) / scale
    logits = np.log(data.probabilities.clip(1e-9)).astype(np.float32)
    return FoldData(
        torch.tensor(normalized[training], device=device),
        torch.tensor(normalized[testing], device=device),
        torch.tensor(logits[training], device=device),
        torch.tensor(logits[testing], device=device),
        torch.tensor(data.labels[training], device=device, dtype=torch.long),
    )


def fit_predict(
    fold: FoldData,
    kind: str,
    epochs: int,
) -> np.ndarray:
    """Fits one fixed expert recipe and returns held-out probabilities."""
    torch.manual_seed(SEED)
    model = ExpertModel(fold.train_features.shape[1], kind).to(
        fold.train_features.device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=0.01,
    )
    with torch.no_grad():
        zero = model(fold.test_logits, fold.test_features)
        if not torch.allclose(zero, fold.test_logits, atol=2e-6, rtol=0):
            raise RuntimeError(f"Zero residual parity failed for {kind}")
    for _ in range(epochs):
        corrected = model(fold.train_logits, fold.train_features)
        loss = torch.nn.functional.cross_entropy(corrected, fold.train_labels)
        penalty = (corrected - fold.train_logits).square().mean()
        optimizer.zero_grad()
        (loss + 0.01 * penalty).backward()
        optimizer.step()
    with torch.inference_mode():
        return torch.softmax(model(
            fold.test_logits, fold.test_features,
        ), dim=1).cpu().numpy()


def _utilities(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rewards = np.eye(10) + np.exp(-1) * (
        np.eye(10, k=1) + np.eye(10, k=-1)
    )
    predictions = (probabilities @ rewards).argmax(axis=1)
    return rewards[labels, predictions]


def _metrics(probabilities: np.ndarray, data: PilotData) -> dict[str, object]:
    utility = _utilities(probabilities, data.labels)
    rewards = np.eye(10) + np.exp(-1) * (
        np.eye(10, k=1) + np.eye(10, k=-1)
    )
    error = np.abs((probabilities @ rewards).argmax(axis=1) - data.labels)
    aucs = {}
    for low, high in BOUNDARIES:
        odds = probabilities[:, high] / probabilities[:, [low, high]].sum(axis=1)
        differences = odds[data.labels == high, None] - odds[data.labels == low]
        aucs[f"{LEVELS[low]}/{LEVELS[high]}"] = float(np.mean(
            (differences > 0) + 0.5 * (differences == 0),
        ))
    return {
        "utility": float(utility.mean()),
        "exact": float(np.mean(error == 0)),
        "large_error": float(np.mean(error >= 2)),
        "per_rank": {
            name: float(utility[data.labels == index].mean())
            for index, name in enumerate(LEVELS)
        },
        "by_count": {
            f"{low}-{high}": float(utility[
                (data.counts >= low) & (data.counts <= high)
            ].mean())
            for low, high in ((5, 9), (10, 14), (15, 20))
        },
        "boundary_auc": aucs,
    }


def _paired_interval(differences: np.ndarray, repeats: int) -> list[float]:
    generator = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        positions = generator.integers(0, len(differences), len(differences))
        values.append(float(differences[positions].mean()))
    return [float(differences.mean()), *np.quantile(values, (0.025, 0.975)).tolist()]


def _run_cross_validation(
    data: PilotData,
    folds: np.ndarray,
    arguments: argparse.Namespace,
) -> dict[str, np.ndarray]:
    predictions = {kind: np.zeros_like(data.probabilities) for kind in KINDS}
    predictions["baseline"] = data.probabilities
    print(json.dumps(data.audit), flush=True)
    for half in (0, 1):
        for fold_id in range(5):
            testing = (data.halves == half) & (folds == fold_id)
            training = (data.halves == half) & (folds != fold_id)
            fold = _fold_data(data, training, testing, torch.device(arguments.device))
            for kind in KINDS:
                predictions[kind][testing] = fit_predict(fold, kind, arguments.epochs)
                print(f"half={half} fold={fold_id} kind={kind} complete", flush=True)
    return predictions


def _comparisons(
    predictions: dict[str, np.ndarray],
    data: PilotData,
    repeats: int,
) -> dict[str, dict[str, list[float]]]:
    comparisons: dict[str, dict[str, list[float]]] = {}
    for name in KINDS:
        comparisons[name] = {}
        for reference in ("baseline", "bias", "linear"):
            difference = _utilities(predictions[name], data.labels) - _utilities(
                predictions[reference], data.labels,
            )
            comparisons[name][reference] = _paired_interval(
                difference, repeats,
            )
    return comparisons


def main() -> None:
    """Audits caches, runs fixed CV candidates, and writes an exploratory report."""
    arguments = parse_arguments()
    if arguments.epochs < 1 or arguments.bootstrap < 1:
        raise ValueError("epochs and bootstrap must be positive")
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    arguments.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    data = load_data(arguments.research)
    folds = _fold_ids(data)
    predictions = _run_cross_validation(data, folds, arguments)
    report = {
        "status": "complete_exploratory_only",
        "audit": data.audit,
        "epochs": arguments.epochs,
        "seed": SEED,
        "metrics": {
            name: _metrics(values, data)
            for name, values in predictions.items()
        },
        "paired_delta_95ci": _comparisons(predictions, data, arguments.bootstrap),
        "elapsed_seconds": time.monotonic() - started,
        "production_changed": False,
        "requires_fresh_cohort_confirmation": True,
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        arguments.output / "predictions.npz",
        labels=data.labels,
        players=data.players,
        halves=data.halves,
        folds=folds,
        predictions=np.stack(list(predictions.values())),
        model_names=np.asarray(list(predictions)),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
