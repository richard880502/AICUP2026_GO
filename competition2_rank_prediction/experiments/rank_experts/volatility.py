"""Tests the isolated final-winrate volatility hypothesis on frozen OOF data.

Example from the repository root:
    runtime/competition1_research/.venv/bin/python -m \
        competition2_rank_prediction.experiments.rank_experts.volatility

This is a post-selection exploratory test, not independent confirmation.
"""

import argparse
import json

import numpy as np
import torch

from competition2_rank_prediction.experiments.rank_experts import diagnose
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.rank_experts import run


SEED = 20261107
WINRATE_MEAN_COLUMN = 77
WINRATE_STD_COLUMN = 83


class AxisCorrection(torch.nn.Module):
    """Applies a bounded scalar shift along the ten-rank logit axis."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("axis", torch.linspace(-1.5, 1.5, 10))

    def forward(self, logits: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """Returns logits corrected by a single standardized feature."""
        shift = 0.5 * torch.tanh(self.bias + self.weight * feature)
        return logits + shift[:, None] * self.axis


def parse_arguments() -> argparse.Namespace:
    """Parses immutable source cache and new output directory paths."""
    return common.parse_arguments(
        __doc__ or "", "final_winrate_volatility_20261107",
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() < 1e-8 or second.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _nuisance_residuals(
    data: run.PilotData,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = data.features[:, WINRATE_MEAN_COLUMN]
    nuisance = np.column_stack((
        data.features[:, :10], mean, mean ** 2, np.log(data.counts),
    ))
    nuisance_mean = nuisance[train_mask].mean(axis=0)
    nuisance_scale = nuisance[train_mask].std(axis=0).clip(1e-5)
    design = np.column_stack((
        np.ones(len(mean)), (nuisance - nuisance_mean) / nuisance_scale,
    ))
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0
    volatility = data.features[:, WINRATE_STD_COLUMN]
    weights = np.linalg.solve(
        design[train_mask].T @ design[train_mask] + penalty,
        design[train_mask].T @ volatility[train_mask],
    )
    reconstructed = design @ weights
    return volatility - reconstructed, reconstructed


def _fit_shift(
    data: run.PilotData,
    feature: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    standardized = (
        (feature - feature[train_mask].mean())
        / max(float(feature[train_mask].std()), 1e-5)
    ).astype(np.float32)
    logits = torch.tensor(np.log(data.probabilities.clip(1e-9)), dtype=torch.float32)
    inputs = torch.tensor(standardized)
    labels = torch.tensor(data.labels[train_mask], dtype=torch.long)
    torch.manual_seed(SEED)
    model = AxisCorrection()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    with torch.no_grad():
        if not torch.equal(model(logits, inputs), logits):
            raise RuntimeError("Zero-shift parity failed")
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(logits[train_mask], inputs[train_mask]), labels,
        ) + 0.1 * model.weight.square()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probabilities = torch.softmax(model(logits, inputs), dim=1).numpy()
    return probabilities, {
        "weight": float(model.state_dict()["weight"]),
        "bias": float(model.state_dict()["bias"]),
        "feature_train_mean": float(feature[train_mask].mean()),
        "feature_train_std": float(feature[train_mask].std()),
    }


def _cross_validation(
    data: run.PilotData,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], np.ndarray]:
    predictions = {
        name: np.zeros_like(data.probabilities)
        for name in ("axis_bias", "raw_volatility", "conditional_volatility")
    }
    conditional = np.zeros(len(data.labels))
    records: list[dict[str, object]] = []
    for half in (0, 1):
        for fold in range(5):
            training = (data.halves == half) & (folds != fold)
            testing = (data.halves == half) & (folds == fold)
            residuals, _ = _nuisance_residuals(data, training)
            conditional[testing] = residuals[testing]
            features = {
                "axis_bias": np.zeros(len(data.labels)),
                "raw_volatility": data.features[:, WINRATE_STD_COLUMN],
                "conditional_volatility": residuals,
            }
            for name, feature in features.items():
                probabilities, fitted = _fit_shift(data, feature, training)
                predictions[name][testing] = probabilities[testing]
                records.append({
                    "half": half, "fold": fold, "method": name,
                    "parameters": fitted,
                })
    predictions["baseline"] = data.probabilities
    return predictions, records, conditional


def _point_losses(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    decoded = (probabilities @ diagnose.REWARDS).argmax(axis=1)
    return {
        "utility": diagnose.REWARDS[labels, decoded],
        "nll": -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)),
        "brier": np.square(probabilities - np.eye(10)[labels]).sum(axis=1),
        "large_error": (np.abs(decoded - labels) >= 2).astype(float),
    }


def _bootstrap_means(values: np.ndarray) -> list[float]:
    generator = np.random.default_rng(SEED)
    means = []
    for _ in range(5000):
        indices = generator.integers(0, len(values), len(values))
        means.append(float(values[indices].mean()))
    return [float(values.mean()), *np.quantile(means, (0.025, 0.975)).tolist()]


def _comparisons(
    predictions: dict[str, np.ndarray],
    data: run.PilotData,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for candidate in ("raw_volatility", "conditional_volatility"):
        outcomes = _point_losses(predictions[candidate], data.labels)
        references = {}
        for reference in ("baseline", "axis_bias"):
            original = _point_losses(predictions[reference], data.labels)
            references[reference] = {
                metric: _bootstrap_means(values - original[metric])
                for metric, values in outcomes.items()
            }
        results[candidate] = references
    return results


def _associations(data: run.PilotData, conditional: np.ndarray) -> dict[str, object]:
    rank_error = data.labels - data.probabilities @ np.arange(10)
    volatility = data.features[:, WINRATE_STD_COLUMN]
    mean = data.features[:, WINRATE_MEAN_COLUMN]
    # E[w(1-w)] follows exactly from the first two moments (population std).
    undecidedness = mean * (1 - mean) - volatility ** 2
    groups = {"all": np.ones(len(mean), dtype=bool)}
    for half in (0, 1):
        groups[f"half_{half}"] = data.halves == half
    results: dict[str, object] = {}
    for name, selected in groups.items():
        target = rank_error[selected]
        total = np.square(volatility[selected] - volatility[selected].mean()).sum()
        results[name] = {
            "players": int(selected.sum()),
            "raw_volatility_rank_error_r": _correlation(volatility[selected], target),
            "conditional_volatility_rank_error_r": _correlation(conditional[selected], target),
            "outcome_balance_rank_error_r": _correlation(
                (mean * (1 - mean))[selected], target,
            ),
            "terminal_undecidedness_rank_error_r": _correlation(undecidedness[selected], target),
            "nuisance_oof_r2": float(1 - np.square(conditional[selected]).sum() / total),
        }
    return results


def _subgroup_deltas(
    predictions: dict[str, np.ndarray],
    data: run.PilotData,
) -> dict[str, object]:
    groups = common.subgroup_masks(data)
    results: dict[str, object] = {}
    baseline = _point_losses(predictions["baseline"], data.labels)["utility"]
    for name, selected in groups.items():
        deltas = {}
        for method, values in predictions.items():
            utility = _point_losses(values, data.labels)["utility"]
            deltas[method] = float((utility - baseline)[selected].mean())
        results[name] = {"players": int(selected.sum()), "utility_delta": deltas}
    return results


def _gate(
    comparisons: dict[str, object],
    predictions: dict[str, np.ndarray],
    data: run.PilotData,
) -> dict[str, object]:
    # The gate is deliberately derived from OOF predictions, not training fit.
    primary = _point_losses(predictions["conditional_volatility"], data.labels)
    baseline = _point_losses(predictions["baseline"], data.labels)
    bias = _point_losses(predictions["axis_bias"], data.labels)
    vs_baseline = _bootstrap_means(primary["utility"] - baseline["utility"])
    vs_bias = _bootstrap_means(primary["utility"] - bias["utility"])
    half_deltas = []
    for half in (0, 1):
        selected = data.halves == half
        half_deltas.append(float((primary["utility"] - baseline["utility"])[selected].mean()))
    passed = (
        vs_baseline[0] >= 0.003 and vs_baseline[1] > 0 and vs_bias[1] > 0
        and min(half_deltas) > 0
        and np.mean(primary["large_error"] - baseline["large_error"]) <= 0.005
    )
    return {
        "passes_exploratory_interest_gate": bool(passed),
        "primary_half_utility_deltas": half_deltas,
        "compared_method_count": len(comparisons),
        "production_adoption_permitted": False,
    }


def main() -> None:
    """Runs the fixed targeted test and preserves all OOF predictions."""
    arguments = parse_arguments()
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    data = run.load_data(arguments.research)
    with np.load(arguments.pilot) as payload:
        saved = {name: payload[name].copy() for name in payload.files}
    if not np.array_equal(saved["players"], data.players):
        raise ValueError("Pilot player order differs from audited data")
    if not np.array_equal(saved["labels"], data.labels):
        raise ValueError("Pilot labels differ from audited data")
    if not np.array_equal(saved["halves"], data.halves):
        raise ValueError("Pilot upstream halves differ from audited data")
    if np.any(data.features[:, WINRATE_STD_COLUMN] > 0.50001):
        raise ValueError("Winrate std is outside the expected [0, 0.5] range")
    torch.set_num_threads(4)
    predictions, coefficients, conditional = _cross_validation(data, saved["folds"])
    comparisons = _comparisons(predictions, data)
    report = {
        "scope": "post-selection exploratory; previously studied cohort",
        "audit": data.audit,
        "associations": _associations(data, conditional),
        "metrics": {
            name: diagnose.summary(values, data.labels, data.probabilities)
            for name, values in predictions.items()
        },
        "paired_delta_95ci": comparisons,
        "subgroups": _subgroup_deltas(predictions, data),
        "coefficients": coefficients,
        "decision": _gate(comparisons, predictions, data),
    }
    arguments.output.mkdir(parents=True)
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        arguments.output / "predictions.npz",
        players=data.players, labels=data.labels, halves=data.halves,
        folds=saved["folds"], conditional_volatility=conditional,
        model_names=np.asarray(list(predictions)),
        probabilities=np.stack(list(predictions.values())),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
