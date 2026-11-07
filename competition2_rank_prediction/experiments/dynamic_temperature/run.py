"""Runs a preregistered two-parameter dynamic-temperature OOF experiment.

From the repository root:
    runtime/competition1_research/.venv/bin/python -m \
        competition2_rank_prediction.experiments.dynamic_temperature.run
"""

import argparse
import json
import pathlib
import time

import matplotlib.pyplot
import numpy as np
import torch

from competition2_rank_prediction.experiments.dynamic_temperature import audit
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.rank_experts import diagnose
from competition2_rank_prediction.experiments.rank_experts import run as pilot_run


SEED = 20261107


class TemperatureModel(torch.nn.Module):
    """Scales frozen logits using one global or two dynamic parameters."""

    def __init__(self, dynamic: bool) -> None:
        super().__init__()
        self.dynamic = dynamic
        self.coefficients = torch.nn.Parameter(torch.zeros(2 if dynamic else 1))

    def temperature(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Returns bounded per-question temperatures."""
        log_temperature = self.coefficients[0] + torch.zeros_like(uncertainty)
        if self.dynamic:
            log_temperature = log_temperature + self.coefficients[1] * uncertainty
        return log_temperature.clamp(float(np.log(0.5)), float(np.log(2))).exp()

    def forward(self, logits: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """Returns scaled logits without changing class ordering."""
        return logits / self.temperature(uncertainty)[:, None]


def parse_arguments() -> argparse.Namespace:
    """Parses immutable cache paths and experiment output location."""
    return common.parse_arguments(__doc__ or "", "dynamic_temperature_20261107")


def _fit(
    data: pilot_run.PilotData,
    uncertainty: np.ndarray,
    training: np.ndarray,
    dynamic: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = uncertainty[training].mean()
    scale = max(float(uncertainty[training].std()), 1e-5)
    inputs = torch.tensor((uncertainty - mean) / scale, dtype=torch.float32)
    logits = torch.tensor(np.log(data.probabilities.clip(1e-9)), dtype=torch.float32)
    labels = torch.tensor(data.labels[training], dtype=torch.long)
    torch.manual_seed(SEED)
    model = TemperatureModel(dynamic)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    with torch.no_grad():
        if not torch.equal(model(logits, inputs), logits):
            raise RuntimeError("Initial T=1 logit parity failed")
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(logits[training], inputs[training]), labels,
        )
        if dynamic:
            loss = loss + 0.01 * model.coefficients[1].square()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probabilities = torch.softmax(model(logits, inputs), dim=1).numpy()
        temperatures = model.temperature(inputs).numpy()
    coefficients = model.state_dict()["coefficients"].numpy()
    return probabilities, temperatures, {
        "a": float(coefficients[0]),
        "b": float(coefficients[1]) if dynamic else 0.0,
        "uncertainty_train_mean": float(mean),
        "uncertainty_train_std": scale,
    }


def cross_validate(
    data: pilot_run.PilotData,
    features: audit.AuditData,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]]]:
    """Fits all calibrators separately within each upstream-excluded half."""
    probabilities = {
        name: np.zeros_like(data.probabilities)
        for name in ("global_temperature", "dynamic_temperature")
    }
    temperatures = {name: np.zeros(len(data.labels)) for name in probabilities}
    records: list[dict[str, object]] = []
    for half in (0, 1):
        for fold in range(5):
            training = (data.halves == half) & (folds != fold)
            testing = (data.halves == half) & (folds == fold)
            if set(data.players[training]) & set(data.players[testing]):
                raise RuntimeError("Calibration player leakage")
            for name, dynamic in (("global_temperature", False), ("dynamic_temperature", True)):
                values, scales, coefficients = _fit(
                    data, features.signals[:, audit.PRIMARY_SIGNAL], training, dynamic,
                )
                probabilities[name][testing] = values[testing]
                temperatures[name][testing] = scales[testing]
                records.append({
                    "half": half, "fold": fold, "method": name,
                    "train_players": int(training.sum()),
                    "test_players": int(testing.sum()), "parameters": coefficients,
                })
            print(f"half={half} fold={fold} complete", flush=True)
    probabilities["baseline"] = data.probabilities.copy()
    temperatures["baseline"] = np.ones(len(data.labels))
    return probabilities, temperatures, records


def _outcomes(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    decisions = audit.decode(probabilities)
    return {
        "utility": diagnose.REWARDS[labels, decisions],
        "exact": (decisions == labels).astype(float),
        "large_error": (np.abs(decisions - labels) >= 2).astype(float),
        "nll": -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)),
        "brier": np.square(probabilities - np.eye(10)[labels]).sum(axis=1),
    }


def _bootstrap(differences: np.ndarray) -> dict[str, float]:
    generator = np.random.default_rng(SEED)
    means = []
    for _ in range(5000):
        rows = generator.integers(0, len(differences), len(differences))
        means.append(differences[rows].mean())
    low, high = np.quantile(means, (0.025, 0.975))
    return {"delta": float(differences.mean()), "low": float(low), "high": float(high)}


def paired_comparisons(
    probabilities: dict[str, np.ndarray],
    data: pilot_run.PilotData,
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns paired player bootstrap differences with both frozen controls."""
    results = {}
    candidate = _outcomes(probabilities["dynamic_temperature"], data.labels)
    for reference in ("baseline", "global_temperature"):
        original = _outcomes(probabilities[reference], data.labels)
        results[reference] = {
            name: _bootstrap(values - original[name])
            for name, values in candidate.items()
        }
    return results


def subgroup_report(
    probabilities: dict[str, np.ndarray],
    data: pilot_run.PilotData,
) -> dict[str, object]:
    """Reports fixed true-rank, question-count and upstream-half comparisons."""
    results: dict[str, object] = {}
    for name, selected in common.subgroup_masks(data).items():
        baseline = data.probabilities[selected]
        results[name] = {
            method: diagnose.summary(values[selected], data.labels[selected], baseline)
            for method, values in probabilities.items()
        }
    return results


def _boundary_aucs(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {}
    for low, high in pilot_run.BOUNDARIES:
        odds = probabilities[:, high] / probabilities[:, [low, high]].sum(axis=1)
        differences = odds[labels == high, None] - odds[labels == low]
        result[f"{pilot_run.LEVELS[low]}/{pilot_run.LEVELS[high]}"] = float(np.mean(
            (differences > 0) + 0.5 * (differences == 0),
        ))
    return result


def temperature_report(values: np.ndarray) -> dict[str, float]:
    """Summarizes bounded temperatures, including boundary saturation."""
    return {
        "min": float(values.min()), "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)), "q90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
        "lower_bound_fraction": float(np.mean(values <= 0.500001)),
        "upper_bound_fraction": float(np.mean(values >= 1.999999)),
    }


def decision_gate(
    probabilities: dict[str, np.ndarray],
    comparisons: dict[str, dict[str, dict[str, float]]],
    data: pilot_run.PilotData,
) -> dict[str, object]:
    """Applies the preregistered exploratory gate; never permits deployment."""
    candidate = _outcomes(probabilities["dynamic_temperature"], data.labels)
    baseline = _outcomes(probabilities["baseline"], data.labels)
    differences = candidate["utility"] - baseline["utility"]
    halves = [float(differences[data.halves == half].mean()) for half in (0, 1)]
    rank_deltas = [float(differences[data.labels == rank].mean()) for rank in range(10)]
    passed = (
        comparisons["baseline"]["utility"]["delta"] >= 0.003
        and comparisons["baseline"]["utility"]["low"] > 0
        and comparisons["global_temperature"]["utility"]["low"] > 0
        and min(halves) > 0 and min(rank_deltas) >= -0.03
        and np.mean(candidate["large_error"] - baseline["large_error"]) <= 0.005
    )
    return {
        "passes_exploratory_interest_gate": bool(passed),
        "half_utility_deltas": halves, "rank_utility_deltas": rank_deltas,
        "new_outer_cohort_warranted": bool(passed),
        "production_integration_permitted": False,
    }


def _plot(
    features: audit.AuditData,
    temperatures: dict[str, np.ndarray],
    output: pathlib.Path,
) -> None:
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(
        features.signals[:, audit.PRIMARY_SIGNAL], temperatures["dynamic_temperature"],
        s=8, alpha=0.25,
    )
    axes[0].set_xlabel("Fixed consensus expected-rank SE proxy")
    axes[0].set_ylabel("OOF dynamic temperature")
    for name in ("global_temperature", "dynamic_temperature"):
        axes[1].hist(temperatures[name], bins=25, alpha=0.5, label=name)
    axes[1].set_xlabel("OOF temperature")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "temperature_diagnostics.png", dpi=160)
    matplotlib.pyplot.close(figure)


def _validate_parity(data: pilot_run.PilotData) -> dict[str, float]:
    identity = audit.softmax(np.log(data.probabilities.clip(1e-9)))
    difference = float(np.max(np.abs(identity - data.probabilities)))
    if difference > 1e-6 or not np.array_equal(
        audit.decode(identity), audit.decode(data.probabilities),
    ):
        raise RuntimeError("Identity temperature baseline parity failed")
    baseline = diagnose.summary(data.probabilities, data.labels, data.probabilities)
    if abs(baseline["utility"] - 0.7170117549099193) > 1e-12:
        raise RuntimeError("Cached C1+C2 baseline does not reproduce original score")
    return {
        "identity_max_probability_difference": difference,
        "baseline_utility": baseline["utility"],
    }


def main() -> None:
    """Runs the fixed audit, calibration controls and paired evaluation."""
    arguments = parse_arguments()
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    torch.set_num_threads(4)
    started = time.monotonic()
    data = pilot_run.load_data(arguments.research)
    with np.load(arguments.pilot) as payload:
        saved = {name: payload[name].copy() for name in payload.files}
    for field, values in (
        ("players", data.players), ("labels", data.labels), ("halves", data.halves),
    ):
        if not np.array_equal(saved[field], values):
            raise ValueError(f"Saved pilot {field} differs from audited data")
    parity = _validate_parity(data)
    features = audit.build_audit(arguments.research, data)
    probabilities, temperatures, records = cross_validate(data, features, saved["folds"])
    comparisons = paired_comparisons(probabilities, data)
    metrics = {
        name: diagnose.summary(values, data.labels, data.probabilities)
        for name, values in probabilities.items()
    }
    if abs(metrics["global_temperature"]["utility"] - 0.7169599357481623) > 1e-9:
        raise RuntimeError("Global temperature does not reproduce existing control")
    report = _report(data, features, probabilities, temperatures)
    report.update({
        "scope": "exploratory only; no official-test data or production change",
        "audit": data.audit, "parity": parity, "metrics": metrics,
        "parameters": records, "paired_delta_95ci": comparisons,
        "decision": decision_gate(probabilities, comparisons, data),
        "elapsed_seconds": time.monotonic() - started,
    })
    arguments.output.mkdir(parents=True)
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    _save_arrays(arguments.output, data, features, probabilities, temperatures)
    _plot(features, temperatures, arguments.output)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(json.dumps(report["decision"], indent=2), flush=True)


def _report(
    data: pilot_run.PilotData,
    features: audit.AuditData,
    probabilities: dict[str, np.ndarray],
    temperatures: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "uncertainty_audit": audit.association_report(features, data),
        "same_player_subset_audit": audit.subset_report(features, data),
        "temperature_grid_oracle_audit": audit.temperature_grid_audit(data),
        "subgroups": subgroup_report(probabilities, data),
        "boundary_aucs": {
            name: _boundary_aucs(values, data.labels)
            for name, values in probabilities.items()
        },
        "temperatures": {
            name: temperature_report(values)
            for name, values in temperatures.items()
        },
        "raw_probability_argmax_changes": {
            name: int(np.sum(values.argmax(axis=1) != data.probabilities.argmax(axis=1)))
            for name, values in probabilities.items()
        },
    }


def _save_arrays(
    output: pathlib.Path,
    data: pilot_run.PilotData,
    features: audit.AuditData,
    probabilities: dict[str, np.ndarray],
    temperatures: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        output / "predictions.npz",
        players=data.players, labels=data.labels, halves=data.halves,
        counts=data.counts, model_names=np.asarray(list(probabilities)),
        probabilities=np.stack(list(probabilities.values())),
        temperatures=np.stack(list(temperatures.values())),
        uncertainty_features=features.signals,
        uncertainty_feature_names=np.asarray(audit.SIGNAL_NAMES),
        loo_proxy_flip_fraction=features.loo_flip_fraction,
        subset_proxy_statistics=features.subset_records,
    )


if __name__ == "__main__":
    main()
