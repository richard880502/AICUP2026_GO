"""Diagnoses train/OOF expert behavior using the frozen pilot recipe.

Runs the original fixed models again with diagnostic snapshots, checks parity
against saved OOF predictions, and measures feature redundancy. No production
model or official test data is used. From the repository root:
    runtime/competition1_research/.venv/bin/python -m \
        competition2_rank_prediction.experiments.rank_experts.diagnose
"""

import argparse
import json
import pathlib
import time

import matplotlib.pyplot
import numpy as np
import torch

from competition2_rank_prediction.experiments.rank_experts import run
from competition2_rank_prediction.experiments.rank_experts import common


SNAPSHOTS = (0, 20, 50, 100, 200)
REWARDS = np.eye(10) + np.exp(-1) * (
    np.eye(10, k=1) + np.eye(10, k=-1)
)


def parse_arguments() -> argparse.Namespace:
    """Parses pilot-cache and diagnostic-output locations."""
    return common.parse_arguments(
        __doc__ or "", "rank_experts_diagnostics_20261107", True,
    )


def _ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    result = 0.0
    bins = np.minimum((confidence * 15).astype(int), 14)
    for index in range(15):
        selected = bins == index
        if selected.any():
            result += selected.mean() * abs(
                confidence[selected].mean() - correct[selected].mean(),
            )
    return float(result)


def summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    """Summarizes utility decoding, probability quality and correction size."""
    predictions = (probabilities @ REWARDS).argmax(axis=1)
    before = (baseline @ REWARDS).argmax(axis=1)
    correction = np.log(probabilities.clip(1e-9)) - np.log(baseline.clip(1e-9))
    correction -= correction.mean(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities.clip(1e-9))).sum(axis=1)
    return {
        "utility": float(REWARDS[labels, predictions].mean()),
        "exact": float(np.mean(predictions == labels)),
        "large_error": float(np.mean(np.abs(predictions - labels) >= 2)),
        "nll": float(-np.log(probabilities[
            np.arange(len(labels)), labels,
        ].clip(1e-9)).mean()),
        "brier": float(np.square(probabilities - np.eye(10)[labels]).sum(axis=1).mean()),
        "top_label_ece_15bins": _ece(probabilities, labels),
        "entropy_mean": float(entropy.mean()),
        "entropy_q10": float(np.quantile(entropy, 0.1)),
        "entropy_q90": float(np.quantile(entropy, 0.9)),
        "centered_log_correction_rms": float(np.sqrt(np.square(correction).mean())),
        "correction_abs_q90": float(np.quantile(np.abs(correction), 0.9)),
        "decision_change_fraction": float(np.mean(predictions != before)),
        "positive_utility_changes": float(np.mean(
            REWARDS[labels, predictions] > REWARDS[labels, before],
        )),
        "negative_utility_changes": float(np.mean(
            REWARDS[labels, predictions] < REWARDS[labels, before],
        )),
    }


def _tensors(
    data: run.PilotData,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
) -> run.FoldData:
    mean = data.features[train_mask].mean(axis=0)
    scale = data.features[train_mask].std(axis=0).clip(1e-5)
    features = (data.features - mean) / scale
    logits = np.log(data.probabilities.clip(1e-9)).astype(np.float32)
    return run.FoldData(
        torch.tensor(features[train_mask], device=device),
        torch.tensor(features[test_mask], device=device),
        torch.tensor(logits[train_mask], device=device),
        torch.tensor(logits[test_mask], device=device),
        torch.tensor(data.labels[train_mask], device=device, dtype=torch.long),
    )


def _snapshot(
    model: run.ExpertModel,
    fold: run.FoldData,
    epoch: int,
    test_labels: np.ndarray,
) -> dict[str, object]:
    with torch.inference_mode():
        train = torch.softmax(model(fold.train_logits, fold.train_features), 1)
        test = torch.softmax(model(fold.test_logits, fold.test_features), 1)
        base_train = torch.softmax(fold.train_logits, 1).cpu().numpy()
        base_test = torch.softmax(fold.test_logits, 1).cpu().numpy()
    return {
        "epoch": epoch,
        "train": summary(train.cpu().numpy(), fold.train_labels.cpu().numpy(), base_train),
        "test": summary(test.cpu().numpy(), test_labels, base_test),
    }


def _fit_diagnostic(
    fold: run.FoldData,
    kind: str,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    torch.manual_seed(run.SEED)
    model = run.ExpertModel(fold.train_features.shape[1], kind).to(
        fold.train_features.device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=0.01,
    )
    snapshots = [_snapshot(model, fold, 0, test_labels)]
    for epoch in range(1, 201):
        corrected = model(fold.train_logits, fold.train_features)
        loss = torch.nn.functional.cross_entropy(corrected, fold.train_labels)
        penalty = (corrected - fold.train_logits).square().mean()
        optimizer.zero_grad()
        (loss + 0.01 * penalty).backward()
        optimizer.step()
        if epoch in SNAPSHOTS:
            snapshots.append(_snapshot(model, fold, epoch, test_labels))
    with torch.inference_mode():
        predictions = torch.softmax(model(fold.test_logits, fold.test_features), 1)
    return predictions.cpu().numpy(), snapshots


def _temperature(fold: run.FoldData) -> tuple[np.ndarray, float]:
    # This prespecified one-parameter diagnostic is not used to select experts.
    log_temperature = torch.nn.Parameter(torch.zeros(
        (), device=fold.train_logits.device,
    ))
    optimizer = torch.optim.Adam([log_temperature], lr=0.02)
    for _ in range(200):
        temperature = log_temperature.clamp(-0.693147, 0.693147).exp()
        loss = torch.nn.functional.cross_entropy(
            fold.train_logits / temperature, fold.train_labels,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        temperature = log_temperature.clamp(-0.693147, 0.693147).exp()
        predictions = torch.softmax(fold.test_logits / temperature, 1)
    return predictions.cpu().numpy(), float(temperature)


def _boundary_auc(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {}
    for low, high in run.BOUNDARIES:
        odds = probabilities[:, high] / probabilities[:, [low, high]].sum(axis=1)
        differences = odds[labels == high, None] - odds[labels == low]
        result[f"{run.LEVELS[low]}/{run.LEVELS[high]}"] = float(np.mean(
            (differences > 0) + 0.5 * (differences == 0),
        ))
    return result


def _subgroups(
    probabilities: np.ndarray,
    data: run.PilotData,
) -> dict[str, object]:
    groups = common.subgroup_masks(data)
    results: dict[str, object] = {}
    for name, selected in groups.items():
        baseline = data.probabilities[selected]
        metrics = summary(probabilities[selected], data.labels[selected], baseline)
        original = summary(baseline, data.labels[selected], baseline)
        results[name] = {
            "count": int(selected.sum()),
            "metrics": metrics,
            "utility_delta": metrics["utility"] - original["utility"],
        }
    results["boundary_auc"] = _boundary_auc(probabilities, data.labels)
    return results


def _ridge_prediction(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    mean = features[train_mask].mean(axis=0)
    scale = features[train_mask].std(axis=0).clip(1e-5)
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features)), normalized))
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design[train_mask].T @ design[train_mask] + penalty,
        design[train_mask].T @ targets[train_mask],
    )
    return design @ weights


def redundancy(data: run.PilotData, folds: np.ndarray) -> dict[str, object]:
    """Measures held-out linear reconstruction and residual associations."""
    extras = data.features[:, 10:]
    reconstructed = np.zeros_like(extras)
    for half in (0, 1):
        for fold_id in range(5):
            training = (data.halves == half) & (folds != fold_id)
            testing = (data.halves == half) & (folds == fold_id)
            reconstructed[testing] = _ridge_prediction(
                data.features[:, :10], extras, training,
            )[testing]
    total = np.square(extras - extras.mean(axis=0)).sum(axis=0)
    r_squared = 1 - np.square(extras - reconstructed).sum(axis=0) / total.clip(1e-8)
    rank_residual = data.labels - data.probabilities @ np.arange(10)
    correlations = []
    for index in range(extras.shape[1]):
        conditional = extras[:, index] - reconstructed[:, index]
        value = 0.0
        if conditional.std() > 1e-6:
            value = float(np.corrcoef(conditional, rank_residual)[0, 1])
        correlations.append(value)
    return {
        "extra_feature_count": extras.shape[1],
        "extra_feature_names": _extra_names(),
        "ridge_lambda": 10,
        "oof_linear_reconstruction_r2": r_squared.tolist(),
        "median_r2": float(np.median(r_squared)),
        "conditional_rank_residual_correlations": correlations,
        "max_absolute_correlation_exploratory": float(np.max(np.abs(correlations))),
        "interpretation": "linear redundancy only; max correlation is multiple-tested",
    }


def _extra_names() -> list[str]:
    names = []
    for member in ("d96", "resnet", "opp"):
        for statistic in ("mean", "std"):
            for rank in run.LEVELS:
                names.append(f"{member}_probability_{statistic}_{rank}")
    names.extend((
        "own_rank_mean", "own_rank_std", "opponent_rank_mean",
        "opponent_rank_std", "rank_gap_mean", "rank_gap_std", "log_game_count",
    ))
    for statistic in ("mean", "std"):
        for field in (
            "winrate", "score_lead", "move_count", "last_move_by_player",
            "ended_two_passes", "leading_black_stones",
        ):
            names.append(f"final_{field}_{statistic}")
    return names


def _diagnose_folds(
    data: run.PilotData,
    folds: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    predictions = {kind: np.zeros_like(data.probabilities) for kind in run.KINDS}
    predictions["temperature"] = np.zeros_like(data.probabilities)
    records: list[dict[str, object]] = []
    for half in (0, 1):
        for fold_id in range(5):
            train_mask = (data.halves == half) & (folds != fold_id)
            test_mask = (data.halves == half) & (folds == fold_id)
            fold = _tensors(data, train_mask, test_mask, device)
            for kind in run.KINDS:
                values, snapshots = _fit_diagnostic(fold, kind, data.labels[test_mask])
                predictions[kind][test_mask] = values
                records.append({
                    "half": half, "fold": fold_id, "kind": kind,
                    "train_count": int(train_mask.sum()),
                    "test_count": int(test_mask.sum()), "snapshots": snapshots,
                })
            values, temperature = _temperature(fold)
            predictions["temperature"][test_mask] = values
            records.append({
                "half": half, "fold": fold_id, "kind": "temperature",
                "temperature": temperature,
            })
            print(f"half={half} fold={fold_id} diagnostics complete", flush=True)
    return predictions, records


def _plot_curves(records: list[dict[str, object]], output: pathlib.Path) -> None:
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(11, 4))
    for kind in ("linear", "region", "boundary"):
        matching = [record for record in records if record["kind"] == kind]
        for split, style in (("train", "-"), ("test", "--")):
            for axis, metric in zip(axes, ("utility", "nll")):
                values = _curve_values(matching, split, metric)
                axis.plot(SNAPSHOTS, values, style, label=f"{kind} {split}")
                axis.set_xlabel("Epoch (fixed snapshots; no model selection)")
                axis.set_ylabel(metric)
    axes[0].legend(fontsize=8)
    figure.suptitle("Frozen expert diagnostics: within-half player CV")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    matplotlib.pyplot.close(figure)


def _curve_values(
    records: list[dict[str, object]],
    split: str,
    metric: str,
) -> list[float]:
    values = []
    for epoch in SNAPSHOTS:
        scores = []
        for record in records:
            snapshots = record["snapshots"]
            if not isinstance(snapshots, list):
                raise TypeError("Diagnostic snapshots must be a list")
            for snapshot in snapshots:
                if snapshot["epoch"] == epoch:
                    scores.append(float(snapshot[split][metric]))
        values.append(float(np.mean(scores)))
    return values


def main() -> None:
    """Repeats the pilot for diagnosis and writes an analysis-only report."""
    arguments = parse_arguments()
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    arguments.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    data = run.load_data(arguments.research)
    with np.load(arguments.pilot) as payload:
        saved = {key: payload[key].copy() for key in payload.files}
    if not np.array_equal(saved["players"], data.players):
        raise ValueError("Saved pilot players do not match audited input")
    if not np.array_equal(saved["labels"], data.labels):
        raise ValueError("Saved pilot labels do not match audited input")
    if not np.array_equal(saved["halves"], data.halves):
        raise ValueError("Saved pilot halves do not match audited input")
    predictions, records = _diagnose_folds(
        data, saved["folds"], torch.device(arguments.device),
    )
    parity = {}
    for index, name in enumerate(saved["model_names"]):
        if name == "baseline":
            continue
        parity[str(name)] = float(np.max(np.abs(
            predictions[str(name)] - saved["predictions"][index],
        )))
    if max(parity.values()) > 1e-6:
        raise RuntimeError(f"Refit does not reproduce pilot: {parity}")
    predictions["baseline"] = data.probabilities
    report = {
        "scope": "exploratory diagnostics; no new cohort or production change",
        "audit": data.audit,
        "pilot_prediction_max_abs_difference": parity,
        "ece_definition": "15-bin max-probability argmax-class calibration",
        "nll_definition": "probabilities clipped at 1e-9, matching pilot logits",
        "training_summary_caveat": "each player appears in four training folds",
        "fold_records": records,
        "oof_metrics": {
            name: summary(values, data.labels, data.probabilities)
            for name, values in predictions.items()
        },
        "oof_subgroups": {
            name: _subgroups(values, data)
            for name, values in predictions.items()
        },
        "redundancy": redundancy(data, saved["folds"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    _plot_curves(records, arguments.output)
    print(json.dumps(report["oof_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
