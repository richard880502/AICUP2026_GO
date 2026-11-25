"""Runs the preregistered 1D/2D linear specialist against C1+C2.

Extract excluded-half embeddings first using the companion extract module.
"""

import json

import numpy as np
import sklearn.linear_model  # type: ignore[import-untyped]
import sklearn.metrics  # type: ignore[import-untyped]
import sklearn.preprocessing  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.dynamic_temperature import audit
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.rank_experts import diagnose
from competition2_rank_prediction.experiments.rank_experts import run as pilot_run


SEED = 20261107
LOW = 4
HIGH = 5


def _pair_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(sklearn.metrics.roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(sklearn.metrics.balanced_accuracy_score(
            labels, probabilities >= 0.5,
        )),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def _cross_validate(
    data: pilot_run.PilotData,
    embeddings: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    scores = np.zeros(len(data.labels))
    records: list[dict[str, object]] = []
    for half in (0, 1):
        for fold in range(5):
            train = (data.halves == half) & (folds != fold)
            train &= np.isin(data.labels, (LOW, HIGH))
            test = (data.halves == half) & (folds == fold)
            if set(data.players[train]) & set(data.players[test]):
                raise ValueError("Player overlap between specialist folds")
            if train.sum() != 160:
                raise ValueError(f"Unexpected pair training count: {train.sum()}")
            scaler = sklearn.preprocessing.StandardScaler()
            train_x = scaler.fit_transform(embeddings[train])
            test_x = scaler.transform(embeddings[test])
            model = sklearn.linear_model.LogisticRegression(
                C=0.01, class_weight="balanced", solver="lbfgs",
                max_iter=1000, random_state=SEED,
            )
            labels = (data.labels[train] == HIGH).astype(int)
            model.fit(train_x, labels)
            scores[test] = model.predict_proba(test_x)[:, 1]
            records.append({
                "half": half, "fold": fold,
                "train_pair_players": int(train.sum()),
                "test_players": int(test.sum()),
                "train_pair_metrics": _pair_metrics(
                    labels, model.predict_proba(train_x)[:, 1],
                ),
                "iterations": int(model.n_iter_[0]),
            })
    return scores, records


def _blend(
    baseline: np.ndarray,
    specialist: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    mass = baseline[:, LOW] + baseline[:, HIGH]
    decision = audit.decode(baseline)
    routed = (mass >= 0.35) & np.isin(decision, (LOW, HIGH))
    if weight == 0:
        return baseline.copy(), routed
    base_q = (baseline[:, HIGH] / mass.clip(1e-12)).clip(1e-6, 1 - 1e-6)
    spec_q = specialist.clip(1e-6, 1 - 1e-6)
    odds = (1 - weight) * np.log(base_q / (1 - base_q))
    odds += weight * np.log(spec_q / (1 - spec_q))
    blended = 1 / (1 + np.exp(-odds))
    output = baseline.copy()
    output[routed, LOW] = mass[routed] * (1 - blended[routed])
    output[routed, HIGH] = mass[routed] * blended[routed]
    return output, routed


def _evaluate(
    data: pilot_run.PilotData,
    scores: np.ndarray,
    results: np.ndarray,
    routed: np.ndarray,
) -> dict[str, object]:
    base = data.probabilities
    pair = np.isin(data.labels, (LOW, HIGH))
    original = diagnose.REWARDS[data.labels, audit.decode(base)]
    candidate = diagnose.REWARDS[data.labels, audit.decode(results)]
    differences = candidate - original
    groups = {}
    for name, mask in common.subgroup_masks(data).items():
        groups[name] = {
            "players": int(mask.sum()),
            "baseline_utility": float(original[mask].mean()),
            "candidate_utility": float(candidate[mask].mean()),
            "delta": float(differences[mask].mean()),
        }
    old = audit.decode(base)
    new = audit.decode(results)
    return {
        "pair_metrics": _pair_report(data, scores),
        "baseline": diagnose.summary(base, data.labels, base),
        "candidate": diagnose.summary(results, data.labels, base),
        "paired_utility_95ci": _bootstrap(differences),
        "groups": groups,
        "confusion_pair_true_all_decisions": {
            str(rank): np.bincount(new[data.labels == rank], minlength=10).tolist()
            for rank in (LOW, HIGH)
        },
        "routing": {
            "routed": int(routed.sum()),
            "non_pair_routed": int((routed & ~pair).sum()),
            "changed": int((old != new).sum()),
            "unchanged_routed": int((routed & (old == new)).sum()),
            "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()),
            "non_pair_wins": int(((differences > 0) & ~pair).sum()),
            "non_pair_losses": int(((differences < 0) & ~pair).sum()),
        },
    }


def _pair_report(data: pilot_run.PilotData, scores: np.ndarray) -> dict[str, object]:
    base = data.probabilities
    pair = np.isin(data.labels, (LOW, HIGH))
    base_q = base[:, HIGH] / (base[:, LOW] + base[:, HIGH]).clip(1e-12)
    labels = (data.labels == HIGH).astype(int)
    result: dict[str, object] = {}
    for name, mask in (("all", pair), ("half_0", pair & (data.halves == 0)),
                       ("half_1", pair & (data.halves == 1))):
        result[name] = {
            "players": int(mask.sum()),
            "baseline": _pair_metrics(labels[mask], base_q[mask]),
            "specialist": _pair_metrics(labels[mask], scores[mask]),
        }
    return result


def _bootstrap(differences: np.ndarray) -> dict[str, float]:
    generator = np.random.default_rng(SEED)
    means = []
    for _ in range(5000):
        rows = generator.integers(0, len(differences), len(differences))
        means.append(differences[rows].mean())
    low, high = np.quantile(means, (0.025, 0.975))
    return {"delta": float(differences.mean()), "low": float(low), "high": float(high)}


def main() -> None:
    """Evaluates one frozen specialist with within-half player folds."""
    arguments = common.parse_arguments(
        __doc__ or "", "adjacent_rank_specialist_20261107",
    )
    output = arguments.output
    if (output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite the existing report")
    data = pilot_run.load_data(arguments.research)
    with np.load(output / "embeddings.npz") as payload:
        embeddings = np.asarray(payload["embeddings"])
        for name in ("players", "labels", "halves"):
            if not np.array_equal(payload[name], getattr(data, name)):
                raise ValueError(f"Embedding alignment mismatch: {name}")
    with np.load(arguments.pilot) as payload:
        folds = np.asarray(payload["folds"])
        for name in ("players", "labels", "halves"):
            if not np.array_equal(payload[name], getattr(data, name)):
                raise ValueError(f"Saved fold alignment mismatch: {name}")
    baseline_score = diagnose.summary(
        data.probabilities, data.labels, data.probabilities,
    )["utility"]
    if abs(baseline_score - 0.7170117549099193) > 1e-12:
        raise ValueError("Baseline parity failed")
    scores, folds_info = _cross_validate(data, embeddings, folds)
    identity, _ = _blend(data.probabilities, scores, 0)
    if not np.array_equal(identity, data.probabilities):
        raise ValueError("Zero-weight parity failed")
    values, routed = _blend(data.probabilities, scores, 0.5)
    if not np.allclose(values.sum(axis=1), data.probabilities.sum(axis=1), atol=1e-6):
        raise ValueError("Pair mass was not preserved")
    report = _evaluate(data, scores, values, routed)
    report["folds"] = folds_info
    report["data_audit"] = data.audit
    report["scope"] = "exploratory; reused cohort; no official test"
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        output / "predictions.npz", players=data.players, labels=data.labels,
        halves=data.halves, folds=folds, scores=scores, routed=routed,
        baseline=data.probabilities, candidate=values,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
