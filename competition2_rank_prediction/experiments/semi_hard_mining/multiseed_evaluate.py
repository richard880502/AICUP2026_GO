"""Evaluates all seeds without selecting a winner and applies the mining gate.

Run after multiseed_train completes; results remain exploratory.
"""

import json
import pathlib
from typing import TypedDict

import matplotlib.pyplot
import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import (
    multiseed_train,
)
from competition2_rank_prediction.experiments.semi_hard_mining import train
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


class SeedResult(TypedDict):
    """Types the metrics and paired outcomes for one complete replication."""

    seed: int
    metrics: dict[str, dict[str, float]]
    utility_delta: float
    wins: int
    losses: int


def checkpoint_audit(
    context: train.TrainingContext, seed: int
) -> dict[str, object]:
    """Verifies complete budgets and paired warmup equality before evaluation."""
    histories = {}
    for arm in ("A", "B"):
        histories[arm] = json.loads(
            (context.output / f"seed_{seed}/{arm}_history.json").read_text(),
        )
        if len(histories[arm]) != multiseed_train.EPOCHS:
            raise ValueError(f"Incomplete training: seed={seed}, arm={arm}")
    warmup = all(
        histories["A"][index]["inner"] == histories["B"][index]["inner"]
        and histories["A"][index]["train_nll_pair_gradnorm"]
        == histories["B"][index]["train_nll_pair_gradnorm"]
        for index in range(5)
    )
    if not warmup:
        raise ValueError(f"Warmup parity failed for seed {seed}")
    report: dict[str, object] = {
        "seed": seed,
        "epochs_per_arm": multiseed_train.EPOCHS,
        "updates_per_arm": multiseed_train.EPOCHS * multiseed_train.STEPS,
        "warmup_parity": warmup,
    }
    for arm, history in histories.items():
        report[f"{arm}_expected_selected_epoch"] = (
            int(
                np.argmin(
                    [row["inner"]["nll"] for row in history],
                )
            )
            + 1
        )
    return report


def _probabilities(
    context: train.TrainingContext,
    arms: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    for split in ("inner", "evaluation"):
        parts[split] = {arm: [] for arm in arms}
    audits = []
    for seed in multiseed_train.SEEDS:
        record = checkpoint_audit(context, seed)
        for arm in arms:
            state = torch.load(
                context.output / f"seed_{seed}/{arm}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if state["seed"] != seed or state["recipe"] != {
                "seed": seed,
                "epochs": multiseed_train.EPOCHS,
                "steps": multiseed_train.STEPS,
            }:
                raise ValueError("Checkpoint recipe mismatch")
            record[f"{arm}_selected_epoch"] = state["epoch"]
            if state["epoch"] != record[f"{arm}_expected_selected_epoch"]:
                raise ValueError(
                    "Checkpoint was not selected by minimum inner NLL"
                )
            model = transformer_rank.RankTransformer(
                context.tokens.shape[2],
                context.tokens.shape[1],
            ).to(context.tokens.device)
            model.load_state_dict(state["state"])
            for split, questions in context.data.questions.items():
                parts[split][arm].append(
                    train.question_probabilities(model, context, questions),
                )
        audits.append(record)
    result = {}
    for split, values in parts.items():
        result[split] = {arm: np.stack(rows) for arm, rows in values.items()}
    return result, audits


def summarize(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    counts: np.ndarray,
    reference: str = "A",
    candidate: str = "B",
) -> dict[str, object]:
    """Compares paired seeds, fixed ensembles and per-player aggregate effects."""
    per_seed: list[SeedResult] = []
    deltas = []
    for index, seed in enumerate(multiseed_train.SEEDS):
        values = {arm: rows[index] for arm, rows in probabilities.items()}
        scores = {
            arm: statistics.player_scores(rows, labels)
            for arm, rows in values.items()
        }
        difference = scores[candidate] - scores[reference]
        deltas.append(difference)
        per_seed.append(
            {
                "seed": seed,
                "metrics": {
                    arm: train.metrics(rows, labels)
                    for arm, rows in values.items()
                },
                "utility_delta": float(difference.mean()),
                "wins": int((difference > 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    mean_metrics = {}
    for arm in probabilities:
        mean_metrics[arm] = {
            metric: float(
                np.mean([row["metrics"][arm][metric] for row in per_seed])
            )
            for metric in per_seed[0]["metrics"][arm]
        }
    ensemble = {arm: rows.mean(axis=0) for arm, rows in probabilities.items()}
    return {
        "per_seed": per_seed,
        "mean_seed_metrics": mean_metrics,
        "primary_paired_player_95ci": statistics.bootstrap_players(
            np.mean(deltas, axis=0)
        ),
        "positive_utility_seeds": sum(
            row["utility_delta"] > 0 for row in per_seed
        ),
        "groups": group_statistics(np.stack(deltas), labels, counts),
        "secondary_ensemble_metrics": {
            arm: train.metrics(rows, labels) for arm, rows in ensemble.items()
        },
        "secondary_ensemble_paired_95ci": statistics.bootstrap_players(
            statistics.player_scores(ensemble[candidate], labels)
            - statistics.player_scores(ensemble[reference], labels),
        ),
    }


def group_statistics(
    deltas: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    """Aggregates seed changes within the preregistered player subgroups."""
    masks = {
        name: labels == index for index, name in enumerate(rank_lib.LEVELS)
    }
    for low, high in ((5, 9), (10, 14), (15, 20)):
        masks[f"games_{low}_{high}"] = (counts >= low) & (counts <= high)
    return {
        name: {
            "players": int(mask.sum()),
            "mean_utility_delta": (
                float(deltas[:, mask].mean()) if mask.any() else None
            ),
        }
        for name, mask in masks.items()
    }


def mining_gate(reports: dict[str, dict]) -> dict[str, object]:
    """Applies the fixed second-round gate without inspecting new thresholds."""
    conditions = {}
    for split in ("inner", "evaluation"):
        baseline = reports[split]["mean_seed_metrics"]["A"]
        candidate = reports[split]["mean_seed_metrics"]["B"]
        local = statistics.screening_conditions(
            baseline,
            candidate,
            min(
                reports[split]["groups"][rank]["mean_utility_delta"]
                for rank in rank_lib.LEVELS
            ),
        )
        conditions.update(
            {f"{split}_{name}": value for name, value in local.items()}
        )
    conditions["two_positive_evaluation_seeds"] = (
        reports["evaluation"]["positive_utility_seeds"] >= 2
    )
    return {
        "conditions": conditions,
        "run_C": all(conditions.values()),
        "production_adoption_permitted": False,
    }


def plot_curves(
    context: train.TrainingContext,
    directories: dict[str, pathlib.Path] | None = None,
    filename: str = "multiseed_training_curves.png",
) -> None:
    """Writes all seed histories without selecting a favorable curve."""
    directories = directories or {"A": context.output, "B": context.output}
    figure, axes = matplotlib.pyplot.subplots(1, 3, figsize=(13, 4))
    for axis, seed in zip(axes, multiseed_train.SEEDS):
        for arm, directory in directories.items():
            history = json.loads(
                (directory / f"seed_{seed}/{arm}_history.json").read_text(),
            )
            axis.plot(
                [row["epoch"] for row in history],
                [row["inner"]["utility"] for row in history],
                label=arm,
            )
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Inner utility")
        axis.legend()
    figure.tight_layout()
    figure.savefig(context.output / filename, dpi=160)
    matplotlib.pyplot.close(figure)


def write_predictions(
    context: train.TrainingContext,
    split: str,
    values: dict[str, np.ndarray],
) -> None:
    """Writes aligned per-player seed arrays with fixed question metadata."""
    questions = context.data.questions[split]
    arrays = {
        "players": context.data.splits[split],
        "labels": np.asarray(
            [context.data.labels[rows[0]] for rows in questions]
        ),
        "counts": np.asarray([len(rows) for rows in questions]),
        "seeds": np.asarray(multiseed_train.SEEDS),
        **values,
    }
    np.savez_compressed(
        context.output / f"{split}_predictions.npz",
        allow_pickle=False,
        **arrays,
    )


def main() -> None:
    """Evaluates every replication and records the conditional mining decision."""
    arguments = common.parse_arguments(
        __doc__ or "",
        multiseed_train.OUTPUT_NAME,
        True,
    )
    if (arguments.output / "report.json").exists():
        raise FileExistsError("Refusing to replace multiseed report")
    torch.set_num_threads(4)
    context = multiseed_train.load_context(arguments)
    probabilities, audits = _probabilities(context, ("A", "B"))
    reports = {}
    for split, values in probabilities.items():
        questions = context.data.questions[split]
        labels = np.asarray(
            [context.data.labels[rows[0]] for rows in questions]
        )
        counts = np.asarray([len(rows) for rows in questions])
        reports[split] = summarize(values, labels, counts)
        write_predictions(context, split, values)
    report = {
        "results": reports,
        "gate": mining_gate(reports),
        "training_audit": audits,
        "scope": "exploratory same cohort; single own branch; conditional on three seeds",
    }
    (context.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_curves(context)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
