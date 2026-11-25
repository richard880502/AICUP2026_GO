"""Evaluates frozen A/B selections and applies the preregistered stop rule.

Run only after both fixed training budgets have finished.
"""

import json

import matplotlib.pyplot
import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)
from competition2_rank_prediction.experiments.semi_hard_mining import train
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


def _groups(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    masks = {
        name: labels == index for index, name in enumerate(rank_lib.LEVELS)
    }
    for low, high in ((5, 9), (10, 14), (15, 20)):
        masks[f"games_{low}_{high}"] = (counts >= low) & (counts <= high)
    result: dict[str, object] = {}
    for name, mask in masks.items():
        values = {
            arm: float(
                statistics.player_scores(probability[mask], labels[mask]).mean()
            )
            for arm, probability in probabilities.items()
        }
        result[name] = {
            "players": int(mask.sum()),
            **values,
            "delta": values["B"] - values["A"],
        }
    return result


def _comparison(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    differences = statistics.player_scores(
        probabilities["B"], labels
    ) - statistics.player_scores(probabilities["A"], labels)
    return {
        "metrics": {
            arm: train.metrics(values, labels)
            for arm, values in probabilities.items()
        },
        "paired_utility_95ci": statistics.bootstrap_players(differences),
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "changed": int(
            np.sum(
                rank_lib.decode(probabilities["A"])
                != rank_lib.decode(probabilities["B"]),
            )
        ),
        "groups": _groups(probabilities, labels, counts),
    }


def _gate(
    results: dict[str, dict],
) -> dict[str, object]:
    conditions = {}
    for split in ("inner", "evaluation"):
        baseline = results[split]["metrics"]["A"]
        candidate = results[split]["metrics"]["B"]
        local = statistics.screening_conditions(
            baseline,
            candidate,
            min(
                results[split]["groups"][rank]["delta"]
                for rank in rank_lib.LEVELS
            ),
        )
        conditions.update(
            {f"{split}_{name}": value for name, value in local.items()}
        )
    return {
        "conditions": conditions,
        "run_C": all(conditions.values()),
        "production_adoption_permitted": False,
    }


def _plot(context: train.TrainingContext) -> None:
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(10, 4))
    for arm in ("A", "B"):
        history = json.loads(
            (context.output / f"{arm}_history.json").read_text()
        )
        for axis, metric in zip(axes, ("utility", "pair_auc")):
            axis.plot(
                [row["epoch"] for row in history],
                [row["inner"][metric] for row in history],
                label=arm,
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(f"Inner {metric}")
            axis.legend()
    figure.tight_layout()
    figure.savefig(context.output / "training_curves.png", dpi=160)
    matplotlib.pyplot.close(figure)


def _training_questions(data: player_sampler.PlayerData) -> list[np.ndarray]:
    generator = np.random.default_rng(player_sampler.SEED + 3)
    questions = []
    for rank in range(10):
        eligible = [
            str(player)
            for player in data.splits["train"]
            if len(data.rows[str(player)]) >= 20
            and data.labels[data.rows[str(player)][0]] == rank
        ]
        for player in generator.choice(eligible, 30, replace=False):
            count = int(generator.integers(5, 21))
            questions.append(
                generator.choice(data.rows[str(player)], count, replace=False)
            )
    return questions


def _gradient_diagnostics(
    model: torch.nn.Module,
    context: train.TrainingContext,
) -> dict[str, float]:
    bags: list[np.ndarray] = []
    for low, high in player_sampler.sample_pairs(
        context.data,
        np.random.default_rng(player_sampler.SEED + 4),
    ):
        bags.extend((low, high))
    indices = np.concatenate(bags)
    model.eval()
    nll = torch.nn.functional.nll_loss(
        model(context.tokens[indices], context.masks[indices]),
        context.labels[indices],
    )
    nll_gradient = torch.autograd.grad(nll, model.project[0].weight)[0]
    strengths = model.score(
        model.encode(context.tokens[indices], context.masks[indices])
    ).flatten()
    pooled = torch.stack(
        [part.mean() for part in strengths.split([len(bag) for bag in bags])]
    )
    pair_loss = torch.nn.Softplus()(-(pooled[1::2] - pooled[::2])).mean()
    pair_gradient = torch.autograd.grad(pair_loss, model.project[0].weight)[0]
    return {
        "diagnostic_train_pair_nll": float(nll.detach()),
        "diagnostic_train_pair_loss": float(pair_loss.detach()),
        "project_weight_nll_gradient_norm": float(nll_gradient.norm()),
        "project_weight_unweighted_pair_gradient_norm": float(
            pair_gradient.norm()
        ),
        "weighted_pair_to_nll_gradient_ratio": float(
            0.05 * pair_gradient.norm() / nll_gradient.norm(),
        ),
    }


def _diagnostics(
    model: torch.nn.Module, context: train.TrainingContext
) -> dict[str, object]:
    questions = _training_questions(context.data)
    labels = np.asarray([context.data.labels[rows[0]] for rows in questions])
    return {
        "train_dense_300_player_metrics": train.metrics(
            train.question_probabilities(model, context, questions),
            labels,
        ),
        **_gradient_diagnostics(model, context),
    }


def main() -> None:
    """Loads selected checkpoints and evaluates fixed player-level queries."""
    arguments = common.parse_arguments(
        __doc__ or "", "semi_hard_mining_20261108", True
    )
    if (arguments.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite report.json")
    torch.set_num_threads(4)
    data = player_sampler.load_data(arguments.research)
    with np.load(arguments.output / "splits.npz") as saved:
        for name, players in data.splits.items():
            if not np.array_equal(saved[name], players):
                raise ValueError(f"Player partitions changed: {name}")
    context = train.TrainingContext(
        data,
        torch.from_numpy(data.tokens).to(arguments.device),
        torch.from_numpy(data.masks).to(arguments.device),
        torch.from_numpy(data.labels).to(arguments.device),
        arguments.output,
    )
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "inner": {},
        "evaluation": {},
    }
    diagnostics = {}
    for arm in ("A", "B"):
        state = torch.load(
            arguments.output / f"{arm}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = transformer_rank.RankTransformer(data.tokens.shape[2], 96).to(
            arguments.device
        )
        model.load_state_dict(state["state"])
        diagnostics[arm] = _diagnostics(model, context)
        diagnostics[arm]["selected_epoch"] = state["epoch"]
        for name, questions in data.questions.items():
            probabilities[name][arm] = train.question_probabilities(
                model, context, questions
            )
    results = {}
    for name, questions in data.questions.items():
        labels = np.asarray([data.labels[rows[0]] for rows in questions])
        results[name] = _comparison(
            probabilities[name],
            labels,
            np.asarray([len(rows) for rows in questions]),
        )
        np.savez_compressed(
            arguments.output / f"{name}_predictions.npz",
            players=data.splits[name],
            labels=labels,
            counts=np.asarray([len(rows) for rows in questions]),
            A=probabilities[name]["A"],
            B=probabilities[name]["B"],
        )
    report = {
        "results": results,
        "gate": _gate(results),
        "diagnostics": diagnostics,
        "data_audit": data.provenance.audit,
        "scope": "single-seed own branch; not full C1+C2; exploratory",
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot(context)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
