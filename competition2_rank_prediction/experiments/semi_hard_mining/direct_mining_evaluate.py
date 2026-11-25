"""Compares mining C to frozen matched A/B with player-level uncertainty.

Run after all three direct-mining fits finish. No winning seed is selected.
"""

import json

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.semi_hard_mining import (
    direct_mining_train,
)
from competition2_rank_prediction.experiments.semi_hard_mining import (
    multiseed_evaluate,
)
from competition2_rank_prediction.experiments.semi_hard_mining import (
    multiseed_train,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics
from competition2_rank_prediction.experiments.semi_hard_mining import train


def mining_audit(
    context: train.TrainingContext, seed: int
) -> dict[str, object]:
    """Verifies completed C budget, warmup identity and actual mining counts."""
    directory = context.output / f"seed_{seed}"
    history = json.loads((directory / "C_history.json").read_text())
    if len(history) != multiseed_train.EPOCHS:
        raise ValueError(f"C training is incomplete for seed {seed}")
    control = direct_mining_train.controls_directory(context.output)
    for arm in ("A", "B"):
        previous = json.loads(
            (control / f"seed_{seed}/{arm}_history.json").read_text()
        )
        if not all(
            row["inner"] == previous[index]["inner"]
            and row["train_nll_pair_gradnorm"]
            == previous[index]["train_nll_pair_gradnorm"]
            for index, row in enumerate(history[:5])
        ):
            raise ValueError(f"C warmup differs from {arm} for seed {seed}")
    counts = history[-1]["mining"]
    if counts["refreshes"] != 28 or counts["mined"] + counts["random"] != 14080:
        raise ValueError("Incorrect mining refresh or pair budget")
    if counts["mined_attempts"] != counts["mined"] + counts["fallback"]:
        raise ValueError("Inconsistent mining fallback accounting")
    if (
        counts["mined"] == 0
        or not 0 < counts["mined_margin_min"] <= counts["mined_margin_max"] <= 1
    ):
        raise ValueError("Mining did not produce valid positive-margin pairs")
    return {
        "seed": seed,
        "warmup_matches_A_and_B": True,
        "expected_selected_epoch": int(
            np.argmin([row["inner"]["nll"] for row in history])
        )
        + 1,
        "mining": counts,
        "epochs": len(history),
    }


def _c_probabilities(
    context: train.TrainingContext,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    parts: dict[str, list[np.ndarray]] = {"inner": [], "evaluation": []}
    audits = []
    for seed in multiseed_train.SEEDS:
        record = mining_audit(context, seed)
        state = torch.load(
            context.output / f"seed_{seed}/C.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            state["epoch"] != record["expected_selected_epoch"]
            or state["seed"] != seed
        ):
            raise ValueError("C checkpoint selection or seed mismatch")
        if state["recipe"] != {"seed": seed, "epochs": 60, "steps": 64}:
            raise ValueError("C checkpoint budget mismatch")
        for name, values in context.data.provenance.normalization.items():
            if not np.array_equal(state["normalization"][name], values):
                raise ValueError("C feature normalization mismatch")
        record["selected_epoch"] = state["epoch"]
        model = transformer_rank.RankTransformer(
            context.tokens.shape[2], 96
        ).to(context.tokens.device)
        model.load_state_dict(state["state"])
        for split, questions in context.data.questions.items():
            parts[split].append(
                train.question_probabilities(model, context, questions)
            )
        audits.append(record)
    return {split: np.stack(rows) for split, rows in parts.items()}, audits


def _comparison(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    counts: np.ndarray,
    reference: str,
) -> dict[str, object]:
    selected = {reference: values[reference], "C": values["C"]}
    report = multiseed_evaluate.summarize(
        selected, labels, counts, reference, "C"
    )
    deltas = []
    for index in range(3):
        deltas.append(
            statistics.player_scores(values["C"][index], labels)
            - statistics.player_scores(values[reference][index], labels)
        )
    report["paired_player_97_5ci"] = statistics.bootstrap_players(
        np.mean(deltas, axis=0),
        confidence=0.975,
    )
    return report


def _gate(reports: dict[str, dict]) -> dict[str, object]:
    evaluation = reports["evaluation"]
    versus_b = evaluation["C_vs_B"]
    metrics = versus_b["mean_seed_metrics"]
    conditions = {
        "positive_utility_vs_B": versus_b["primary_paired_player_95ci"]["delta"]
        > 0,
        "two_positive_seeds_vs_B": versus_b["positive_utility_seeds"] >= 2,
        "adjusted_interval_positive_vs_B": versus_b["paired_player_97_5ci"][
            "low"
        ]
        > 0,
        "adjusted_interval_positive_vs_A": evaluation["C_vs_A"][
            "paired_player_97_5ci"
        ]["low"]
        > 0,
        "pair_auc_not_below_B": metrics["C"]["pair_auc"]
        >= metrics["B"]["pair_auc"],
        "large_error_safe": metrics["C"]["large_error"]
        - metrics["B"]["large_error"]
        <= 0.01,
        "rank_losses_safe": min(
            versus_b["groups"][rank]["mean_utility_delta"]
            for rank in rank_lib.LEVELS
        )
        >= -0.05,
    }
    return {
        "conditions": conditions,
        "further_study_supported": all(conditions.values()),
        "production_adoption_permitted": False,
    }


def _plots(context: train.TrainingContext) -> None:
    controls = direct_mining_train.controls_directory(context.output)
    multiseed_evaluate.plot_curves(
        context,
        {"A": controls, "B": controls, "C": context.output},
        "ABC_training_curves.png",
    )


def _aligned_controls(
    context: train.TrainingContext,
    split: str,
    mining_values: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    questions = context.data.questions[split]
    labels = np.asarray([context.data.labels[rows[0]] for rows in questions])
    counts = np.asarray([len(rows) for rows in questions])
    source = direct_mining_train.controls_directory(context.output)
    with np.load(source / f"{split}_predictions.npz") as saved:
        for name, expected in (
            ("players", context.data.splits[split]),
            ("labels", labels),
            ("counts", counts),
        ):
            if not np.array_equal(saved[name], expected):
                raise ValueError(
                    f"Fixed question alignment differs: {split} {name}"
                )
        values = {
            "A": np.asarray(saved["A"]),
            "B": np.asarray(saved["B"]),
            "C": mining_values,
        }
    return values, labels, counts


def main() -> None:
    """Produces the fixed C-B primary and C-A control comparisons."""
    arguments = common.parse_arguments(
        __doc__ or "", direct_mining_train.OUTPUT_NAME, True
    )
    if (arguments.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite direct mining report")
    torch.set_num_threads(4)
    context = multiseed_train.load_context(arguments)
    protocol = json.loads((context.output / "protocol.json").read_text())
    if (
        direct_mining_train.verify_controls(context)
        != protocol["control_checkpoints_sha256"]
    ):
        raise ValueError("Frozen A/B checkpoints changed")
    probabilities, audits = _c_probabilities(context)
    reports = {}
    for split, mining_values in probabilities.items():
        values, labels, counts = _aligned_controls(
            context, split, mining_values
        )
        reports[split] = {
            f"C_vs_{arm}": _comparison(values, labels, counts, arm)
            for arm in ("A", "B")
        }
        multiseed_evaluate.write_predictions(context, split, values)
    report = {
        "results": reports,
        "gate": _gate(reports),
        "training_audit": audits,
        "scope": (
            "direct exploratory mining after prior gate failure; "
            "fixed seeds and reused cohort"
        ),
    }
    (context.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _plots(context)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
