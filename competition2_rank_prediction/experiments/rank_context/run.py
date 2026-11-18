"""Cross-fit registered-rank context experiments against v47.

Variants:
  baseline  - exact v47: HD>=2 discarded.
  handicap  - include HD>=2 with exact handicap/receiver transition.
  context   - frozen v47 + linear question structural residual logits.
  both      - handicap transition + structural residual logits.

For each target fold, v47 is first fitted on the opposite fold. Variant models
are initialized from that fitted v47 state, all original v47 parameters are
frozen, and only the newly introduced parameters are optimized. This removes
baseline-refit improvements as a confound.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from competition2_rank_prediction.experiments.rank_context.model import (
    RankContextModel,
)

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
ADJACENT = float(np.exp(-1.0))
GAIN = torch.tensor(
    np.eye(10, dtype=np.float32)
    + ADJACENT
    * (
        np.eye(10, k=1, dtype=np.float32)
        + np.eye(10, k=-1, dtype=np.float32)
    )
)


def decode(probability: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Expected-competition-utility decoding."""
    return (probability @ gain).argmax(1)


def per_question_gain(
    probability: torch.Tensor,
    truth: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    """Official utility per question."""
    return gain[decode(probability, gain), truth]


def fit_baseline(
    model: RankContextModel,
    arguments: tuple[torch.Tensor, ...],
    truth: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    lr: float,
) -> RankContextModel:
    """Fits all original v47 parameters on one cross-fit training half."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def new_parameter_names(use_handicap: bool, use_context: bool) -> set[str]:
    """Parameter names introduced by the requested ablation."""
    names: set[str] = set()
    if use_handicap:
        names.update(
            {
                "hd_center_raw",
                "hd_slope_raw",
                "hd_sigma_raw",
                "hd_unknown",
                "hd_gate_raw",
            }
        )
    if use_context:
        names.add("context_head.weight")
    return names


def fit_new_parameters(
    model: RankContextModel,
    arguments: tuple[torch.Tensor, ...],
    truth: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    lr: float,
    context_l2: float,
    use_handicap: bool,
    use_context: bool,
) -> RankContextModel:
    """Fits only newly added parameters, keeping the v47 fit frozen."""
    trainable = new_parameter_names(use_handicap, use_context)
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
        if parameter.requires_grad:
            parameters.append(parameter)
    if not parameters:
        return model
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for _ in range(steps):
        logits = model(*arguments)
        loss = (
            F.cross_entropy(logits[mask], truth[mask])
            + model.regularization(context_l2)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def load_inputs(
    state_path: pathlib.Path,
    context_path: pathlib.Path,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    tuple[torch.Tensor, ...],
    list[str],
    list[str],
]:
    """Loads v47 state plus exact structural context and validates alignment."""
    saved = torch.load(
        state_path, map_location="cpu", weights_only=False
    )
    player_rank = np.asarray(saved["player_rank"], dtype=np.int64)
    ranks = F.one_hot(
        torch.from_numpy(player_rank).to(device),
        num_classes=10,
    ).float()

    scores = saved["sims"].float().to(device)
    holdout_classes = torch.as_tensor(
        saved["holdout_classes"],
        dtype=torch.long,
        device=device,
    )
    scores[:, holdout_classes] = -5.0
    kind = torch.as_tensor(
        saved["kind"], dtype=torch.long, device=device
    )
    game_question = torch.as_tensor(
        saved["game_question"], dtype=torch.long, device=device
    )
    truth = torch.as_tensor(
        saved["truth"], dtype=torch.long, device=device
    )
    fold = np.asarray(saved["fold"], dtype=np.int64)
    base = saved["base"].float().to(device)
    order = [str(value) for value in saved["order"]]

    with np.load(context_path) as context_file:
        context_order = [
            str(value) for value in context_file["question_id"]
        ]
        if context_order != order:
            raise ValueError("context question order != v47 state order")
        if not np.array_equal(
            np.asarray(context_file["game_question"], dtype=np.int64),
            game_question.cpu().numpy(),
        ):
            raise ValueError("context game_question != v47 state")
        handicap = torch.from_numpy(
            np.asarray(context_file["handicap"], dtype=np.int64)
        ).to(device)
        own_black = torch.from_numpy(
            np.asarray(context_file["own_black"], dtype=bool)
        ).to(device)
        context = torch.from_numpy(
            np.asarray(
                context_file["question_features"], dtype=np.float32
            )
        ).to(device)
        feature_names = [
            str(value) for value in context_file["feature_names"]
        ]

    if len(handicap) != len(scores):
        raise ValueError("context game count != v47 state game count")
    arguments = (
        scores,
        kind,
        game_question,
        base,
        handicap,
        own_black,
        context,
    )
    return ranks, fold, arguments, order, feature_names


def context_importance(
    model: RankContextModel,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean absolute class-logit coefficient for each structural feature."""
    if model.context_head is None:
        return {}
    magnitude = (
        model.context_head.weight.detach().abs().mean(0).cpu().numpy()
    )
    order = np.argsort(-magnitude)
    return {
        feature_names[int(index)]: float(magnitude[int(index)])
        for index in order
    }


def paired_interval(
    difference: np.ndarray,
    seed: int,
    samples: int,
) -> list[float]:
    """Paired question bootstrap interval for utility delta."""
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rows = rng.integers(0, len(difference), size=len(difference))
        values[index] = float(difference[rows].mean())
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def summarize_variant(
    probability: torch.Tensor,
    baseline: torch.Tensor,
    truth: torch.Tensor,
    fold: np.ndarray,
    has_hd2: np.ndarray,
    all_hd2: np.ndarray,
    gain: torch.Tensor,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict:
    """Scores overall, by half, and on HD>=2 affected questions."""
    new_gain = per_question_gain(probability, truth, gain).cpu().numpy()
    old_gain = per_question_gain(baseline, truth, gain).cpu().numpy()
    difference = new_gain - old_gain
    new_pred = decode(probability, gain).cpu().numpy()
    old_pred = decode(baseline, gain).cpu().numpy()
    truth_np = truth.cpu().numpy()

    def subset(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"questions": 0}
        return {
            "questions": int(mask.sum()),
            "baseline": float(old_gain[mask].mean()),
            "new": float(new_gain[mask].mean()),
            "delta": float(difference[mask].mean()),
            "exact_before": float(
                (old_pred[mask] == truth_np[mask]).mean()
            ),
            "exact_after": float(
                (new_pred[mask] == truth_np[mask]).mean()
            ),
        }

    return {
        "score": float(new_gain.mean()),
        "baseline": float(old_gain.mean()),
        "delta": float(difference.mean()),
        "paired_95_ci": paired_interval(
            difference, bootstrap_seed, bootstrap_samples
        ),
        "halves": {
            str(target): float(difference[fold == target].mean())
            for target in (0, 1)
        },
        "changed_answers": int((new_pred != old_pred).sum()),
        "off_by_one_before": float(
            (np.abs(old_pred - truth_np) == 1).mean()
        ),
        "off_by_one_after": float(
            (np.abs(new_pred - truth_np) == 1).mean()
        ),
        "subsets": {
            "has_hd2plus": subset(has_hd2),
            "no_hd2plus": subset(~has_hd2),
            "all_hd2plus": subset(all_hd2),
        },
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Runs v47, handicap, context, and combined two-fold comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt"
        ),
    )
    parser.add_argument(
        "--context",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "rank_context/holdout3_context.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent
        / "results"
        / "report.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-steps", type=int, default=300)
    parser.add_argument("--variant-steps", type=int, default=200)
    parser.add_argument("--baseline-lr", type=float, default=0.05)
    parser.add_argument("--variant-lr", type=float, default=0.03)
    parser.add_argument("--context-l2", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--gate", type=float, default=0.003)
    args = parser.parse_args()

    device = torch.device(args.device)
    ranks, fold, arguments, order, feature_names = load_inputs(
        args.state, args.context, device
    )
    truth = torch.as_tensor(
        torch.load(
            args.state, map_location="cpu", weights_only=False
        )["truth"],
        dtype=torch.long,
        device=device,
    )
    context_tensor = arguments[-1]
    feature_index = {
        name: index for index, name in enumerate(feature_names)
    }
    has_hd2 = (
        context_tensor[:, feature_index["hd2plus_fraction"]]
        .detach()
        .cpu()
        .numpy()
        > 0
    )
    game_question = arguments[2].detach().cpu().numpy()
    handicap = arguments[4].detach().cpu().numpy()
    all_hd2 = np.ones(len(order), dtype=bool)
    for question in range(len(order)):
        rows = game_question == question
        all_hd2[question] = bool(
            rows.any() and np.all(handicap[rows] >= 2)
        )

    gain = GAIN.to(device)
    variants = {
        "handicap": (True, False),
        "context": (False, True),
        "both": (True, True),
    }
    baseline_probability = torch.zeros(
        len(truth), 10, device=device
    )
    variant_probability = {
        name: torch.zeros(len(truth), 10, device=device)
        for name in variants
    }
    parameter_reports: dict[str, list[dict]] = {
        name: [] for name in variants
    }

    for target_fold in (0, 1):
        train_mask = torch.from_numpy(fold != target_fold).to(device)
        test_mask = ~train_mask

        baseline = RankContextModel(
            ranks,
            context_tensor.shape[1],
            use_handicap=False,
            use_context=False,
        ).to(device)
        baseline = fit_baseline(
            baseline,
            arguments,
            truth,
            train_mask,
            args.baseline_steps,
            args.baseline_lr,
        )
        baseline_state = copy.deepcopy(baseline.state_dict())
        with torch.no_grad():
            baseline_probability[test_mask] = baseline(
                *arguments
            ).softmax(1)[test_mask]

        for name, (use_handicap, use_context) in variants.items():
            model = RankContextModel(
                ranks,
                context_tensor.shape[1],
                use_handicap=use_handicap,
                use_context=use_context,
            ).to(device)
            model.load_state_dict(baseline_state, strict=False)
            model = fit_new_parameters(
                model,
                arguments,
                truth,
                train_mask,
                args.variant_steps,
                args.variant_lr,
                args.context_l2,
                use_handicap,
                use_context,
            )
            with torch.no_grad():
                variant_probability[name][test_mask] = model(
                    *arguments
                ).softmax(1)[test_mask]
            snapshot = {
                "target_fold": target_fold,
                **model.diagnostic_parameters(),
            }
            if use_context:
                snapshot["context_importance"] = context_importance(
                    model, feature_names
                )
            parameter_reports[name].append(snapshot)
        print(f"finished target fold {target_fold}", flush=True)

    baseline_gain = per_question_gain(
        baseline_probability, truth, gain
    ).cpu().numpy()
    report: dict[str, object] = {
        "baseline": {
            "score": float(baseline_gain.mean()),
            "exact": float(
                (
                    decode(baseline_probability, gain)
                    == truth
                )
                .float()
                .mean()
                .cpu()
            ),
            "questions": len(order),
            "questions_with_hd2plus": int(has_hd2.sum()),
            "questions_all_hd2plus": int(all_hd2.sum()),
        },
        "gate": args.gate,
        "protocol": (
            "Fit v47 on opposite fold, freeze all v47 parameters, "
            "fit only new parameters on that same training half."
        ),
        "variants": {},
    }
    for index, name in enumerate(variants):
        summary = summarize_variant(
            variant_probability[name],
            baseline_probability,
            truth,
            fold,
            has_hd2,
            all_hd2,
            gain,
            20261118 + index,
            args.bootstrap_samples,
        )
        low, _ = summary["paired_95_ci"]
        summary["passes_gate"] = bool(
            summary["delta"] >= args.gate
            and low > 0
            and all(value >= 0 for value in summary["halves"].values())
        )
        summary["fitted_parameters"] = parameter_reports[name]
        report["variants"][name] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
