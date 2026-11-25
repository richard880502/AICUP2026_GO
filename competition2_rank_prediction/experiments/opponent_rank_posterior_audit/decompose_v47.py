# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""P2: separate retrieval-value, transition sensitivity, and unresolved v47 errors.

Read-only on existing holdout3 v47 cache; never writes a submission or updates
production. The true opponent IDs/ranks are NOT in this cache. Hence this is
an observable counterfactual *diagnostic*, not oracle causal attribution.

From repo root on research host, with
AUDIT_DIR=/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit::
    CUDA_VISIBLE_DEVICES=1 runtime/competition2_research/c1/.venv/bin/python -m \
      competition2_rank_prediction.experiments.opponent_rank_posterior_audit.decompose_v47 \
      --output "$AUDIT_DIR/decomposition.json" \
      --device cuda:0

The full model mirrors experiments.ten_profile_production.p2_evaluate v47
(including float32 inputs, optimizer, deterministic algorithms). A production
replay at utility 0.765302 is mandatory by default. Both halves share no
players. Holdout3 has been inspected repeatedly: all findings exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import numpy as np
import torch
import torch.nn.functional as F
# pylint: enable=wrong-import-position

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
GAIN = np.eye(10) + np.exp(-1) * (np.eye(10, k=1) + np.eye(10, k=-1))
DEFAULT_STATE = Path("/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt")
DEFAULT_LABELS = Path("runtime/competition2_research/features/holdout3/question_folds.csv")
DEFAULT_EXPECTED = 0.765302
SEED = 20261120


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-instance-attributes
class V47(torch.nn.Module):
    """Device-agnostic, numerical reproduction of the frozen v47 model class.

    `mode='no_retrieval'` removes the *game-specific* rank posterior. Models
    trained with that mode retain kind, question game count, own-move base, and
    bias, so this is a conservative **refitted information ablation**.
    """

    def __init__(self, ranks: torch.Tensor, gallery_prior: torch.Tensor):
        super().__init__()
        self.register_buffer("ranks", ranks)
        self.register_buffer("gallery_prior", gallery_prior)
        self.log_temperature = torch.nn.Parameter(torch.tensor(np.log(0.04)))
        self.unknown = torch.nn.Parameter(torch.full((3,), -1.0))
        self.prior = torch.nn.Parameter(torch.zeros(10))
        self.offsets = torch.nn.Parameter(torch.zeros(3, 7))
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))
        self.bias = torch.nn.Parameter(torch.zeros(10))
        self.base_weight = torch.nn.Parameter(torch.tensor(0.0))
        offset = (torch.arange(10)[None] - torch.arange(10)[:, None]).clamp(-3, 3) + 3
        self.register_buffer("offset_index", offset.to(ranks.device))

    def rank_evidence(self, sims: torch.Tensor, temperature_multiplier: float = 1.0,
                      mode: str = "full") -> torch.Tensor:
        """Per-game rank posterior before the learned unknown-opponent mixture."""
        if mode == "no_retrieval":
            return self.gallery_prior[None, :].expand(len(sims), -1)
        if temperature_multiplier <= 0:
            raise ValueError("temperature_multiplier must be > 0")
        posterior = ((sims / self.log_temperature.exp()) * temperature_multiplier).softmax(1)
        return posterior @ self.ranks

    # Archived research script: one linear procedure, intentionally not split.
    # pylint: disable-next=too-many-arguments,too-many-locals
    def forward(self, sims: torch.Tensor, kind: torch.Tensor,
                game_question: torch.Tensor, base: torch.Tensor, *,
                mode: str = "full", temperature_multiplier: float = 1.0,
                transition_multiplier: float = 1.0) -> torch.Tensor:
        """Question logits. The defaults must replay production v47 exactly."""
        if transition_multiplier <= 0:
            raise ValueError("transition_multiplier must be > 0")
        usable = kind < 3
        posterior = self.rank_evidence(sims, temperature_multiplier, mode)
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * self.prior.softmax(0)
        transition = (self.offsets[:, self.offset_index] * transition_multiplier).log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None]
            + transition[kind.clamp(max=2)], dim=1,
        )
        per_game = per_game * usable[:, None]
        total = torch.zeros(len(base), 10, device=base.device).index_add_(
            0, game_question, per_game,
        )
        used = torch.zeros(len(base), device=base.device).index_add_(
            0, game_question, usable.float(),
        ).clamp(min=1)
        logits = self.log_alpha.exp() * total / used[:, None] ** self.gamma + self.bias
        return logits + torch.sigmoid(self.base_weight) * 2 * base.clamp(min=1e-9).log()


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-branches
def validate_metadata(saved: dict, records: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Validate true own-rank labels, fold/player isolation and game arrays."""
    order = np.asarray(saved["order"], dtype=str)
    if len(order) == 0 or len(order) != len(set(order)):
        raise ValueError("question IDs are missing or duplicated")
    labels = {row["question_id"]: row for row in records}
    if len(labels) != len(records) or set(labels) != set(order):
        raise ValueError("question_folds.csv must align 1:1 to saved v47 questions")
    truth = np.asarray(saved["truth"], dtype=np.int64)
    fold = np.asarray(saved["fold"], dtype=np.int64)
    label_truth = np.asarray([LEVELS.index(labels[q]["rank"]) for q in order])
    label_fold = np.asarray([int(labels[q]["fold"]) for q in order])
    players = np.asarray([labels[q]["player"] for q in order], dtype=str)
    if not (np.array_equal(truth, label_truth) and np.array_equal(fold, label_fold)):
        raise ValueError("v47 rank/fold labels differ from question_folds.csv")
    if set(fold) != {0, 1}:
        raise ValueError("expected exactly two v47 cross-fit folds")
    if set(players[fold == 0]).intersection(players[fold == 1]):
        raise ValueError("player leakage across the two holdout folds")
    if any(not str(x) for x in players):
        raise ValueError("missing player IDs")
    kinds = np.asarray(saved["kind"], dtype=np.int64)
    questions = np.asarray(saved["game_question"], dtype=np.int64)
    sims = np.asarray(saved["sims"])
    base = np.asarray(saved["base"])
    class_rank = np.asarray(saved["player_rank"], dtype=np.int64)
    holdout = np.asarray(saved["holdout_classes"], dtype=np.int64)
    if sims.ndim != 2 or sims.shape[1] != len(class_rank) or len(sims) != len(kinds):
        raise ValueError("class similarity/game shape mismatch")
    if len(questions) != len(kinds) or np.any(questions < 0) or np.any(questions >= len(order)):
        raise ValueError("game-to-question index out of bounds")
    if np.any(kinds < 0) or np.any(kinds > 3):
        raise ValueError("invalid game kinds; expected 0/1/2/3")
    if np.any(class_rank < 0) or np.any(class_rank > 9):
        raise ValueError("missing or invalid reference player rank")
    if (
        np.any(holdout < 0)
        or np.any(holdout >= len(class_rank))
        or len(set(holdout)) != len(holdout)
    ):
        raise ValueError("invalid/duplicate heldout class index")
    if base.shape != (len(order), 10) or not np.isfinite(base).all():
        raise ValueError("invalid saved C1+C2 probabilities")
    if np.any(base <= 0) or not np.allclose(base.sum(1), 1.0, atol=1e-4):
        raise ValueError("saved C1+C2 probabilities must be positive and normalized")
    if np.any(np.bincount(questions, minlength=len(order)) == 0):
        raise ValueError("question without a game")
    if not np.isfinite(sims).all():
        raise ValueError("nonfinite similarity scores")
    return {"order": order, "players": players, "truth": truth, "fold": fold,
            "kind": kinds, "game_question": questions, "class_rank": class_rank,
            "holdout": holdout}


def prepare(saved: dict, meta: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    """Apply the exact production class mask and create tensors."""
    sims = torch.as_tensor(saved["sims"], dtype=torch.float32, device=device).clone()
    holdout = meta["holdout"]
    sims[:, torch.as_tensor(holdout, device=device)] = -5.0
    class_rank = meta["class_rank"]
    keep = np.ones(len(class_rank), dtype=bool)
    keep[holdout] = False
    counts = np.bincount(class_rank[keep], minlength=10)
    if np.any(counts == 0):
        raise ValueError("no gallery classes remain in a rank")
    gallery_prior = torch.as_tensor(counts / counts.sum(), dtype=torch.float32, device=device)
    # pylint false positive: torch callables are not inferred.
    # pylint: disable-next=not-callable
    ranks = F.one_hot(torch.as_tensor(class_rank, device=device), 10).float()
    return {"sims": sims, "ranks": ranks, "gallery_prior": gallery_prior,
            "kind": torch.as_tensor(meta["kind"], dtype=torch.long, device=device),
            "game_question": torch.as_tensor(
                meta["game_question"], dtype=torch.long, device=device
            ),
            "base": torch.as_tensor(saved["base"], dtype=torch.float32, device=device),
            "truth": torch.as_tensor(meta["truth"], dtype=torch.long, device=device),
            "fold": meta["fold"], "counts": counts}


def decode(prob: np.ndarray) -> np.ndarray:
    """Competition expected-utility decoding, not argmax."""
    return (prob @ GAIN).argmax(1)


def utility(prob: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question competition utility."""
    delta = np.abs(decode(prob) - truth)
    return np.where(delta == 0, 1.0, np.where(delta == 1, np.exp(-1), 0.0))


def nll(prob: np.ndarray, truth: np.ndarray) -> float:
    """Question-level NLL. Predictions must be strictly positive."""
    return float(-np.log(np.clip(prob[np.arange(len(truth)), truth], 1e-12, 1)).mean())


def paired_delta(new: np.ndarray, old: np.ndarray, players: np.ndarray, folds: np.ndarray,
                 draws: int = 3000) -> dict[str, Any]:
    """Player-clustered bootstrap; differences refer to competition utility."""
    if len(new) == 0:
        return {"questions": 0, "delta": None, "ci95": None, "by_half": {}}
    delta = new - old
    unique, inverse = np.unique(players, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    generator = np.random.default_rng(SEED)
    sample = generator.integers(0, len(unique), size=(draws, len(unique)))
    samples = sums[sample].sum(1) / counts[sample].sum(1)
    return {"questions": int(len(delta)), "players": int(len(unique)),
            "delta": float(delta.mean()),
            "ci95": list(np.quantile(samples, (0.025, 0.975)).astype(float)),
            "by_half": {str(f): float(delta[folds == f].mean())
                        for f in (0, 1) if (folds == f).any()}}


def question_confidence(model: V47, ctx: dict, chosen: np.ndarray) -> np.ndarray:
    """Mean top *raw rank* posterior per question, using only HD0/HD1 games."""
    with torch.no_grad():
        posterior = model.rank_evidence(ctx["sims"]).detach().cpu().numpy()
    kinds = ctx["kind"].cpu().numpy()
    q = ctx["game_question"].cpu().numpy()
    valid = kinds < 3
    totals = np.bincount(q[valid], weights=posterior[valid].max(1), minlength=len(chosen))
    counts = np.bincount(q[valid], minlength=len(chosen))
    return np.divide(totals, counts, out=np.full(len(chosen), np.nan), where=counts > 0)


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def run_models(ctx: dict, steps: int) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    """Cross-fit frozen full v47 and a *re-fitted* no-game-specific-identity ablation.

    Counterfactual tempering uses the *same fitted full model*, never refits.
    """
    args = (ctx["sims"], ctx["kind"], ctx["game_question"], ctx["base"])
    truth, fold = ctx["truth"], ctx["fold"]
    scenarios = ("full", "no_retrieval", "retrieval_softer", "retrieval_sharper",
                 "transition_softer", "transition_sharper")
    result = {name: np.full((len(truth), 10), np.nan) for name in scenarios}
    confidence = np.full(len(truth), np.nan)
    fitted = []
    for held in (0, 1):
        train = torch.as_tensor(fold != held, device=truth.device)
        test = fold == held
        models = {}
        for name, mode in (("full", "full"), ("no_retrieval", "no_retrieval")):
            model = V47(ctx["ranks"], ctx["gallery_prior"]).to(truth.device)
            optim = torch.optim.Adam(model.parameters(), lr=0.05)
            for _ in range(steps):
                loss = F.cross_entropy(model(*args, mode=mode)[train], truth[train])
                optim.zero_grad()
                loss.backward()
                optim.step()
            models[name] = model
        with torch.no_grad():
            scenarios_t = {
                "full": models["full"](*args),
                "no_retrieval": models["no_retrieval"](*args, mode="no_retrieval"),
                "retrieval_softer": models["full"](*args, temperature_multiplier=0.75),
                "retrieval_sharper": models["full"](*args, temperature_multiplier=1.25),
                "transition_softer": models["full"](*args, transition_multiplier=0.75),
                "transition_sharper": models["full"](*args, transition_multiplier=1.25),
            }
            for name, logits in scenarios_t.items():
                result[name][test] = logits.softmax(1).detach().cpu().numpy()[test]
            confidence[test] = question_confidence(models["full"], ctx, ctx["fold"])[test]
        fitted.append({"target_half": held,
                       "full_temperature": float(models["full"].log_temperature.exp().detach()),
                       "full_unknown_by_kind": (
                           torch.sigmoid(models["full"].unknown).detach().cpu().tolist()
                       ),
                       "full_base_weight": float(
                           (2 * torch.sigmoid(models["full"].base_weight)).detach()
                       )})
    if not all(np.isfinite(probs).all() for probs in result.values()):
        raise ValueError("cross-fit left question probabilities unscored")
    return result, confidence, fitted


def rank_auc(prob: np.ndarray, truth: np.ndarray, boundary: int) -> float | None:
    """Boundary AUC on only adjacent classes using tie-averaged ranks."""
    mask = (truth == boundary) | (truth == boundary + 1)
    if (
        not mask.any()
        or not (truth[mask] == boundary).any()
        or not (truth[mask] == boundary + 1).any()
    ):
        return None
    sliced = np.clip(prob[mask][:, [boundary, boundary + 1]], 1e-12, 1)
    value = np.log(sliced[:, 1]) - np.log(sliced[:, 0])
    # Deferred import, as in the original script.
    # pylint: disable-next=import-outside-toplevel
    from scipy.stats import rankdata  # type: ignore[import-untyped]
    ranks = rankdata(value, method="average")
    positive = truth[mask] == boundary + 1
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def summarize(ctx: dict, meta: dict, probs: dict[str, np.ndarray], conf: np.ndarray,
              bootstrap: int) -> tuple[dict, dict]:
    """Observable evidence and perturbation sensitivity, without opponent oracle claims."""
    truth, fold, players = meta["truth"], meta["fold"], meta["players"]
    base = ctx["base"].cpu().numpy().astype(np.float64)
    output = {"base": base, **probs}
    kinds, q = meta["kind"], meta["game_question"]
    eligible = ~(np.bincount(q[kinds == 3], minlength=len(truth)) > 0)
    has_usable = np.bincount(q[kinds < 3], minlength=len(truth)) > 0
    compatible = eligible & has_usable
    correct_base = decode(base) == truth
    correct_full = decode(probs["full"]) == truth
    score = {name: utility(prob, truth) for name, prob in output.items()}
    summary: dict[str, Any] = {
        "status": "exploratory, no true opponent rank labels; no oracle attribution",
        "production_gate_passed": False,
        "questions": int(len(truth)),
        "players": int(len(np.unique(players))),
        "official_compatible_questions": int(compatible.sum()),
        "excluded_hd2plus_questions": int((~eligible).sum()),
        "usable_games": int((kinds < 3).sum()),
        "all_hd2plus_questions": int((~has_usable).sum()),
        "by_arm": {name: {"utility": float(values.mean()),
                          "nll": nll(output[name], truth),
                          "exact": float((decode(output[name]) == truth).mean()),
                          "utility_official_compatible": float(values[compatible].mean())}
                   for name, values in score.items()},
        "paired_vs_full": {
            name: paired_delta(score[name], score["full"], players, fold, bootstrap)
            for name in output if name != "full"
        },
        "retrieval_increment_full_minus_refitted_no_retrieval": paired_delta(
            score["full"], score["no_retrieval"], players, fold, bootstrap),
        "retrieval_increment_official_compatible": paired_delta(
            score["full"][compatible], score["no_retrieval"][compatible],
            players[compatible], fold[compatible], bootstrap),
        "hard_boundary_auc": {
            f"{LEVELS[i]}/{LEVELS[i+1]}": {
                name: rank_auc(prob, truth, i) for name, prob in output.items()
            } for i in (3, 4, 5)
        },
    }
    error_groups = {
        "base_and_full_correct": correct_base & correct_full,
        "retrieval_stack_rescues_base_error": (~correct_base) & correct_full,
        "retrieval_stack_hurts_base_correct": correct_base & (~correct_full),
        "both_wrong_unattributed": (~correct_base) & (~correct_full),
    }
    summary["base_full_error_table"] = {
        name: {"questions": int(mask.sum()),
               "mean_full_confidence_proxy": float(np.nanmean(conf[mask])) if mask.any() else None,
               "full_utility": float(score["full"][mask].mean()) if mask.any() else None}
        for name, mask in error_groups.items()
    }
    # Only analyze questions with at least one usable opponent game.
    median = float(np.nanmedian(conf[compatible]))
    error_full = ~correct_full
    sensitivity = {
        "retrieval": (decode(probs["retrieval_softer"]) != decode(probs["full"]))
                     | (decode(probs["retrieval_sharper"]) != decode(probs["full"])),
        "transition": (decode(probs["transition_softer"]) != decode(probs["full"]))
                      | (decode(probs["transition_sharper"]) != decode(probs["full"])),
    }
    sensitivity_masks = {
        "retrieval_only": sensitivity["retrieval"] & ~sensitivity["transition"],
        "transition_only": sensitivity["transition"] & ~sensitivity["retrieval"],
        "both": sensitivity["retrieval"] & sensitivity["transition"],
        "neither": ~sensitivity["retrieval"] & ~sensitivity["transition"],
    }
    summary["error_sensitivity_not_causal_attribution"] = {
        name: {"all_eligible": int((mask & compatible).sum()),
               "wrong_eligible": int((mask & error_full & compatible).sum()),
               "wrong_high_confidence": int(
                   (mask & error_full & compatible & (conf >= median)).sum()
               ),
               "wrong_low_confidence": int(
                   (mask & error_full & compatible & (conf < median)).sum()
               )}
        for name, mask in sensitivity_masks.items()
    }
    summary["confidence_bins"] = {
        "definition": (
            "raw opponent rank posterior top-rank mass; median of eligible OOF questions; "
            "not calibrated P(correct opponent rank)"
        ),
        "median": median,
        "low": {"questions": int((compatible & (conf < median)).sum()),
                "full_utility": float(score["full"][compatible & (conf < median)].mean())},
        "high": {"questions": int((compatible & (conf >= median)).sum()),
                 "full_utility": float(score["full"][compatible & (conf >= median)].mean())},
    }
    arrays = {"question_id": meta["order"], "player": players, "fold": fold,
              "truth": truth, "confidence_proxy": conf,
              "official_compatible": compatible, "valid_game_count": np.bincount(
                  q[kinds < 3], minlength=len(truth)),
              "sensitive_retrieval": sensitivity["retrieval"],
              "sensitive_transition": sensitivity["transition"],
              "base_correct": correct_base, "full_correct": correct_full,
              **{f"qprob_{name}": prob for name, prob in output.items()}}
    return summary, arrays


def main() -> None:
    """Run cross-fit controls, strict replay and print conservative diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--expected-utility", type=float, default=DEFAULT_EXPECTED)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    if args.steps < 1 or args.bootstrap < 100:
        parser.error("--steps >= 1 and --bootstrap >= 100 required")
    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    saved = torch.load(args.state, map_location="cpu", weights_only=False)
    metadata = validate_metadata(saved, labels)
    ctx = prepare(saved, metadata, torch.device(args.device))
    probabilities, confidence, fitted = run_models(ctx, args.steps)
    report, arrays = summarize(ctx, metadata, probabilities, confidence, args.bootstrap)
    observed = report["by_arm"]["full"]["utility"]
    report["sanity"] = {"expected_full_v47_utility": args.expected_utility,
                        "replayed_full_v47_utility": observed,
                        "absolute_difference": abs(observed - args.expected_utility),
                        "fitted_by_fold": fitted}
    # No output is written when the locked v47 replay fails: other comparisons
    # would not be interpretable until the mismatch is investigated.
    if abs(observed - args.expected_utility) > 1e-5:
        raise ValueError(f"v47 replay {observed:.8f} != expected {args.expected_utility:.8f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps({"sanity": report["sanity"],
                      "retrieval_increment": report[
                          "retrieval_increment_full_minus_refitted_no_retrieval"
                      ],
                      "error_sensitivity": report["error_sensitivity_not_causal_attribution"]},
                     indent=2), flush=True)
    print(f"Saved {args.output} and {args.output.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
