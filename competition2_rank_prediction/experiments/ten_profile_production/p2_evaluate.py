"""P2: ten-profile d96 replacement inside the locked C1+C2 cross-fit, then under v47.

Stage A (secondary). The locked C1+C2 holdout3 cross-fit of
``crossfit_cohort/evaluate.py`` is replayed three times per half with
everything identical except the ``d96`` member, which is

* ``production``: the original ``d96_c<half>`` (must replay 0.7170117549);
* ``tenp3``: the ten-profile cache sliced to normal/5k/1d/5d;
* ``tenp10``: the full ten-profile cache.

``d96`` is replaced everywhere the pipeline reads it: as a fusion member
(weights refitted on validation) and as the player member of the C1 shift
residual (refitted on validation). ``res``, ``oppC2`` and the final-result
features are untouched.

Stage B (primary). The v47 opponent-identity layer is refitted on the two
holdout3 halves (fit on one, score the other) with its ``base`` replaced by
each arm's C1+C2 question probabilities. The original saved base is refitted
in the same run as the sanity anchor (about 0.7640). Fitting runs with
``torch.use_deterministic_algorithms(True)``, and the original base is
fitted twice to show that the refit is reproducible.

The v47 optimiser is sensitive to tiny base perturbations (float32 rounding of
an identical member already moved it by about 0.001), so a diagnostic refits
every arm on bases jittered by exp(1e-6 * N(0, 1)). It is reported next to the
gate and does not change it.

Gate (preregistered, stage B): v47(tenp10) - v47(tenp3) >= +0.003, paired
player-bootstrap 95% lower bound > 0, and both halves >= 0.

Example (research root, after p2_score.py for both folds):
    CUDA_VISIBLE_DEVICES=1 c1/.venv/bin/python -m \
        experiments.ten_profile_production.p2_evaluate \
        --probability-dir $P2/probabilities --output $P2/p2_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

# Deterministic cuBLAS for the stage-B refits; must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import importlib.util
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from scipy.stats import rankdata  # type: ignore[import-untyped]

# Project modules of the research root (run with -m from there).
import blend_eval  # pylint: disable=import-error
import rank_lib as lib  # pylint: disable=import-error
import residual_shift  # type: ignore[import-not-found]  # pylint: disable=import-error
# pylint: enable=wrong-import-position


def _load_crossfit() -> Any:
    """The locked C1+C2 cross-fit module, loaded by path like opponent_identity/full_qprob.py."""
    path = pathlib.Path(__file__).resolve().parents[1] / "crossfit_cohort/evaluate.py"
    spec = importlib.util.spec_from_file_location("crossfit_cohort_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crossfit = _load_crossfit()

ARMS = ("tenp3", "tenp10")
MEMBERS = ("d96", "res", "oppC2")
SEED = 20261119
GATE = 0.003
GAIN = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
DEFAULT_SIMS = pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt")
DEFAULT_QPROB = pathlib.Path("/tmp2/richard/claude_scratch/rank_probe/full_qprob.npz")


# --------------------------------------------------------------------------- shared metrics


def utility(qprob: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question competition utility of expected-utility decoding."""
    gap = np.abs((qprob @ GAIN).argmax(1) - truth)
    return np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))


def paired(
    new: np.ndarray,
    old: np.ndarray,
    players: np.ndarray,
    folds: np.ndarray,
    draws: int,
) -> dict:
    """Paired delta, player-bootstrap 95% CI and per-half deltas."""
    difference = new - old
    unique, inverse = np.unique(players, return_inverse=True)
    count = np.bincount(inverse)
    per_player = np.bincount(inverse, weights=difference) / count
    weight = count / count.sum()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(unique), size=(draws, len(unique)))
    # Resampled players keep their own question counts (all one on holdout3).
    means = (per_player[picks] * weight[picks]).sum(1) / weight[picks].sum(1)
    return {
        "delta": float(difference.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "by_half": {
            str(half): float(difference[folds == half].mean()) for half in (0, 1)
        },
        "improved": int((difference > 0).sum()),
        "worsened": int((difference < 0).sum()),
    }


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC (as evaluate_p1.auc)."""
    if positive.all() or not positive.any():
        return float("nan")
    ranks = rankdata(score, method="average")
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boundary_auc(qprob: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Adjacent pair-odds AUCs, log p(k+1) - log p(k) on questions of rank k or k+1
    (as evaluate_p1.boundary_auc)."""
    output = {}
    for low in range(9):
        mask = (truth == low) | (truth == low + 1)
        pair = np.clip(qprob[mask][:, [low, low + 1]], 1e-9, 1)
        name = f"{lib.LEVELS[low]}/{lib.LEVELS[low + 1]}"
        output[name] = auc(np.log(pair[:, 1]) - np.log(pair[:, 0]), truth[mask] == low + 1)
    return output


def describe(qprob: np.ndarray, truth: np.ndarray, folds: np.ndarray) -> dict:
    """Utility, halves, per-rank utility and adjacent-boundary AUCs of one arm."""
    score = utility(qprob, truth)
    prediction = (qprob @ GAIN).argmax(1)
    return {
        "utility": float(score.mean()),
        "exact": float((prediction == truth).mean()),
        "by_half": {str(half): float(score[folds == half].mean()) for half in (0, 1)},
        "per_rank_utility": {
            name: float(score[truth == rank].mean()) for rank, name in enumerate(lib.LEVELS)
        },
        "boundary_auc": boundary_auc(qprob, truth),
    }


def comparisons(
    qprobs: dict[str, np.ndarray],
    reference: str,
    labels: pd.DataFrame,
    truth: np.ndarray,
    draws: int,
) -> dict:
    """tenp10 - tenp3 (primary pairing) and each arm against the reference."""
    players = labels["player"].to_numpy().astype(str)
    folds = labels["fold"].to_numpy()
    scores = {name: utility(q, truth) for name, q in qprobs.items()}
    pairs = [("tenp10", "tenp3")] + [(arm, reference) for arm in ARMS]
    output = {}
    for new, old in pairs:
        result = paired(scores[new], scores[old], players, folds, draws)
        result["per_rank_delta"] = {
            name: float((scores[new] - scores[old])[truth == rank].mean())
            for rank, name in enumerate(lib.LEVELS)
        }
        result["changed_decisions"] = int(
            ((qprobs[new] @ GAIN).argmax(1) != (qprobs[old] @ GAIN).argmax(1)).sum()
        )
        output[f"{new}-{old}"] = result
    return output


def gate(result: dict) -> dict:
    """The preregistered P2 rule on one paired comparison."""
    return {
        "delta": result["delta"],
        "ci95_lower": result["ci95"][0],
        "by_half": result["by_half"],
        "passes": bool(
            result["delta"] >= GATE
            and result["ci95"][0] > 0
            and all(value >= 0 for value in result["by_half"].values())
        ),
    }


# --------------------------------------------------------------------------- stage A


def aligned_probabilities(path: pathlib.Path, game_keys: np.ndarray, clip: bool) -> np.ndarray:
    """Per-game probabilities in the requested game order.

    clip mirrors how the pipeline reads the production d96 member: validation
    probabilities are clipped to [1e-6, 1] (crossfit.aligned_val), holdout3 ones
    are used as dumped.
    """
    with np.load(path, allow_pickle=True) as saved:
        keys = np.asarray(saved["game_key"]).astype(str)
        # float32 like the production d96 files, so fusion runs at the same precision
        probabilities = np.asarray(saved["probabilities"], dtype=np.float32)
    row_of = {key: row for row, key in enumerate(keys)}
    missing = [key for key in game_keys.astype(str) if key not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} game keys missing from {path}")
    rows = np.asarray([row_of[key] for key in game_keys.astype(str)])
    return np.clip(probabilities[rows], 1e-6, 1.0) if clip else probabilities[rows]


def replay_half(
    half: Any,
    member: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, list[float]]:
    """C1+C2 question probabilities of one half with d96 := member (validation, test)."""
    replaced = half._replace(
        val={**half.val, "d96": member[0]},
        test={**half.test, "d96": member[1]},
    )
    validation_logs = np.log(np.stack([replaced.val[name] for name in MEMBERS]))
    test_logs = np.log(np.stack([replaced.test[name] for name in MEMBERS]))
    weights = blend_eval.fit_weights(validation_logs, replaced.vy)
    qprob = crossfit.residual_probs(replaced, "oppC2", (validation_logs, test_logs), weights)
    return qprob, np.asarray(weights).tolist()


def stage_a(  # pylint: disable=too-many-locals
    research: pathlib.Path,
    probability_dir: pathlib.Path,
    labels: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict]:
    """Replays C1+C2 with each d96 variant; returns question probabilities in labels order."""
    crossfit.DEV = torch.device("cpu")  # the legacy default is GPU 0, which hosts another service
    opponent_features = lib.load_features(research / "features/feat96opp/holdout3")
    final_results = (
        residual_shift.load_final(research / "features/final_result/val"),
        residual_shift.load_final(research / "features/final_result/holdout3"),
    )
    variants = ("production",) + ARMS
    qprobs = {name: np.full((len(labels), 10), np.nan) for name in variants}
    weights: dict[str, list[float]] = {}
    minimum: dict[str, float] = {}
    for fold in (0, 1):
        half, rows = crossfit.load_half(fold, labels, opponent_features, final_results)
        with np.load(crossfit.RES / f"val_probs_d96_c{fold}.npz", allow_pickle=True) as saved:
            validation_keys = np.asarray(saved["game_key"]).astype(str)
        with np.load(crossfit.member_dump(half.sfx), allow_pickle=True) as saved:
            test_keys = np.asarray(saved["game_key"]).astype(str)
        members = {"production": (half.val["d96"], half.test["d96"])}
        for arm in ARMS:
            members[arm] = (
                aligned_probabilities(
                    probability_dir / f"{arm}_c{fold}_validation.npz", validation_keys, True
                ),
                aligned_probabilities(
                    probability_dir / f"{arm}_c{fold}_holdout3.npz", test_keys, False
                ),
            )
        for name, (_, test) in members.items():
            minimum[f"{name} c{fold}"] = float(test.min())
        for name in variants:
            qprobs[name][rows], weights[f"{name} c{fold}"] = replay_half(half, members[name])
        print(
            f"stage A half {fold}: "
            + "  ".join(
                f"{name} {utility(qprobs[name][rows], half.ty).mean():.6f}"
                for name in variants
            ),
            flush=True,
        )
    if not all(np.isfinite(q).all() for q in qprobs.values()):
        raise ValueError("stage A left questions unscored")
    return qprobs, {
        "fusion_weights_d96_res_oppC2": weights,
        "min_holdout3_d96_probability": minimum,
    }


# --------------------------------------------------------------------------- stage B


class OpponentIdentity(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """The v47 model of experiments/opponent_identity/h3_final_v47.py, device-agnostic."""

    def __init__(self, ranks: torch.Tensor):
        super().__init__()
        self.ranks = ranks
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

    def forward(
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        game_question: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """Question logits from per-game class scores, game kinds and base probabilities."""
        usable = kind < 3
        posterior = (scores / self.log_temperature.exp()).softmax(1) @ self.ranks
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * self.prior.softmax(0)
        transition = self.offsets[:, self.offset_index].log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None] + transition[kind.clamp(max=2)], dim=1
        )
        per_game = per_game * usable[:, None]
        total = torch.zeros(len(base), 10, device=base.device).index_add_(
            0, game_question, per_game
        )
        used = torch.zeros(len(base), device=base.device).index_add_(
            0, game_question, usable.float()
        ).clamp(min=1)
        logits = self.log_alpha.exp() * total / used[:, None] ** self.gamma + self.bias
        return logits + torch.sigmoid(self.base_weight) * 2 * base.clamp(min=1e-9).log()


def v47_crossfit(
    inputs: dict,
    base: torch.Tensor,
    steps: int,
) -> np.ndarray:
    """Fit on one holdout3 half, predict the other (h3_final_v47.leak_check)."""
    arguments = (inputs["sims"], inputs["kind"], inputs["game_question"], base)
    truth = inputs["truth"]
    output = torch.zeros(len(truth), 10, device=base.device)
    for half in (0, 1):
        mask = torch.from_numpy(inputs["fold"] != half).to(base.device)
        model = OpponentIdentity(inputs["ranks"]).to(base.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(steps):
            loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            output[~mask] = model(*arguments).softmax(1)[~mask]
    return output.double().cpu().numpy()


def load_v47(path: pathlib.Path, device: torch.device) -> dict:
    """h3_sims_v47.pt prepared exactly as h3_final_v47.main does."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    sims = saved["sims"].float().to(device)
    sims[:, torch.tensor(saved["holdout_classes"], device=device)] = -5.0
    return {
        "ranks": F.one_hot(  # pylint: disable=not-callable
            torch.from_numpy(saved["player_rank"]), 10
        ).float().to(device),
        "sims": sims,
        "kind": torch.from_numpy(saved["kind"]).to(device),
        "game_question": saved["game_question"].to(device),
        "truth": saved["truth"].to(device),
        "fold": np.asarray(saved["fold"]),
        "base": saved["base"].float().to(device),
        "order": [str(q) for q in saved["order"]],
    }


def stage_b(  # pylint: disable=too-many-locals
    inputs: dict,
    qprobs: dict[str, np.ndarray],
    labels: pd.DataFrame,
    steps: int,
) -> tuple[dict[str, np.ndarray], dict, tuple[dict[str, torch.Tensor], np.ndarray]]:
    """v47 cross-fits for the saved base (twice) and every stage-A arm, in labels order;
    also returns the bases and the saved-to-labels row order for the diagnostic."""
    question_ids = labels["question_id"].astype(str).tolist()
    position = {q: i for i, q in enumerate(question_ids)}
    to_saved = np.asarray([position[q] for q in inputs["order"]])  # labels row per saved row
    if not np.array_equal(
        inputs["truth"].cpu().numpy(), lib.rank_ids(labels["rank"].to_numpy())[to_saved]
    ):
        raise ValueError("v47 truth differs from question_folds.csv")
    if not np.array_equal(inputs["fold"], labels["fold"].to_numpy()[to_saved]):
        raise ValueError("v47 folds differ from question_folds.csv")
    device = inputs["base"].device
    to_labels = np.argsort(to_saved)  # saved row per labels row
    bases = {"original": inputs["base"]}
    for name, qprob in qprobs.items():
        base = torch.from_numpy(qprob[to_saved]).float().to(device)  # as h3_dump.question_targets
        bases[name] = base / base.sum(1, keepdim=True)
    sanity = {
        "production_replay_vs_saved_base_max_abs_diff": float(
            (bases["production"] - bases["original"]).abs().max().item()
        )
    }
    outputs = {}
    for name, base in bases.items():
        outputs[name] = v47_crossfit(inputs, base, steps)[to_labels]
        score = utility(outputs[name], lib.rank_ids(labels["rank"].to_numpy())).mean()
        print(f"stage B {name}: v47 cross-fit {score:.6f}", flush=True)
    repeat = v47_crossfit(inputs, bases["original"], steps)[to_labels]
    sanity["original_refit_max_abs_diff"] = float(np.abs(repeat - outputs["original"]).max())
    return outputs, sanity, (bases, to_labels)


def refit_sensitivity(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    inputs: dict,
    bases: dict[str, torch.Tensor],
    to_labels: np.ndarray,
    truth: np.ndarray,
    steps: int,
    replicates: int,
    scale: float,
) -> dict:
    """Diagnostic only (not part of the gate): v47 refits after multiplying every base by
    exp(scale * N(0, 1)), the same draw for every arm within a replicate. It measures how
    much of a stage-B difference the v47 optimiser alone can produce."""
    generator = torch.Generator().manual_seed(SEED)
    names = ("original",) + ARMS
    scores: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        noise = torch.randn(bases["original"].shape, generator=generator).to(
            bases["original"].device
        )
        for name in names:
            base = bases[name] * torch.exp(scale * noise)
            output = v47_crossfit(inputs, base / base.sum(1, keepdim=True), steps)[to_labels]
            scores[name].append(float(utility(output, truth).mean()))
        print(f"refit sensitivity: {[values[-1] for values in scores.values()]}", flush=True)
    delta = np.asarray(scores["tenp10"]) - np.asarray(scores["tenp3"])
    return {
        "log_scale": scale,
        "replicates": replicates,
        "utility": scores,
        "utility_sd": {name: float(np.std(values, ddof=1)) for name, values in scores.items()},
        "tenp10-tenp3": delta.tolist(),
        "tenp10-tenp3_mean": float(delta.mean()),
        "tenp10-tenp3_sd": float(delta.std(ddof=1)),
    }


# --------------------------------------------------------------------------- driver


def parse_args() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--research", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--probability-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sims", type=pathlib.Path, default=DEFAULT_SIMS)
    parser.add_argument("--saved-qprob", type=pathlib.Path, default=DEFAULT_QPROB)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--jitter-refits", type=int, default=5,
                        help="diagnostic v47 refits on jittered bases (0 = skip)")
    parser.add_argument("--jitter-scale", type=float, default=1e-6)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def stage_a_sanity(
    research: pathlib.Path,
    saved_qprob: pathlib.Path,
    labels: pd.DataFrame,
    production: np.ndarray,
) -> dict:
    """Production replay against the recorded C1+C2 score and question probabilities."""
    reference = json.loads(
        (research / "experiments/crossfit_cohort/results/report.json").read_text(encoding="utf-8")
    )
    expected = float(reference["scores"]["C1+C2"])
    observed = float(utility(production, lib.rank_ids(labels["rank"].to_numpy())).mean())
    if abs(observed - expected) > 1e-9:
        raise ValueError(f"C1+C2 replay {observed:.10f} != recorded {expected:.10f}")
    with np.load(saved_qprob, allow_pickle=True) as saved:
        recorded_ids = np.asarray(saved["qid"]).astype(str)
        if not np.array_equal(recorded_ids, labels["question_id"].astype(str)):
            raise ValueError("full_qprob.npz question order differs")
        recorded = np.asarray(saved["qfull"], dtype=np.float64)
    return {
        "recorded_c1c2": expected,
        "replayed_c1c2": observed,
        "qprob_max_abs_diff_vs_full_qprob": float(np.abs(production - recorded).max()),
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Stage A, stage B, gate and report."""
    args = parse_args()
    torch.set_num_threads(8)
    labels = pd.read_csv(args.research / "features/holdout3/question_folds.csv")
    truth = lib.rank_ids(labels["rank"].to_numpy())
    folds = labels["fold"].to_numpy()

    qprobs, details = stage_a(args.research, args.probability_dir, labels)
    sanity_a = stage_a_sanity(args.research, args.saved_qprob, labels, qprobs["production"])
    print(json.dumps(sanity_a, indent=2), flush=True)

    torch.use_deterministic_algorithms(True)  # stage B only; stage A is the locked CPU replay
    inputs = load_v47(args.sims, torch.device(args.device))
    v47, sanity_b, (bases, to_labels) = stage_b(inputs, qprobs, labels, args.steps)
    sensitivity = (
        refit_sensitivity(
            inputs, bases, to_labels, truth, args.steps, args.jitter_refits, args.jitter_scale
        )
        if args.jitter_refits > 1
        else None
    )
    del inputs, bases
    sanity_b["original_v47_crossfit"] = float(utility(v47["original"], truth).mean())

    stage_a_report: dict[str, Any] = {
        **details,
        "sanity": sanity_a,
        "arms": {name: describe(q, truth, folds) for name, q in qprobs.items()},
        "paired": comparisons(qprobs, "production", labels, truth, args.bootstrap),
    }
    stage_b_report: dict[str, Any] = {
        "sanity": sanity_b,
        "arms": {name: describe(q, truth, folds) for name, q in v47.items()},
        "paired": comparisons(v47, "original", labels, truth, args.bootstrap),
        "refit_sensitivity_diagnostic": sensitivity,
    }
    report = {
        "questions": int(len(labels)),
        "players": int(labels["player"].nunique()),
        "gate_rule": (
            "v47(tenp10) - v47(tenp3) >= +0.003, paired player-bootstrap 95% lower "
            "bound > 0, both holdout3 halves >= 0"
        ),
        "gate_primary_stage_b": gate(stage_b_report["paired"]["tenp10-tenp3"]),
        "gate_secondary_stage_a": gate(stage_a_report["paired"]["tenp10-tenp3"]),
        "stage_a_c1c2": stage_a_report,
        "stage_b_v47": stage_b_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arrays: dict[str, Any] = {
        "question_id": labels["question_id"].to_numpy().astype(str),
        "truth": truth,
        "fold": folds,
        **{f"c1c2_{name}": q for name, q in qprobs.items()},
        **{f"v47_{name}": q for name, q in v47.items()},
    }
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps({k: report[k] for k in ("gate_primary_stage_b", "gate_secondary_stage_a")},
                     indent=2))


if __name__ == "__main__":
    main()
