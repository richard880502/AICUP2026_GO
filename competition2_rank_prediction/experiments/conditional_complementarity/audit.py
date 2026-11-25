"""Exploratory confidence-stratified tenp10 residual audit on frozen v47 OOF.

No source models are retrained, no predictions are overwritten, and no
production/submission gate can pass on this repeatedly inspected holdout3.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ADJACENT = float(np.exp(-1))
GAIN = np.eye(10) + ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
ALPHAS = (0.0, 0.25, 0.5, 1.0)  # fixed before touching evaluation labels
SEED = 20261120


def required(npz: Any, key: str) -> np.ndarray:
    """Fetch a required array without silent fallbacks."""
    if key not in npz:
        raise ValueError(f"missing {key!r} in supplied cache")
    return np.asarray(npz[key])


def probabilities(arr: np.ndarray, name: str, n: int) -> np.ndarray:
    """Check that a cached probability tensor is aligned and usable."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (n, 10) or not np.isfinite(arr).all():
        raise ValueError(f"invalid probabilities: {name} has shape {arr.shape}")
    if (arr < 0).any() or not np.allclose(arr.sum(1), 1, atol=2e-4):
        raise ValueError(f"invalid probability values: {name}")
    return np.maximum(arr, 1e-9)


def align_unique(question_id: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
    """Exact 1:1 alignment; no positional guesses."""
    ids = [str(x) for x in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("source contains duplicate question ids")
    mapping = {qid: i for i, qid in enumerate(ids)}
    requested = [str(q) for q in question_id]
    if len(requested) != len(set(requested)) or set(requested) != set(ids):
        raise ValueError("question sets differ between caches")
    return np.asarray([mapping[q] for q in requested], dtype=np.int64)


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def load_inputs(p2: Path, confidence: Path, labels: Path) -> dict[str, np.ndarray]:
    """Strictly align P2, confidence cache and question metadata by question_id."""
    with np.load(p2, allow_pickle=False) as src:
        qid = required(src, "question_id").astype(str)
        truth = required(src, "truth").astype(np.int64)
        fold = required(src, "fold").astype(np.int64)
        variants = {
            key: probabilities(required(src, "v47_" + key), key, len(qid))
            for key in ("original", "tenp3", "tenp10")
        }
    if len(qid) != len(truth) or len(qid) != len(fold):
        raise ValueError("P2 question array lengths disagree")
    if not np.isin(fold, [0, 1]).all() or set(fold) != {0, 1}:
        raise ValueError("P2 cross-fit must contain folds 0 and 1")
    if (truth < 0).any() or (truth > 9).any():
        raise ValueError("invalid rank labels")
    with np.load(confidence, allow_pickle=False) as src:
        order = align_unique(qid, required(src, "question_id"))
        margin = required(src, "identity_margin")[order].astype(float)
        usable = required(src, "usable_games")[order].astype(int)
        all_games = required(src, "all_games")[order].astype(int)
        hd2plus = required(src, "has_hd2plus")[order].astype(bool)
    with labels.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"question_id", "player", "rank", "fold"}.issubset(rows[0]):
        raise ValueError("labels must contain question_id, player, rank and fold")
    row_of = {str(row["question_id"]): row for row in rows}
    if len(row_of) != len(rows) or set(row_of) != set(qid):
        raise ValueError("label question ids do not match P2")
    metadata = [row_of[q] for q in qid]
    if any(LEVELS[int(y)] != row["rank"] or int(f) != int(row["fold"])
           for y, f, row in zip(truth, fold, metadata)):
        raise ValueError("truth/fold disagree with question_folds.csv")
    if (usable > all_games).any() or (all_games <= 0).any():
        raise ValueError("invalid question game counts")
    return {
        "question_id": qid, "truth": truth, "fold": fold,
        "player": np.asarray([row["player"] for row in metadata], dtype=str),
        "margin": margin, "usable": usable, "all_games": all_games,
        "has_hd2plus": hd2plus,
        **variants,
    }


def scores(prob: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Official utility: exact 1, adjacent exp(-1), else 0."""
    decision = np.argmax(prob @ GAIN, axis=1)
    distance = np.abs(decision - truth)
    return np.where(distance == 0, 1.0, np.where(distance == 1, ADJACENT, 0.0))


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Stable rowwise log softmax."""
    maximum = z.max(axis=1, keepdims=True)
    centered = z - maximum
    return centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))


def make_residual(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Match the two retrained arms, then anchor to the frozen original v47."""
    base = np.log(data["original"])
    diff = np.log(data["tenp10"]) - np.log(data["tenp3"])
    diff -= diff.mean(axis=1, keepdims=True)
    return base, np.clip(diff, -2.0, 2.0)


def fuse(base: np.ndarray, diff: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """Renormalized frozen production + bounded tenp10-vs-tenp3 logit residual."""
    alpha = np.asarray(alpha)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    return np.exp(log_softmax(base + alpha * diff))


def nll(prob: np.ndarray, truth: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (probabilities floored at 1e-9)."""
    return float(-np.log(np.maximum(prob[np.arange(len(truth)), truth], 1e-9)).mean())


def choose_global(base: np.ndarray, diff: np.ndarray, truth: np.ndarray) -> float:
    """Choose one nonnegative coefficient by train-half NLL only."""
    return float(min(ALPHAS, key=lambda a: nll(fuse(base, diff, a), truth)))


def choose_conditional(
    base: np.ndarray, diff: np.ndarray, truth: np.ndarray, low: np.ndarray,
) -> tuple[float, float]:
    """Two weights, trained only on the training half; tiny fixed search space."""
    return min(
        ((a, b) for a in ALPHAS for b in ALPHAS),
        key=lambda pair: nll(
            fuse(base, diff, np.where(low, pair[0], pair[1])), truth
        ),
    )


def bootstrap_delta(new: np.ndarray, old: np.ndarray, players: np.ndarray,
                    draws: int) -> list[float]:
    """Resample players rather than potentially overlapping questions."""
    unique, inv = np.unique(players, return_inverse=True)
    per_player = np.bincount(inv, weights=new-old) / np.bincount(inv)
    count = np.bincount(inv)
    rng = np.random.default_rng(SEED)
    samples: list[float] = []
    for _ in range(0, draws, 256):
        width = min(256, draws - len(samples))
        selection = rng.integers(len(unique), size=(width, len(unique)))
        # Player bootstrap, accounting for unequal question counts per player.
        samples.extend((
            (per_player[selection] * count[selection]).sum(1) /
            count[selection].sum(1)
        ).tolist())
    return np.quantile(samples, [0.025, 0.975]).tolist()


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def pair_report(
    new: np.ndarray, old: np.ndarray, truth: np.ndarray, fold: np.ndarray,
    player: np.ndarray, mask: np.ndarray, draws: int,
) -> dict[str, Any]:
    """Paired utility on one pre-defined cohort/subgroup."""
    if not mask.any():
        return {"n": 0, "delta": None, "ci95": None}
    ns = scores(new[mask], truth[mask])
    os = scores(old[mask], truth[mask])
    difference = ns - os
    return {
        "n": int(mask.sum()),
        "players": int(len(np.unique(player[mask]))),
        "baseline_utility": float(os.mean()),
        "candidate_utility": float(ns.mean()),
        "delta": float(difference.mean()),
        "ci95": bootstrap_delta(ns, os, player[mask], draws),
        "changed": int((np.argmax(new[mask] @ GAIN, 1) != np.argmax(old[mask] @ GAIN, 1)).sum()),
        "better": int((difference > 0).sum()),
        "worse": int((difference < 0).sum()),
        "halves": {
            str(k): (float(difference[fold[mask] == k].mean())
                     if np.any(fold[mask] == k) else None)
            for k in (0, 1)
        },
    }


def boundary_auc(prob: np.ndarray, truth: np.ndarray, boundary: int,
                 eligible: np.ndarray) -> float | None:
    """Adjacent rank pair-odds AUC; no post-hoc sign flipping."""
    use = eligible & ((truth == boundary) | (truth == boundary + 1))
    label = truth[use] == boundary + 1
    if not label.any() or label.all():
        return None
    pair_log = np.log(prob[use, boundary + 1]) - np.log(prob[use, boundary])
    # Tied ranks use average rank (equivalent to the Mann-Whitney statistic).
    from scipy.stats import rankdata  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,import-error
    ranks = rankdata(pair_log)
    npos, nneg = int(label.sum()), int((~label).sum())
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def run(data: dict[str, np.ndarray], draws: int = 4000,
        official_compatible_only: bool = True) -> tuple[dict, dict[str, np.ndarray]]:
    """Fixed fold-wise train-half NLL; evaluate the opposite half once."""
    truth, fold, player = data["truth"], data["fold"], data["player"]
    eligible = (~data["has_hd2plus"] if official_compatible_only
                else np.ones(len(truth), dtype=bool)) & (data["usable"] > 0)
    if not eligible.any() or not np.isfinite(data["margin"][eligible]).all():
        raise ValueError("no eligible questions or missing identity confidence")
    base, diff = make_residual(data)
    variants = {key: np.zeros_like(data["original"]) for key in ("global", "conditional")}
    low_conf = np.zeros(len(truth), dtype=bool)
    fitted = {}
    for heldout in (0, 1):
        train = eligible & (fold != heldout)
        test = eligible & (fold == heldout)
        if train.sum() < 10 or test.sum() < 10:
            raise ValueError("need >=10 eligible questions per fold")
        threshold = float(np.median(data["margin"][train]))
        low_conf[test] = data["margin"][test] < threshold
        low_train = data["margin"][train] < threshold
        g = choose_global(base[train], diff[train], truth[train])
        a_low, a_high = choose_conditional(
            base[train], diff[train], truth[train], low_train
        )
        variants["global"][test] = fuse(base[test], diff[test], g)
        variants["conditional"][test] = fuse(
            base[test], diff[test], np.where(low_conf[test], a_low, a_high)
        )
        fitted[str(heldout)] = {
            "train_questions": int(train.sum()), "test_questions": int(test.sum()),
            "median_identity_margin_train": threshold,
            "alpha_global": g, "alpha_conditional_low": a_low,
            "alpha_conditional_high": a_high,
        }
    reference = data["original"]
    common = (truth, fold, player, eligible, draws)
    full = {
        "tenp10_vs_tenp3": pair_report(
            data["tenp10"], data["tenp3"], *common
        ),
        "tenp10_vs_original": pair_report(data["tenp10"], reference, *common),
        "global_vs_original": pair_report(variants["global"], reference, *common),
        "conditional_vs_original": pair_report(variants["conditional"], reference, *common),
    }
    strata = {}
    for name, mask in (("low_identity_margin", low_conf),
                       ("high_identity_margin", eligible & ~low_conf)):
        strata[name] = {
            "tenp10_vs_tenp3": pair_report(
                data["tenp10"], data["tenp3"], truth, fold, player,
                mask & eligible, draws,
            ),
            "conditional_vs_original": pair_report(
                variants["conditional"], reference, truth, fold, player,
                mask & eligible, draws,
            ),
        }
    hard = {}
    for b in (3, 4, 5):
        key = f"{LEVELS[b]}/{LEVELS[b+1]}"
        hard[key] = {
            "original_auc": boundary_auc(reference, truth, b, eligible),
            "conditional_auc": boundary_auc(variants["conditional"], truth, b, eligible),
        }
    candidate = full["conditional_vs_original"]
    screen = bool(
        candidate["delta"] >= 0.005 and candidate["ci95"][0] > 0
        and all(value is not None and value >= 0 for value in candidate["halves"].values())
    )
    report = {
        "status": "exploratory_only_not_approved_for_production",
        "reason": ("holdout3 has been inspected repeatedly and the v47 P2 base models "
                   "cross-fit across the same halves; fresh nested outer evaluation required"),
        "official_compatible_only": official_compatible_only,
        "n_total": int(len(truth)),
        "n_eligible": int(eligible.sum()),
        "n_excluded_hd2plus": int(data["has_hd2plus"].sum()),
        "n_eligible_players": int(len(np.unique(player[eligible]))),
        "confidence_proxy": (
            "mean top-2 v47 class cosine margin per HD0/HD1 game; not known-player probability"
        ),
        "fixed_alpha_grid": list(ALPHAS),
        "fitted_on_opposite_half": fitted,
        "paired": full,
        "strata": strata,
        "hard_boundaries": hard,
        "exploratory_screen_rule": "delta>=0.005, player-bootstrap CI lower>0, both halves>=0",
        "exploratory_screen_only": screen,
        "production_gate_passed": False,
    }
    outputs = {
        "question_id": data["question_id"],
        "eligible": eligible, "low_confidence": low_conf,
        "identity_margin": data["margin"],
        "original": reference,
        "global": variants["global"],
        "conditional": variants["conditional"],
    }
    return report, outputs


def main() -> None:
    """Analyze existing P2 predictions without changing any deployed artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True, help="P2 p2_report.npz")
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--include-hd2plus", action="store_true",
                        help="diagnostic only: official test has no HD>=2")
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("bootstrap draws must be >=100")
    data = load_inputs(args.p2, args.confidence, args.labels)
    report, outputs = run(data, args.bootstrap, not args.include_hd2plus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **outputs)  # type: ignore[arg-type]
    print(json.dumps({"paired": report["paired"], "exploratory_screen":
                      report["exploratory_screen_only"]}, indent=2))
    print(f"exploratory only; reports at {args.output}")


if __name__ == "__main__":
    main()
