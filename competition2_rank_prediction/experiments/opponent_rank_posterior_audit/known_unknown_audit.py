"""Frozen v4 known-vs-absent identity gate: player-disjoint control audit.

A control-only classifier; v47 transfer is explicitly a domain-shifted *proxy*.
Does not fit to holdout3 player ranks, create new submissions, or alter v47.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]
from scipy.stats import rankdata  # type: ignore[import-untyped]

SEED = 20261120
LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
GAIN = np.eye(10) + np.exp(-1) * (np.eye(10, k=1) + np.eye(10, k=-1))
DEFAULT_OUT = Path("/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit")
DEFAULT_PROBE = Path("/tmp2/richard/claude_scratch/opp_probe")


def player_folds(groups: np.ndarray) -> np.ndarray:
    """Stable group-disjoint folds; no game of one player crosses a fold."""
    labels = np.asarray(groups).astype(str)
    known = np.unique(labels)
    bucket = {g: hashlib.sha256(("known-unknown-v1:" + g).encode()).digest()[0] % 2
              for g in known}
    return np.asarray([bucket[g] for g in labels], dtype=np.int8)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Two-class AUC, ties handled by average ranks."""
    positive = (y == 1)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if not n1 or not n0:
        raise ValueError("AUC needs two classes")
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Give each class 0.5 weight; each player equal weight within class."""
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    weights = np.zeros(len(y), dtype=np.float64)
    for cls in (0, 1):
        ids, inverse, counts = np.unique(groups[y == cls], return_inverse=True,
                                          return_counts=True)
        if not len(ids):
            raise ValueError("train fold is missing a control class")
        weights[y == cls] = 0.5 / (len(ids) * counts[inverse])
    return weights


def features(top: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Only features available on BOTH existing known and absent controls."""
    x = np.column_stack((top, margin)).astype(np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("top cosine and top-2 margin must be finite and aligned")
    if (x[:, 1] < -1e-5).any():
        raise ValueError("top-2 margin cannot be negative")
    return x


def fit_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 ridge: float = 0.02) -> dict:
    """Fixed-capacity ridge logistic via deterministic Newton steps, player balanced."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-6)
    design = np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -20, 20)))
    weight = group_weights(y, groups)
    coefficient = np.zeros(3, dtype=np.float64)
    penalty = np.diag([0., ridge, ridge])
    for _ in range(80):
        probability = expit(design @ coefficient)
        gradient = design.T @ (weight * (probability - y)) + penalty @ coefficient
        curvature = design.T @ ((weight * probability * (1 - probability))[:, None] * design)
        delta = np.linalg.solve(curvature + penalty + 1e-9 * np.eye(3), gradient)
        coefficient -= np.clip(delta, -3, 3)
        if np.max(np.abs(delta)) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    """Score under a BALANCED known/unknown class prior, NOT deployment prior."""
    scaled = np.clip((x - model["mean"]) / model["scale"], -20, 20)
    return expit(model["coefficient"][0] + scaled @ model["coefficient"][1:])


def load_controls(path: Path, present_csv: Path, absent_games_per_player: int) -> dict:
    """Match controls.npz to original recorded present order and absent blocks."""
    with np.load(path) as saved:
        required = ("p_top", "p_margin", "a_top", "a_margin")
        if any(k not in saved for k in required):
            raise ValueError(f"{path} missing required control arrays {required}")
        p_top, p_margin, a_top, a_margin = [np.asarray(saved[k], np.float64)
                                           for k in required]
    with present_csv.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(p_top) or len(p_top) != len(p_margin):
        raise ValueError("present.csv must align 1:1 to controls p_top/p_margin")
    if len(a_top) != len(a_margin) or not len(a_top):
        raise ValueError("absent controls are empty or misaligned")
    if absent_games_per_player <= 0 or len(a_top) % absent_games_per_player:
        raise ValueError("absent control length incompatible with games/player")
    if len(a_top) != 18000 or absent_games_per_player != 15:
        raise ValueError("expected 1200 absent cohort players × 15 games; "
                         "inspect absent_controls() before changing grouping")
    player = np.asarray([row["player_id"] for row in rows], dtype=str)
    if any(not p for p in player) or any(not row.get("note") for row in rows):
        raise ValueError("present.csv is missing player ID or note")
    # All absent games are concatenated from 600x15 validation and 600x15 final.
    absent_group = np.asarray([f"absent-{i // absent_games_per_player:04d}"
                               for i in range(len(a_top))], dtype=str)
    known_group = np.asarray(["known-" + p for p in player], dtype=str)
    combined = {
        "x": np.concatenate([features(p_top, p_margin), features(a_top, a_margin)]),
        "y": np.r_[np.ones(len(p_top), dtype=np.int8),
                    np.zeros(len(a_top), dtype=np.int8)],
        "groups": np.r_[known_group, absent_group],
        "n_known": int(len(p_top)), "n_absent": int(len(a_top)),
    }
    if len(np.unique(known_group)) < 200 or len(np.unique(absent_group)) != 1200:
        raise ValueError("control group metadata is unexpectedly small")
    return combined


def crossfit(data: dict) -> tuple[dict, np.ndarray, list[dict]]:
    """Train on opposite control PLAYERS, report held-out known/absent detection."""
    x, y, groups = data["x"], data["y"], data["groups"]
    fold = player_folds(groups)
    if set(fold) != {0, 1} or set(groups[fold == 0]) & set(groups[fold == 1]):
        raise ValueError("control player-fold leakage or degenerate split")
    scores, cutoffs, fitted = np.full(len(y), np.nan), {}, []
    for held in (0, 1):
        train, test = fold != held, fold == held
        model = fit_logistic(x[train], y[train], groups[train])
        scores[test] = predict(model, x[test])
        train_absent = predict(model, x[train & (y == 0)])
        cutoffs[str(held)] = {str(fpr): float(np.quantile(train_absent, 1 - fpr))
                              for fpr in (0.01, 0.05)}
        fitted.append(model)
    if not np.isfinite(scores).all():
        raise ValueError("some control scores were not cross-fitted")
    result = {"sample_type": "known-v4-gallery vs never-seen-v4-cohort",
              "known_games": data["n_known"], "absent_games": data["n_absent"],
              "known_players": int(len(np.unique(groups[y == 1]))),
              "absent_players": int(len(np.unique(groups[y == 0]))),
              "control_auc": auc(y, scores),
              "control_balanced_brier": float(np.sum(group_weights(y, groups) * (scores-y)**2)),
              "known_score_median": float(np.median(scores[y == 1])),
              "absent_score_median": float(np.median(scores[y == 0])),
              "per_fold": {}, "predeclared_fpr_screens": {},
              "training_prior": "equal weight for each known/absent class and player",
              "score_interpretation": "balanced-prior gate score, NOT calibrated true P(known)"}
    for held in (0, 1):
        test = fold == held
        result["per_fold"][str(held)] = {
            "known_games": int((test & (y == 1)).sum()),
            "absent_games": int((test & (y == 0)).sum()),
            "auc": auc(y[test], scores[test])}
    for fpr in ("0.01", "0.05"):
        positive = np.asarray([scores[i] >= cutoffs[str(fold[i])][fpr]
                               for i in range(len(y))])
        fpr_observed = float(positive[y == 0].mean())
        tpr_observed = float(positive[y == 1].mean())
        result["predeclared_fpr_screens"][fpr] = {
            "actual_false_positive_rate": fpr_observed,
            "known_recall": tpr_observed,
            "precision_if_known_prevalence_0.50": float(
                .5*tpr_observed/(.5*tpr_observed+.5*fpr_observed)) if tpr_observed+fpr_observed else None,
            "precision_if_known_prevalence_0.70": float(
                .7*tpr_observed/(.7*tpr_observed+.3*fpr_observed)) if tpr_observed+fpr_observed else None,
            "threshold_by_held_fold": {k: c[fpr] for k, c in cutoffs.items()},
        }
    return result, scores, fitted


def _decode(prob: np.ndarray) -> np.ndarray:
    return (prob @ GAIN).argmax(1)


def proxy_v47(h3_v4: Path, decomposition: Path, models: list[dict],
               control_data: dict) -> tuple[dict, dict]:
    """EXPLORATORY v4 transfer onto v47 games. No held-out opponent ground truth."""
    import torch  # pylint: disable=import-outside-toplevel

    saved = torch.load(h3_v4, weights_only=False, map_location="cpu")
    expected = ("sims", "game_question", "order", "kind", "holdout_classes")
    if any(k not in saved for k in expected):
        raise ValueError("h3_sims.pt is missing v4 correspondence arrays")
    with np.load(decomposition) as out:
        need = ("question_id", "truth", "fold", "confidence_proxy", "qprob_full",
                "official_compatible", "player")
        if any(k not in out for k in need):
            raise ValueError("decomposition.npz missing mandatory OOF arrays")
        result = {key: np.asarray(out[key]) for key in need}
    order = np.asarray(saved["order"]).astype(str)
    row_by_question = {q: i for i, q in enumerate(order)}
    if len(row_by_question) != len(order) or set(order) != set(result["question_id"].astype(str)):
        raise ValueError("v4 h3 cache and v47 OOF questions do not match")
    index_out = {q: i for i, q in enumerate(result["question_id"].astype(str))}
    if len(index_out) != len(result["question_id"]):
        raise ValueError("duplicate v47 OOF question IDs")
    to_out = np.asarray([index_out[q] for q in order])
    if not np.array_equal(np.asarray(saved["truth"]), result["truth"][to_out]) or \
       not np.array_equal(np.asarray(saved["fold"]), result["fold"][to_out]):
        raise ValueError("v4 cache truth/folds disagree with OOF output")
    v4 = np.asarray(saved["sims"], dtype=np.float32)
    group = np.asarray(saved["game_question"], dtype=np.int64)
    kind = np.asarray(saved["kind"], dtype=np.int64)
    if len(v4) != len(group) or len(kind) != len(group) or len(v4) == 0:
        raise ValueError("v4 game arrays are unaligned")
    if np.any(group < 0) or np.any(group >= len(order)):
        raise ValueError("v4 game_question indices are invalid")
    if len(saved["holdout_classes"]):
        v4[:, np.asarray(saved["holdout_classes"], dtype=int)] = -5.0
    top2 = np.partition(v4, -2, axis=1)[:, -2:]
    x = features(top2.max(1), top2.max(1) - top2.min(1))
    # No heldout3 labels enter the gate fit. Two control-fitted models' average.
    game_score = (predict(models[0], x) + predict(models[1], x)) / 2
    usable = kind < 3
    counts = np.bincount(group[usable], minlength=len(order))
    question_score = np.bincount(group[usable], weights=game_score[usable],
                                 minlength=len(order)) / np.maximum(counts, 1)
    # Domain shift from v4 unmasked control gallery to holdout-masked h3 gallery.
    ctrl_known = control_data["x"][:control_data["n_known"], 0]
    ctrl_absent = control_data["x"][control_data["n_known"]:, 0]
    pooled = control_data["x"][:, 0]
    support_low, support_high = np.quantile(pooled, [.01, .99])
    transfer = {"status": "EXPLORATORY proxy, not a calibrated P(known) or opponent correctness",
                "gallery_shift": "holdout3 classes masked, unlike known/absent controls; class priors differ",
                "v4_h3_top_cosine_median": float(np.median(x[:, 0])),
                "known_control_top_cosine_median": float(np.median(ctrl_known)),
                "absent_control_top_cosine_median": float(np.median(ctrl_absent)),
                "v4_h3_game_score_median": float(np.median(game_score)),
                "usable_games": int(usable.sum()), "questions": len(order),
                "h3_v4_top_cosine_outside_control_1_to_99pct_fraction": float(
                    ((x[:, 0] < support_low) | (x[:, 0] > support_high)).mean()),
                "reliability_gate": "No v47 model change or production gate; masks and cohorts differ"}
    eligible = result["official_compatible"][to_out].astype(bool)
    expected_eligible = (np.bincount(group[kind == 3], minlength=len(order)) == 0) & (counts > 0)
    if not np.array_equal(eligible, expected_eligible):
        raise ValueError("v4 h3 game kinds disagree with OOF official-compatible mask")
    truth = result["truth"][to_out]
    probs = result["qprob_full"][to_out]
    v47conf = result["confidence_proxy"][to_out]
    errors = (_decode(probs) != truth)
    # Cutoffs from UNLABELLED distributions; never fit holdout3 truth.
    valid = eligible & np.isfinite(v47conf)
    if not valid.any():
        raise ValueError("no official-compatible questions for proxy diagnostic")
    def summarize(mask: np.ndarray) -> dict | None:
        if not mask.any():
            return None
        return {"questions": int(mask.sum()), "error_rate": float(errors[mask].mean()),
                "utility": float(np.where(abs(_decode(probs[mask])-truth[mask]) == 0, 1.,
                                  np.where(abs(_decode(probs[mask])-truth[mask]) == 1, np.exp(-1), 0.)).mean())}
    cut = float(np.median(question_score[valid]))
    transfer["median_split"] = {"threshold_from_unlabelled_proxy": cut,
                                 "low": summarize(valid & (question_score < cut)),
                                 "high": summarize(valid & (question_score >= cut))}
    a, b = question_score[valid], v47conf[valid]
    transfer["existing_v47_rankmass_confidence_correlation"] = (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None)
    # Compare within existing v47 posterior-confidence strata: descriptive only.
    edges = np.quantile(v47conf[valid], [0, .25, .5, .75, 1.])
    transfer["within_existing_confidence_quartiles"] = {}
    for idx in range(4):
        mask = valid & (v47conf >= edges[idx]) & (v47conf <= edges[idx+1] if idx == 3 else v47conf < edges[idx+1])
        if not mask.any():
            continue
        middle = float(np.median(question_score[mask]))
        lo, hi = mask & (question_score < middle), mask & (question_score >= middle)
        transfer["within_existing_confidence_quartiles"][str(idx+1)] = {
            "questions": int(mask.sum()),
            "low_proxy": summarize(lo) if lo.any() else None,
            "high_proxy": summarize(hi) if hi.any() else None}
    arrays = {"question_id": order, "v4_known_proxy": question_score,
              "existing_rankmass_confidence": v47conf,
              "official_compatible": eligible, "truth": truth, "v47_error": errors,
              "valid_game_count": counts, "v4_known_game_scores": game_score,
              "game_question": group}
    transfer["hard_1d_2d_subset"] = {
        "questions": int((valid & np.isin(truth, (4, 5))).sum()),
        "by_own_rank": {LEVELS[i]: summarize(valid & (truth == i)) for i in (4, 5)}
    }
    return transfer, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=DEFAULT_PROBE / "controls.npz")
    parser.add_argument("--present-csv", type=Path, default=DEFAULT_PROBE / "present.csv")
    parser.add_argument("--absent-games-per-player", type=int, default=15)
    parser.add_argument("--h3-v4", type=Path, default=None)
    parser.add_argument("--decomposition", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.h3_v4 is None) != (args.decomposition is None):
        parser.error("pass both --h3-v4 and --decomposition, or neither")
    data = load_controls(args.controls, args.present_csv, args.absent_games_per_player)
    scores, oof, models = crossfit(data)
    report = {"experiment": "known_unknown_v4_control_audit",
              "control": scores, "production_gate_passed": False,
              "v47_modified": False,
              "limitations": ["v4 known-control players were in encoder training; held-out games only",
                              "unknown-cohort controls contain different players and sampled positions",
                              "control data are not representative of actual opponent prevalence",
                              "holdout3 opponent true IDs/ranks are unavailable",
                              "masked v47 gallery makes transfer scores uncalibrated"]}
    arrays = {"control_score_oof": oof, "control_known_label": data["y"],
              "control_fold": player_folds(data["groups"]), "control_group": data["groups"]}
    if args.h3_v4 is not None:
        transfer, extra = proxy_v47(args.h3_v4, args.decomposition, models, data)
        report["v47_proxy_exploratory"] = transfer
        arrays.update({f"v47_{k}": v for k, v in extra.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps({"control": {"auc": scores["control_auc"],
                                 "fpr_screens": scores["predeclared_fpr_screens"]},
                      "v47_proxy": report.get("v47_proxy_exploratory", {"status": "not run"}),
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()