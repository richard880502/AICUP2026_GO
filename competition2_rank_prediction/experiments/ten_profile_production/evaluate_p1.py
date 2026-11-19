"""Matched P1 evaluation: production Transformer with 3 vs 10 HumanSL profiles.

Both arms are trained from the SAME ten-profile extraction. The 3-profile arm
selects indices 0,3,5,9 (normal, 5k, 1d, 5d), so extraction randomness,
game/move sampling, labels, architecture and training budget are matched.

Primary endpoint:
    question-level logmean + expected-utility decoding, 30 questions/player,
    fixed sampling seed 0.

Gate:
    ten - three >= +0.003 and player-bootstrap 95% lower bound > 0.

Secondary diagnostics include game-count buckets, adjacent-boundary pair-odds
AUC, per-rank utility and ten alternative question-sampling seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import rankdata

import rank_lib as lib

SEED = 20261119
BOUNDARIES = tuple(f"{lib.LEVELS[i]}/{lib.LEVELS[i + 1]}" for i in range(9))


def auc(score: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC for binary 0/1 labels."""
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(score, method="average")
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def score_vector(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question official utility."""
    gap = np.abs(prediction - truth)
    return np.where(
        gap == 0,
        1.0,
        np.where(gap == 1, lib.ADJACENT, 0.0),
    )


def load_prob(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Loads one transformer validation-probability file."""
    with np.load(path, allow_pickle=True) as saved:
        return {
            "prob": np.asarray(saved["transformer"], dtype=np.float64),
            "truth": np.asarray(saved["y"], dtype=np.int64),
            "player": saved["player"].astype(str),
            "game_key": saved["game_key"].astype(str),
        }


def assert_matched(three: dict[str, np.ndarray], ten: dict[str, np.ndarray]) -> None:
    """Refuse an unfair comparison."""
    for key in ("truth", "player", "game_key"):
        if not np.array_equal(three[key], ten[key]):
            raise ValueError(f"3-profile and 10-profile {key} arrays differ")
    for name, data in (("three", three), ("ten", ten)):
        prob = data["prob"]
        if prob.shape != (len(data["truth"]), 10):
            raise ValueError(f"{name}: bad probability shape {prob.shape}")
        if not np.isfinite(prob).all():
            raise ValueError(f"{name}: non-finite probabilities")


def question_rows(
    players: np.ndarray,
    truth: np.ndarray,
    per_player: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Fixed 5-20 game question samples grouped by player."""
    rng = np.random.default_rng(seed)
    members: list[np.ndarray] = []
    question_player: list[str] = []
    labels: list[int] = []
    sizes: list[int] = []
    for player in np.unique(players):
        rows = np.flatnonzero(players == player)
        if len(rows) < 5:
            continue
        label_values = np.unique(truth[rows])
        if len(label_values) != 1:
            raise ValueError(f"player {player} has multiple labels")
        for _ in range(per_player):
            size = int(rng.integers(5, min(20, len(rows)) + 1))
            members.append(rng.choice(rows, size=size, replace=False))
            question_player.append(player)
            labels.append(int(label_values[0]))
            sizes.append(size)
    return (
        members,
        np.asarray(question_player),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64),
    )


def pool_questions(prob: np.ndarray, members: list[np.ndarray]) -> np.ndarray:
    """Production-style log-mean pooling."""
    return np.stack([lib.pool(prob[rows], "logmean") for rows in members])


def paired_bootstrap(
    new: np.ndarray,
    old: np.ndarray,
    players: np.ndarray,
    draws: int,
) -> dict:
    """Paired bootstrap over players, not overlapping sampled questions."""
    unique, inverse = np.unique(players, return_inverse=True)
    count = np.bincount(inverse)
    per_player = (
        np.bincount(inverse, weights=new - old) / np.maximum(count, 1)
    )
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(unique), size=(draws, len(unique)))
    means = per_player[picks].mean(1)
    return {
        "delta": float((new - old).mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def summarize_arm(
    qprob: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> dict:
    """Utility and error summaries."""
    pred = lib.decode(qprob)
    score = score_vector(pred, labels)
    gap = np.abs(pred - labels)
    result = {
        "utility": float(score.mean()),
        "exact": float((gap == 0).mean()),
        "adjacent_error": float((gap == 1).mean()),
        "far_error": float((gap >= 2).mean()),
        "by_games": {},
        "per_rank_utility": {},
    }
    for label, mask in (
        ("5-9", sizes <= 9),
        ("10-14", (sizes >= 10) & (sizes <= 14)),
        ("15-20", sizes >= 15),
    ):
        result["by_games"][label] = {
            "n": int(mask.sum()),
            "utility": float(score[mask].mean()),
        }
    for rank, name in enumerate(lib.LEVELS):
        mask = labels == rank
        result["per_rank_utility"][name] = float(score[mask].mean())
    return result


def boundary_auc(qprob: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Adjacent pair-odds AUCs."""
    output = {}
    for low, name in enumerate(BOUNDARIES):
        mask = (labels == low) | (labels == low + 1)
        pair = qprob[mask][:, [low, low + 1]]
        score = np.log(np.clip(pair[:, 1], 1e-9, 1)) - np.log(
            np.clip(pair[:, 0], 1e-9, 1)
        )
        output[name] = auc(score, (labels[mask] == low + 1).astype(np.int64))
    return output


def one_sampling(
    three: dict[str, np.ndarray],
    ten: dict[str, np.ndarray],
    per_player: int,
    seed: int,
    bootstrap: int,
) -> dict:
    """One matched question-sampling realization."""
    members, q_player, labels, sizes = question_rows(
        three["player"], three["truth"], per_player, seed
    )
    p3 = pool_questions(three["prob"], members)
    p10 = pool_questions(ten["prob"], members)
    pred3, pred10 = lib.decode(p3), lib.decode(p10)
    s3, s10 = score_vector(pred3, labels), score_vector(pred10, labels)
    paired = paired_bootstrap(s10, s3, q_player, bootstrap)
    return {
        "seed": seed,
        "questions": int(len(labels)),
        "players": int(len(np.unique(q_player))),
        "three": summarize_arm(p3, labels, sizes),
        "ten": summarize_arm(p10, labels, sizes),
        "paired_ten_minus_three": paired,
        "boundary_auc": {
            "three": boundary_auc(p3, labels),
            "ten": boundary_auc(p10, labels),
        },
        "decision_changes": {
            "changed": int((pred3 != pred10).sum()),
            "improved": int((s10 > s3).sum()),
            "worsened": int((s10 < s3).sum()),
            "same_utility": int((s10 == s3).sum()),
        },
    }


def main() -> None:
    """Runs the preregistered primary comparison and sampling sensitivity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three", type=pathlib.Path, required=True)
    parser.add_argument("--ten", type=pathlib.Path, required=True)
    parser.add_argument("--questions-per-player", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--sensitivity-seeds", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    three = load_prob(args.three)
    ten = load_prob(args.ten)
    assert_matched(three, ten)

    primary = one_sampling(
        three, ten, args.questions_per_player, 0, args.bootstrap
    )
    sensitivity = [
        one_sampling(
            three,
            ten,
            args.questions_per_player,
            seed,
            max(1000, args.bootstrap // 5),
        )["paired_ten_minus_three"]["delta"]
        for seed in range(args.sensitivity_seeds)
    ]
    delta = primary["paired_ten_minus_three"]["delta"]
    lower = primary["paired_ten_minus_three"]["ci95"][0]
    report = {
        "protocol": {
            "comparison": (
                "same ten-profile extraction; 3-profile arm selects "
                "normal/5k/1d/5d only"
            ),
            "architecture": "same transformer_rank.py defaults, 3 seeds each",
            "primary_questions_per_player": args.questions_per_player,
            "primary_sampling_seed": 0,
            "gate": "delta >= +0.003 and player-bootstrap 95% lower bound > 0",
        },
        "primary": primary,
        "sampling_sensitivity": {
            "seeds": args.sensitivity_seeds,
            "deltas": sensitivity,
            "mean": float(np.mean(sensitivity)),
            "min": float(np.min(sensitivity)),
            "max": float(np.max(sensitivity)),
            "positive_seeds": int((np.asarray(sensitivity) > 0).sum()),
        },
        "gate": {
            "delta": delta,
            "ci95_lower": lower,
            "passes": bool(delta >= 0.003 and lower > 0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
