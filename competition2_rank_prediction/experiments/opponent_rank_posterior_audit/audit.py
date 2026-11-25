"""Player-disjoint, ground-truthed audit of frozen v4 identity -> rank posteriors.

Input is a *known-player held-out game* control, NOT unknown official opponents.
No test labels, new encoder training, or end-to-end v47 oracle is involved.
Run via -m from repository root, or with PYTHONPATH set to repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
DEFAULT_TEMPERATURE = 0.04  # v47 identity softmax starts at this temperature
TEMPERATURES = (0.02, 0.03, 0.04, 0.06, 0.08, 0.12)
SEED = 20261120


def validate(data: dict[str, np.ndarray]) -> None:
    """Reject malformed, misaligned or ambiguous label/class mappings."""
    needed = ("cosines", "true_class", "class_rank", "player_id", "game_id")
    missing = [key for key in needed if key not in data]
    if missing:
        raise ValueError(f"missing cache arrays: {missing}")
    cos = data["cosines"]
    truth = data["true_class"]
    ranks = data["class_rank"]
    if cos.ndim != 2 or len(cos) == 0 or cos.shape[1] != len(ranks):
        raise ValueError("cosines must be nonempty [games, num_classes]")
    if not np.isfinite(cos).all():
        raise ValueError("non-finite class similarities")
    if not np.issubdtype(truth.dtype, np.integer) or len(truth) != len(cos):
        raise ValueError("true_class must be a class index for every game")
    if np.any(truth < 0) or np.any(truth >= len(ranks)):
        raise ValueError("true class outside reference gallery")
    if not np.issubdtype(ranks.dtype, np.integer) or np.any((ranks < 0) | (ranks > 9)):
        raise ValueError("reference class rank is missing or invalid")
    for key in ("player_id", "game_id"):
        if len(data[key]) != len(cos):
            raise ValueError(f"{key} not aligned to cosine rows")
    if len(set(data["game_id"].astype(str))) != len(cos):
        raise ValueError("duplicate game_id in controls")
    group_class: dict[str, int] = {}
    for player, cls in zip(data["player_id"].astype(str), truth):
        previous = group_class.setdefault(player, int(cls))
        if previous != cls:
            raise ValueError(f"player {player} has inconsistent class ground truth")
    if "true_rank" in data and not np.array_equal(data["true_rank"], ranks[truth]):
        raise ValueError("true_rank disagrees with truth class's registered rank")


def split_players(players: np.ndarray, ranks: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Two predetermined, rank-stratified player-disjoint folds."""
    unique, inverse = np.unique(players.astype(str), return_inverse=True)
    first = np.full(len(unique), -1, dtype=np.int64)
    # Index loop kept as written (the index also drives other arrays).
    # pylint: disable-next=consider-using-enumerate
    for index in range(len(unique)):
        seen = np.unique(ranks[inverse == index])
        if len(seen) != 1:
            raise ValueError(f"conflicting ranks for player {unique[index]}")
        first[index] = int(seen[0])
    rng = np.random.default_rng(seed)
    folds = np.full(len(unique), -1, dtype=np.int8)
    for rank in range(10):
        group = np.flatnonzero(first == rank)
        order = rng.permutation(group)
        folds[order] = np.arange(len(order)) % 2
    if np.any(folds < 0):
        raise ValueError("unassigned player fold")
    return folds[inverse]


def rank_posterior(cosines: np.ndarray, classes: np.ndarray, temperature: float) -> np.ndarray:
    """Replicate v47: softmax over *classes*, then sum probabilities by registered rank."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(cosines, np.float32) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    denominator = exponent.sum(axis=1, keepdims=True)
    posterior = np.column_stack(
        [exponent[:, classes == rank].sum(axis=1) for rank in range(10)]
    )
    result = posterior / denominator
    if not np.allclose(result.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError("rank posteriors do not sum to one")
    return result


def temperature_crossfit(
    cosines: np.ndarray, class_rank: np.ndarray, truth_rank: np.ndarray,
    fold: np.ndarray, grid: tuple[float, ...] = TEMPERATURES,
) -> tuple[np.ndarray, dict[str, float]]:
    """Choose T on the opposite group; no evaluation labels enter the fit."""
    if len(grid) == 0 or not np.isin(fold, (0, 1)).all() or set(fold) != {0, 1}:
        raise ValueError("need both player-disjoint folds and nonempty temperature grid")
    posteriors = {temp: rank_posterior(cosines, class_rank, temp) for temp in grid}
    output = np.full((len(truth_rank), 10), np.nan, np.float64)
    chosen: dict[str, float] = {}
    for held in (0, 1):
        train = fold != held
        select = fold == held
        losses = {
            temp: -np.log(np.clip(post[train, truth_rank[train]], 1e-12, 1)).mean()
            for temp, post in posteriors.items()
        }
        # deterministic tie-break, prefers the existing production default
        # The lambda is consumed within the iteration that defines the variable.
        # pylint: disable-next=cell-var-from-loop
        best = min(grid, key=lambda t: (losses[t], abs(t - DEFAULT_TEMPERATURE)))
        chosen[str(held)] = float(best)
        output[select] = posteriors[best][select]
    if not np.isfinite(output).all():
        raise ValueError("cross-fitting left out unscored games")
    return output, chosen


def auc(scores: np.ndarray, truth: np.ndarray) -> float | None:
    """Mann-Whitney AUC for ascending rank, preserving ties."""
    if min(int(truth.sum()), int((~truth).sum())) == 0:
        return None
    ranks = rankdata(scores, method="average")
    p = int(truth.sum())
    n = len(truth) - p
    return float((ranks[truth].sum() - p * (p + 1) / 2) / (p * n))


def boundaries(prob: np.ndarray, truth: np.ndarray) -> dict[str, dict]:
    """Per-boundary log odds AUC, with class counts for interpretation."""
    answer = {}
    for low in range(9):
        selected = (truth == low) | (truth == low + 1)
        chosen = truth[selected]
        values = np.log(np.maximum(prob[selected, low + 1], 1e-12)) - np.log(
            np.maximum(prob[selected, low], 1e-12)
        )
        answer[f"{LEVELS[low]}/{LEVELS[low + 1]}"] = {
            "auc": auc(values, chosen == low + 1),
            "counts": [int((chosen == low).sum()), int((chosen == low + 1).sum())],
        }
    return answer


def rank_utility(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rank-only diagnostic (NOT the question-level Competition 2 score)."""
    distance = np.abs(pred - truth)
    return np.where(distance == 0, 1.0, np.where(distance == 1, np.exp(-1), 0.0))


def metrics(posterior: np.ndarray, truth: np.ndarray) -> dict:
    """Rank retrieval metrics, not final pipeline metrics."""
    prediction = posterior.argmax(axis=1)
    p_true = np.clip(posterior[np.arange(len(truth)), truth], 1e-12, 1.0)
    onehot = np.eye(10)[truth]
    return {
        "rank_top1": float(np.mean(prediction == truth)),
        "rank_off_by_one_or_less": float(np.mean(np.abs(prediction - truth) <= 1)),
        "mean_true_rank_mass": float(np.mean(p_true)),
        "rank_nll": float(-np.log(p_true).mean()),
        "rank_brier": float(np.sum((posterior - onehot) ** 2, axis=1).mean()),
        "rank_only_utility_not_competition2": float(rank_utility(prediction, truth).mean()),
        "confusion_true_rows_pred_columns": np.histogram2d(
            truth, prediction, bins=(np.arange(11) - 0.5, np.arange(11) - 0.5)
        )[0].astype(int).tolist(),
    }


def player_bootstrap(diff: np.ndarray, players: np.ndarray, draws: int, seed: int) -> dict:
    """Paired player bootstrap of a per-game difference, resampling whole players."""
    if draws < 100:
        raise ValueError("bootstrap draws must be >= 100")
    groups, inverse = np.unique(players.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse)
    random = np.random.default_rng(seed)
    means = np.empty(draws, float)
    for start in range(0, draws, 256):
        end = min(start + 256, draws)
        sampled = random.integers(0, len(groups), size=(end - start, len(groups)))
        means[start:end] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "players": int(len(groups)),
    }


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def low_high_margin(
    cosines: np.ndarray, fold: np.ndarray, true_rank: np.ndarray,
    fixed: np.ndarray, soft: np.ndarray,
) -> dict:
    """Label-free thresholds fitted on the opposite fold; diagnostic only."""
    top = np.partition(cosines, kth=cosines.shape[1] - 2, axis=1)[:, -2:]
    margin = np.abs(top[:, 1] - top[:, 0])
    low = np.zeros(len(fold), bool)
    cuts = {}
    for held in (0, 1):
        cut = float(np.median(margin[fold != held]))
        low[fold == held] = margin[fold == held] <= cut
        cuts[str(held)] = cut
    answer = {"thresholds_from_opposite_fold": cuts}
    for name, mask in (("low", low), ("high", ~low)):
        hard_accuracy = (fixed.argmax(1)[mask] == true_rank[mask]).mean()
        soft_accuracy = (soft.argmax(1)[mask] == true_rank[mask]).mean()
        answer[name] = {
            "games": int(mask.sum()),
            "rank_soft_accuracy": float(soft_accuracy),
            "rank_hard_accuracy": float(hard_accuracy),
            "soft_minus_hard": float(soft_accuracy - hard_accuracy),
        }
    return answer


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def analyze(data: dict[str, np.ndarray], draws: int = 2000) -> dict:
    """Main frozen-only diagnostic. Preserves game IDs and ground truth checks."""
    validate(data)
    scores = np.asarray(data["cosines"], dtype=np.float32)
    classes = np.asarray(data["class_rank"], dtype=int)
    truth_class = np.asarray(data["true_class"], dtype=int)
    truth_rank = classes[truth_class]
    players = data["player_id"].astype(str)
    fold = split_players(players, truth_rank)
    pred_class = scores.argmax(axis=1)
    hard_rank = classes[pred_class]
    hard = np.eye(10)[hard_rank]
    fixed = rank_posterior(scores, classes, DEFAULT_TEMPERATURE)
    tuned, selection = temperature_crossfit(scores, classes, truth_rank, fold)
    hard_correct = hard_rank == truth_rank
    identity_correct = pred_class == truth_class
    soft_correct = fixed.argmax(axis=1) == truth_rank
    tuned_correct = tuned.argmax(axis=1) == truth_rank
    wrong_id = ~identity_correct
    details: dict[str, dict[str, float | None]] = {}
    for rank, name in enumerate(LEVELS):
        rows = truth_rank == rank
        if not rows.any():
            details[name] = {"games": 0, "players": 0, "rank_soft_top1": None}
            continue
        details[name] = {
            "games": int(rows.sum()),
            "players": int(len(np.unique(players[rows]))),
            "identity_top1": float(identity_correct[rows].mean()),
            "rank_hard_top1": float(hard_correct[rows].mean()),
            "rank_soft_top1": float(soft_correct[rows].mean()),
            "rank_tuned_top1": float(tuned_correct[rows].mean()),
        }
    return {
        "population": (
            "known training identities, unseen control games; NOT official unknown opponents"
        ),
        "no_official_test_labels_used": True,
        "not_v47_question_utility": True,
        "oracle_note": (
            "Correct identity reveals its registered rank by construction; "
            "rank oracle = 1.0 is not an end-to-end v47 attainable score."
        ),
        "sample": {
            "games": int(len(scores)),
            "players": int(len(np.unique(players))),
            "classes": int(len(classes)),
            "fold0_games": int((fold == 0).sum()),
            "fold1_games": int((fold == 1).sum()),
            "rank_levels": list(LEVELS),
        },
        "identity_top1": float(identity_correct.mean()),
        "identity_top1_wrong_but_rank_top1_correct": {
            "count": int((wrong_id & hard_correct).sum()),
            "fraction_of_identity_errors": (
                float(hard_correct[wrong_id].mean()) if wrong_id.any() else None
            ),
        },
        "rank_oracle_top1": 1.0,
        "hard_identity_rank": metrics(hard, truth_rank),
        "v47_soft_rank_fixed_t04": metrics(fixed, truth_rank),
        "v47_soft_rank_temp_crossfit": metrics(tuned, truth_rank),
        "temp_fitted_only_on_other_player_fold": selection,
        "paired_soft_minus_hard_rank_top1": player_bootstrap(
            soft_correct.astype(float) - hard_correct.astype(float), players, draws, SEED
        ),
        "paired_tuned_minus_fixed_rank_top1": player_bootstrap(
            tuned_correct.astype(float) - soft_correct.astype(float), players, draws, SEED + 1
        ),
        "paired_tuned_minus_fixed_rank_nll_improvement": player_bootstrap(
            np.log(np.clip(tuned[np.arange(len(scores)), truth_rank], 1e-12, 1.0))
            - np.log(np.clip(fixed[np.arange(len(scores)), truth_rank], 1e-12, 1.0)),
            players, draws, SEED + 2,
        ),
        "boundary_auc_soft_t04": boundaries(fixed, truth_rank),
        "boundary_auc_tuned_crossfit": boundaries(tuned, truth_rank),
        "opponent_identity_margin_buckets": low_high_margin(scores, fold, truth_rank, hard, fixed),
        "per_rank": details,
    }


def main() -> None:
    """CLI for the frozen known-player controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if not args.scores.exists():
        parser.error(f"scores not found: {args.scores}")
    with np.load(args.scores, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}
    report = analyze(data, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "score_file": str(args.scores),
        "identity_top1": report["identity_top1"],
        "hard_rank_top1": report["hard_identity_rank"]["rank_top1"],
        "soft_rank_top1": report["v47_soft_rank_fixed_t04"]["rank_top1"],
        "tuned_rank_top1": report["v47_soft_rank_temp_crossfit"]["rank_top1"],
        "rank_1d2d_auc": report["boundary_auc_soft_t04"]["1D/2D"],
        "report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
