"""Train one player-disjoint fold of the HumanSL continuous Strength Score."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from competition2_rank_prediction import rank_lib
from competition2_rank_prediction.experiments.strength_score_v2.model import (
    DISTANCE_WEIGHTINGS,
    StrengthScoreNetwork,
    relative_strength_loss,
)

LEVELS = rank_lib.LEVELS


def load_train_cache(directory: pathlib.Path) -> dict:
    tokens = np.load(directory / "tokens.npy", mmap_mode="r")
    masks = np.load(directory / "masks.npy", mmap_mode="r")
    with np.load(directory / "metadata.npz") as meta:
        game_player = np.asarray(meta["game_player_index"], dtype=np.int64)
        player_id = np.asarray(meta["player_id"]).astype(str)
        player_rank = np.asarray(meta["player_rank"], dtype=np.int64)
    groups = [
        np.flatnonzero(game_player == index)
        for index in range(len(player_id))
    ]
    return {
        "tokens": tokens,
        "masks": masks,
        "player_id": player_id,
        "player_rank": player_rank,
        "groups": groups,
    }


def load_eval_cache(directory: pathlib.Path) -> dict:
    with np.load(directory / "metadata.npz") as meta:
        result = {
            "question_id": np.asarray(meta["question_id"]).astype(str),
            "player_id": np.asarray(meta["player_id"]).astype(str),
            "rank": np.asarray(meta["rank"], dtype=np.int64),
            "fold": np.asarray(meta["fold"], dtype=np.int64),
            "valid_game": np.asarray(meta["valid_game"], dtype=bool),
        }
    result["tokens"] = np.load(directory / "tokens.npy", mmap_mode="r")
    result["masks"] = np.load(directory / "masks.npy", mmap_mode="r")
    return result


def split_players(
    cache: dict,
    excluded: set[str],
    inner_players_per_rank: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(len(LEVELS)):
        candidates = np.asarray(
            [
                index
                for index, player in enumerate(cache["player_id"])
                if cache["player_rank"][index] == label
                and player not in excluded
            ],
            dtype=np.int64,
        )
        if len(candidates) <= inner_players_per_rank:
            raise ValueError(
                f"{LEVELS[label]} only has {len(candidates)} usable players"
            )
        chosen = rng.choice(
            candidates, size=inner_players_per_rank, replace=False
        )
        train.append(candidates[~np.isin(candidates, chosen)])
        val.append(np.sort(chosen))
    return train, val


def sample_batch(
    cache: dict,
    by_rank: list[np.ndarray],
    players_per_rank: int,
    games_per_player: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    player_rows = []
    labels = []
    for label, candidates in enumerate(by_rank):
        chosen = rng.choice(
            candidates,
            size=players_per_rank,
            replace=len(candidates) < players_per_rank,
        )
        for player in chosen:
            rows = cache["groups"][int(player)]
            selected = rng.choice(
                rows,
                size=games_per_player,
                replace=len(rows) < games_per_player,
            )
            player_rows.append(selected)
            labels.append(label)
    rows = np.asarray(player_rows, dtype=np.int64)
    flat = rows.ravel()
    tokens = np.asarray(cache["tokens"][flat]).reshape(
        len(rows), games_per_player, *cache["tokens"].shape[1:]
    )
    masks = np.asarray(cache["masks"][flat]).reshape(
        len(rows), games_per_player, *cache["masks"].shape[1:]
    )
    return tokens, masks, np.asarray(labels, dtype=np.int64)


def fixed_rows(
    cache: dict,
    player_indices: np.ndarray,
    games_per_player: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            rng.choice(
                cache["groups"][int(player)],
                size=games_per_player,
                replace=len(cache["groups"][int(player)]) < games_per_player,
            )
            for player in player_indices
        ],
        dtype=np.int64,
    )


def score_fixed(
    model: StrengthScoreNetwork,
    cache: dict,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    games = rows.shape[1]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            flat = selected.ravel()
            tokens = np.asarray(cache["tokens"][flat]).reshape(
                len(selected), games, *cache["tokens"].shape[1:]
            )
            masks = np.asarray(cache["masks"][flat]).reshape(
                len(selected), games, *cache["masks"].shape[1:]
            )
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                score, _, _ = model(
                    torch.from_numpy(tokens).to(device),
                    torch.from_numpy(masks).to(device),
                )
            output.append(score.float().cpu().numpy())
    return np.concatenate(output)


def adjacent_auc(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {}
    for lower in range(len(LEVELS) - 1):
        mask = (labels == lower) | (labels == lower + 1)
        result[f"{LEVELS[lower]}/{LEVELS[lower + 1]}"] = float(
            roc_auc_score(
                (labels[mask] == lower + 1).astype(np.int64),
                scores[mask],
            )
        )
    return result


def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(
        [
            scores[labels == label].mean()
            for label in range(len(LEVELS))
        ],
        dtype=np.float64,
    )
    means = IsotonicRegression(increasing=True).fit_transform(
        np.arange(len(LEVELS), dtype=np.float64),
        raw,
    )
    residual = scores - means[labels]
    sigma = float(max(np.std(residual), np.std(means) * 0.08, 1e-3))
    return np.asarray(means), sigma


def calibrate(
    scores: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> np.ndarray:
    z = (scores[:, None] - means[None, :]) / sigma
    logits = -0.5 * z * z
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    return prob / prob.sum(axis=1, keepdims=True)


def validation_report(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, np.ndarray, float]:
    auc = adjacent_auc(scores, labels)
    means, sigma = fit_calibrator(scores, labels)
    probabilities = calibrate(scores, means, sigma)
    prediction = rank_lib.decode(probabilities)
    report = {
        "spearman": float(spearmanr(scores, labels).statistic),
        "adjacent_auc": auc,
        "mean_adjacent_auc": float(np.mean(list(auc.values()))),
        "score": rank_lib.competition_score(prediction, labels),
        "exact": float((prediction == labels).mean()),
        "rank_mean_strength": {
            LEVELS[label]: float(scores[labels == label].mean())
            for label in range(len(LEVELS))
        },
        "calibration_sigma": sigma,
    }
    return report, means, sigma


def score_questions(
    model: StrengthScoreNetwork,
    cache: dict,
    target_fold: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.flatnonzero(
        cache["valid_game"] & (cache["fold"] == target_fold)
    )
    grouped: dict[str, list[int]] = {}
    for row in valid:
        grouped.setdefault(str(cache["question_id"][row]), []).append(int(row))
    qids = np.asarray(sorted(grouped))
    truth = np.empty(len(qids), dtype=np.int64)
    counts = np.empty(len(qids), dtype=np.int64)
    scores = np.empty(len(qids), dtype=np.float32)
    by_count: dict[int, list[int]] = {}
    for index, qid in enumerate(qids):
        rows = grouped[str(qid)]
        truth[index] = int(np.unique(cache["rank"][rows])[0])
        counts[index] = len(rows)
        by_count.setdefault(len(rows), []).append(index)

    model.eval()
    with torch.no_grad():
        for count, indices in sorted(by_count.items()):
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                row_matrix = np.asarray(
                    [
                        grouped[str(qids[index])]
                        for index in batch_indices
                    ],
                    dtype=np.int64,
                )
                flat = row_matrix.ravel()
                tokens = np.asarray(cache["tokens"][flat]).reshape(
                    len(batch_indices),
                    count,
                    *cache["tokens"].shape[1:],
                )
                masks = np.asarray(cache["masks"][flat]).reshape(
                    len(batch_indices),
                    count,
                    *cache["masks"].shape[1:],
                )
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    batch_scores, _, _ = model(
                        torch.from_numpy(tokens).to(device),
                        torch.from_numpy(masks).to(device),
                    )
                scores[np.asarray(batch_indices)] = (
                    batch_scores.float().cpu().numpy()
                )
    return qids, truth, counts, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "strength_score_v2/train_cache"
        ),
    )
    parser.add_argument(
        "--eval-cache",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "strength_score_v2/holdout3_cache"
        ),
    )
    parser.add_argument("--exclude-players", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=160)
    parser.add_argument("--players-per-rank", type=int, default=8)
    parser.add_argument("--games-per-player", type=int, default=4)
    parser.add_argument("--validation-games", type=int, default=8)
    parser.add_argument("--inner-players-per-rank", type=int, default=24)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--batch-eval", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--scale-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument(
        "--distance-weighting",
        choices=DISTANCE_WEIGHTINGS,
        default="forward",
        help=(
            "Pair weighting in the ranking loss. 'forward' (default) favours "
            "distant ranks; the others test adjacent-boundary supervision."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.fold)
    rng = np.random.default_rng(args.seed + args.fold)
    cache = load_train_cache(args.cache)
    eval_cache = load_eval_cache(args.eval_cache)
    excluded = set(
        args.exclude_players.read_text(encoding="utf-8").split()
    )
    train_by_rank, val_by_rank = split_players(
        cache,
        excluded,
        args.inner_players_per_rank,
        args.seed + args.fold,
    )
    val_players = np.concatenate(val_by_rank)
    val_labels = cache["player_rank"][val_players]
    val_rows = fixed_rows(
        cache,
        val_players,
        args.validation_games,
        args.seed + 100 + args.fold,
    )

    model = StrengthScoreNetwork(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * args.steps_per_epoch,
        pct_start=0.10,
    )
    best_metric = -float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            tokens, masks, labels = sample_batch(
                cache,
                train_by_rank,
                args.players_per_rank,
                args.games_per_player,
                rng,
            )
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                player_score, game_score, _ = model(
                    torch.from_numpy(tokens).to(device),
                    torch.from_numpy(masks).to(device),
                )
                terms = relative_strength_loss(
                    player_score,
                    game_score,
                    torch.from_numpy(labels).to(device),
                    args.temperature,
                    args.consistency_weight,
                    args.scale_weight,
                    args.distance_weighting,
                )
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(terms.total.detach().cpu()))

        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            val_scores = score_fixed(
                model, cache, val_rows, device, args.batch_eval
            )
            report, _, _ = validation_report(val_scores, val_labels)
            row.update(
                {
                    "validation_spearman": report["spearman"],
                    "validation_mean_adjacent_auc": report[
                        "mean_adjacent_auc"
                    ],
                    "validation_score": report["score"],
                }
            )
            if report["mean_adjacent_auc"] > best_metric:
                best_metric = report["mean_adjacent_auc"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(row)
        print(json.dumps(row), flush=True)

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    val_scores = score_fixed(
        model, cache, val_rows, device, args.batch_eval
    )
    val_report, means, sigma = validation_report(val_scores, val_labels)
    qids, truth, counts, strength = score_questions(
        model,
        eval_cache,
        args.fold,
        device,
        args.batch_eval,
    )
    probabilities = calibrate(strength, means, sigma)
    prediction = rank_lib.decode(probabilities)
    holdout_report = {
        "fold": args.fold,
        "questions": int(len(qids)),
        "score": rank_lib.competition_score(prediction, truth),
        "exact": float((prediction == truth).mean()),
        "spearman": float(spearmanr(strength, truth).statistic),
        "adjacent_auc": adjacent_auc(strength, truth),
    }

    output = args.output_root / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best_state,
            "validation": val_report,
            "calibrator_means": means,
            "calibrator_sigma": sigma,
            "args": vars(args),
        },
        output / "checkpoint.pt",
    )
    np.savez_compressed(
        output / "predictions.npz",
        question_id=qids,
        truth=truth,
        fold=np.full(len(qids), args.fold, dtype=np.int64),
        n_games=counts,
        strength=strength,
        probabilities=probabilities.astype(np.float32),
    )
    report = {
        "excluded_players": len(excluded),
        "inner_validation": val_report,
        "holdout3": holdout_report,
        "history": history,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(holdout_report, indent=2))


if __name__ == "__main__":
    main()
