"""Extract an unlabelled question-level identity-concentration proxy from v47.

The proxy is the mean top-1/top-2 cosine gap across HD0/HD1 opponent games.
It is NOT a calibrated probability that the opponent is a known player.
Run in the existing competition2 research environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def extract(path: Path, batch: int = 256) -> dict[str, np.ndarray]:
    """Read v47 scores on CPU and preserve its original saved question order."""
    if batch < 1:
        raise ValueError("batch must be positive")
    data = torch.load(path, map_location="cpu", weights_only=False)
    scores = data["sims"]
    question_id = np.asarray(data["order"], dtype=str)
    if len(np.unique(question_id)) != len(question_id):
        raise ValueError("duplicate question ids in v47 order")
    question_of_game = np.asarray(data["game_question"], dtype=np.int64)
    kind = np.asarray(data["kind"], dtype=np.int64)
    holdout_classes = np.asarray(data["holdout_classes"], dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("expected [games, classes>=2] similarity tensor")
    if len(kind) != len(scores) or len(question_of_game) != len(scores):
        raise ValueError("inconsistent number of games")
    if (question_of_game < 0).any() or (question_of_game >= len(question_id)).any():
        raise ValueError("out-of-range game_question")
    if (holdout_classes < 0).any() or (holdout_classes >= scores.shape[1]).any():
        raise ValueError("out-of-range excluded class")
    if not np.isin(kind, [0, 1, 2, 3]).all():
        raise ValueError("unexpected game kind")

    total = np.zeros(len(question_id), dtype=np.float64)
    counts = np.zeros(len(question_id), dtype=np.int64)
    all_games = np.bincount(question_of_game, minlength=len(question_id))
    incompatible = np.bincount(
        question_of_game, weights=(kind >= 3).astype(np.int64),
        minlength=len(question_id),
    ).astype(bool)
    for start in range(0, len(scores), batch):
        stop = min(start + batch, len(scores))
        mask = kind[start:stop] < 3
        if not mask.any():
            continue
        batch_scores = scores[start:stop].float().numpy()[mask].copy()
        batch_scores[:, holdout_classes] = -5.0  # match p2_evaluate.load_v47
        top_two = np.partition(batch_scores, -2, axis=1)[:, -2:]
        gap = np.abs(top_two[:, 1] - top_two[:, 0])
        if not np.isfinite(gap).all():
            raise ValueError("non-finite class similarity gap")
        rows = question_of_game[start:stop][mask]
        total += np.bincount(rows, weights=gap, minlength=len(question_id))
        counts += np.bincount(rows, minlength=len(question_id))
    result = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0)
    return {
        "question_id": question_id,
        "identity_margin": result,
        "usable_games": counts,
        "all_games": all_games,
        "has_hd2plus": incompatible,
    }


def main() -> None:
    """Extract a compact CPU cache (no model fitting and no test labels)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    features = extract(args.sims, args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **features)
    print(f"saved {len(features['question_id'])} question confidence proxies to {args.output}")
    print(f"official-compatible HD0/HD1: {(~features['has_hd2plus']).sum()} questions")


if __name__ == "__main__":
    main()
