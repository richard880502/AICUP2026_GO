"""Score held-out games of *known* gallery players with frozen style_v4.

No re-training, official test question labels, or player-identity inference on
unknown opponents. Expects present.csv made by link_inputs.write_present(),
and trunk_style.extract --samples 160 --local-only on that exact CSV order.

Run from repository root with CUDA_VISIBLE_DEVICES set to the desired GPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.trunk_style import apply, train
from competition2_rank_prediction.experiments.opponent_identity import link_inputs
from competition2_rank_prediction.experiments.opponent_rank_posterior_audit.audit import validate

csv.field_size_limit(10**9)
DEFAULT_PROBE = Path("/tmp2/richard/claude_scratch/opp_probe")


def gallery_ranks(num_classes: int) -> np.ndarray:
    """Class to true registered rank; the same mapping as link_test.gallery()."""
    owners, rows, selected = link_inputs.v4_gallery()
    if np.any(owners < 0) or np.any(owners >= num_classes):
        raise ValueError("gallery class id outside checkpoint class_weights")
    rank = selected[rows] // 100000
    if np.any((rank < 0) | (rank >= 10)):
        raise ValueError("gallery class has invalid rank")
    class_rank = np.full(num_classes, -1, dtype=np.int64)
    for owner, label in zip(owners, rank):
        old = class_rank[int(owner)]
        if old >= 0 and old != label:
            raise ValueError(f"class {owner} spans conflicting registered ranks")
        class_rank[int(owner)] = int(label)
    if np.any(class_rank < 0):
        raise ValueError(f"{int((class_rank < 0).sum())} checkpoint classes have no rank")
    return class_rank


def load_csv(path: Path) -> list[dict[str, str]]:
    """Require source metadata preserving the original class and row index."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not all(row.get(k) for k in ("game_id", "player_id", "note", "color"))
                       for row in rows):
        raise ValueError("present.csv is missing rows or required metadata")
    return rows


def verify_shards(directory: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Reject incomplete or out-of-order caches (same-order encoding is essential)."""
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no trunk features at {directory}")
    tokens = train.load_shards(directory, "tokens", 0)
    masks = train.load_shards(directory, "masks", 0)
    identifiers = train.load_shards(directory, "record_ids", 0).astype(str)
    expected = np.asarray([row["game_id"] for row in rows], dtype=str)
    if not np.array_equal(identifiers, expected):
        raise ValueError("trunk shard record_ids are incomplete or out of CSV order")
    if tokens.shape[:2] != masks.shape or tokens.shape[1:] != (160, 384):
        raise ValueError(f"expected local-only 160x384 tokens; got {tokens.shape}")
    return tokens, masks


def evaluate(args: argparse.Namespace) -> None:
    """Load frozen checkpoint, compute all 6,807 class cosines, save validated cache."""
    rows = load_csv(args.present_csv)
    tokens, masks = verify_shards(args.token_dir, rows)
    valid_game = masks.any(axis=1)
    skipped = int((~valid_game).sum())
    if not valid_game.any():
        raise ValueError("all control games have empty decision masks")
    # StyleEncoder's attention softmax is undefined for an all-padded game.
    # Exclude it after full CSV/shard ID alignment, before embedding inference.
    rows = [row for row, keep in zip(rows, valid_game) if keep]
    tokens, masks = tokens[valid_game], masks[valid_game]
    # IMPORTANT: do not import rich_full; it monkeypatches use_all_moves() at import time.
    model, scaling, _weight, _checkpoint = apply.load_model("style_v4")
    classes = F.normalize(model.class_weights.detach().float(), dim=-1)
    class_rank = gallery_ranks(len(classes))
    true_class = np.asarray([int(row["note"].split(":")[0]) for row in rows], dtype=int)
    global_row = np.asarray([int(row["note"].split(":")[1]) for row in rows], dtype=int)
    if np.any(true_class < 0) or np.any(true_class >= len(classes)):
        raise ValueError("present.csv includes a class outside the frozen gallery")
    if not np.array_equal(class_rank[true_class], global_row // 100000):
        raise ValueError("ground truth class/rank disagrees with source training row")
    data = {
        "trunk": tokens,
        "engine": np.zeros((len(rows), 160, 42), np.float16),
        "masks": masks,
        "colors": np.asarray([row["color"] == "W" for row in rows], np.int64),
        "has_engine": np.zeros(len(rows), dtype=bool),
    }
    vectors = torch.from_numpy(train.encode(model, data, scaling, np.arange(len(rows)))).to(
        classes.device
    )
    cosines = (F.normalize(vectors.float(), dim=-1) @ classes.T).cpu().numpy()
    result = {
        "cosines": cosines.astype(np.float32),
        "class_rank": class_rank,
        "true_class": true_class.astype(np.int32),
        "true_rank": class_rank[true_class].astype(np.int8),
        "player_id": np.asarray([row["player_id"] for row in rows], dtype=str),
        "game_id": np.asarray([row["game_id"] for row in rows], dtype=str),
    }
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"saved {len(rows)} frozen known-player scores to {args.output}")
    print(f"excluded empty-decision games: {skipped}")


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--present-csv", type=Path, default=DEFAULT_PROBE / "present.csv")
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_PROBE / "present")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
