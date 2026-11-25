"""Checks training move-sequence overlap with official rank questions.

Example: python -m competition2_rank_prediction.experiments.training_overlap.audit
Loads no training rank labels and never reads other-task candidate libraries.
"""

import csv
import hashlib
import json
import re

import numpy as np

from competition2_rank_prediction.experiments.rank_experts import common

MOVE = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")


def move_hash(sgf: str) -> int:
    """Reproduces the existing authorized index's64bit move-only hash."""
    sequence = "".join(color + point for color, point in MOVE.findall(sgf))
    return int.from_bytes(
        hashlib.blake2b(sequence.encode(), digest_size=8).digest(), "little"
    )


def check_question(row: dict, index: dict[str, np.ndarray]) -> dict:
    """Counts same/opposite-color hits without retrieving rank labels."""
    same = 0
    opposite = 0
    owners = set()
    count = int(row["num_games"])
    if not 5 <= count <= 20:
        raise ValueError(f"Invalid query size: {count!r}")
    for game in range(1, count + 1):
        sgf = row[f"sgf_{game}"]
        color = row[f"color_{game}"].strip().upper()
        if not sgf or color not in ("B", "W"):
            raise ValueError("Missing official SGF or invalid color")
        value = np.uint64(move_hash(sgf))
        left = int(np.searchsorted(index["hash"], value, side="left"))
        right = int(np.searchsorted(index["hash"], value, side="right"))
        matches = index["colour"][left:right] == color
        same += int(matches.any())
        opposite += int((~matches).any())
        owners.update(index["player"][left:right][matches].tolist())
    return {
        "question_id": row["question_id"],
        "games": count,
        "same_actor_move_matches": same,
        "opposite_actor_move_matches": opposite,
        "distinct_anonymous_training_owners": len(owners),
    }


def main() -> None:
    """Writes aggregate diagnostics only; no answers, submissions or new training."""
    args = common.parse_arguments(__doc__ or "", "training_overlap_20261109")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite overlap audit: {args.output}"
        )
    with np.load(
        args.research / "features/preflight/train_move_index.npz"
    ) as data:
        order = np.argsort(data["hash"])
        index = {
            name: np.asarray(data[name])[order]
            for name in ("hash", "colour", "player")
        }
    if len(index["hash"]) != 1000000:
        raise ValueError("Expected complete one-million-game training index")
    results = []
    for split in ("public", "private"):
        path = (
            args.research
            / "data/aicup2026_rank_prediction_tests_20261030/tests"
            / f"rank_prediction_test_{split}.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                results.append({"split": split, **check_question(row, index)})
    if (
        len(results) != 800
        or len({row["question_id"] for row in results}) != 800
    ):
        raise ValueError("Expected800unique official questions")
    report = {
        "scope": "read-only authorized-training/rank-test diagnostics",
        "training_games": len(index["hash"]),
        "questions": len(results),
        "query_games": sum(row["games"] for row in results),
        "same_actor_game_matches": sum(
            row["same_actor_move_matches"] for row in results
        ),
        "opposite_actor_game_matches": sum(
            row["opposite_actor_move_matches"] for row in results
        ),
        "questions_with_same_actor_matches": sum(
            row["same_actor_move_matches"] > 0 for row in results
        ),
        "questions_with_conflicting_training_owners": sum(
            row["distinct_anonymous_training_owners"] > 1 for row in results
        ),
        "training_rank_labels_read": False,
        "other_task_candidate_data_read": False,
        "prediction_changes": False,
        "limitations": (
            "64bit move-only hash, ignores root properties; "
            "zero hits do not prove player-disjointness"
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
