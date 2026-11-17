"""Flattens the internal holdout3 questions into an own-side rich-token CSV."""

from __future__ import annotations

import argparse
import csv
import pathlib


FIELDS = (
    "game_id",
    "player_id",
    "sgf_content",
    "color",
    "note",
    "question_id",
    "rank",
    "fold",
)


def main() -> None:
    """Writes one row per holdout3 game in question order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-root",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/features/holdout3"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "strength_score_v2/holdout3_own.csv"
        ),
    )
    args = parser.parse_args()

    with (args.holdout_root / "question_folds.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        metadata = {
            row["question_id"]: row for row in csv.DictReader(source)
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    questions = 0
    games = 0
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for split in ("public", "private"):
            path = (
                args.holdout_root
                / "tests"
                / f"rank_prediction_test_{split}.csv"
            )
            with path.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    qid = row["question_id"]
                    info = metadata[qid]
                    questions += 1
                    for index in range(1, int(row["num_games"]) + 1):
                        writer.writerow(
                            {
                                "game_id": f"{qid}:{index}",
                                "player_id": info["player"],
                                "sgf_content": row[f"sgf_{index}"],
                                "color": row[f"color_{index}"],
                                "note": qid,
                                "question_id": qid,
                                "rank": info["rank"],
                                "fold": info["fold"],
                            }
                        )
                        games += 1
    print(
        f"wrote {questions} questions / {games} games -> {args.output}"
    )


if __name__ == "__main__":
    main()
