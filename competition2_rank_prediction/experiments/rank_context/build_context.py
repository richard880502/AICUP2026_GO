"""Build exact handicap/color context aligned to h3_sims_v47.pt.

The v47 cache collapses all HD>=2 games into kind=3 and then discards them.
This script recovers the exact SGF handicap and asked-player color from the
holdout3 question CSVs, aligned to the existing h3_opp.csv game order.

No labels are used to construct the context features.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

import numpy as np
import torch

FEATURE_NAMES = (
    "question_size",
    "own_black_fraction",
    "hd1_fraction",
    "hd2plus_fraction",
    "hd2plus_receive_fraction",
    "hd2plus_give_fraction",
    "mean_handicap",
    "signed_mean_handicap",
    "max_handicap",
    "hd2_receive_fraction",
    "hd3_receive_fraction",
    "hd4plus_receive_fraction",
    "hd2_give_fraction",
    "hd3_give_fraction",
    "hd4plus_give_fraction",
)


RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def handicap_of(sgf: str) -> int:
    """Returns SGF HD, defaulting to zero when the property is absent."""
    match = re.search(r"HD\[(\d+)\]", sgf or "")
    return int(match.group(1)) if match else 0


def handicap_by_game(data_root: pathlib.Path, map_path: pathlib.Path) -> dict[str, int]:
    """Maps holdout3 game id to its exact SGF handicap.

    The SGFs inside the holdout3 question CSVs are stripped to `SZ` plus the
    move sequence, so they carry no `HD` property and cannot be the source.
    v47 took the handicap from the authorized training CSVs instead, keyed by
    the global row index in `h3_map.npz`; this reproduces that exact path (see
    `experiments/opponent_identity/h3_dump.py`).
    """
    with np.load(map_path) as saved:
        # pylint: disable-next=no-member  # NpzFile arrays; pylint infers np.array.
        row_of_game = dict(zip(saved["game_key"].tolist(), saved["row"].tolist()))
    wanted = set(row_of_game.values())
    handicap_of_row: dict[int, int] = {}
    for index, rank in enumerate(RANKS):
        path = data_root / f"train_{rank}.csv"
        with path.open(encoding="utf-8", newline="") as source:
            for position, row in enumerate(csv.DictReader(source)):
                key = index * 100000 + position
                if key in wanted:
                    handicap_of_row[key] = handicap_of(row["sgf_content"])
    missing = wanted - set(handicap_of_row)
    if missing:
        raise ValueError(f"{len(missing)} mapped rows absent from the training CSVs")
    return {game: handicap_of_row[row] for game, row in row_of_game.items()}


def load_test_games(root: pathlib.Path) -> dict[tuple[str, int], dict]:
    """Maps (question id, 1-based game index) to SGF and own color."""
    games: dict[tuple[str, int], dict] = {}
    for split in ("public", "private"):
        path = root / "tests" / f"rank_prediction_test_{split}.csv"
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                qid = row["question_id"]
                for index in range(1, int(row["num_games"]) + 1):
                    games[(qid, index)] = {
                        "sgf": row[f"sgf_{index}"],
                        "own_color": row[f"color_{index}"],
                    }
    return games


def game_index(game_id: str, qid: str) -> int:
    """Extracts the 1-based game index from q#i or q:i identifiers."""
    for separator in ("#", ":"):
        prefix = qid + separator
        if game_id.startswith(prefix):
            return int(game_id[len(prefix):])
    tail = re.search(r"(\d+)$", game_id)
    if tail:
        return int(tail.group(1))
    raise ValueError(f"cannot infer game index from {game_id!r} for {qid!r}")


def question_features(
    handicap: np.ndarray,
    own_black: np.ndarray,
    game_question: np.ndarray,
    questions: int,
) -> np.ndarray:
    """Aggregates structural rank-context features per question."""
    result = np.zeros((questions, len(FEATURE_NAMES)), np.float32)
    for question in range(questions):
        rows = np.flatnonzero(game_question == question)
        if not len(rows):
            continue
        hd = handicap[rows].astype(np.float32)
        black = own_black[rows]
        hd2 = hd >= 2
        receive = hd2 & black
        give = hd2 & ~black
        signed = np.where(black, -hd, hd)

        def fraction(mask: np.ndarray) -> float:
            return float(mask.mean())

        result[question] = np.asarray(
            [
                len(rows) / 20.0,
                fraction(black),
                fraction(hd == 1),
                fraction(hd2),
                fraction(receive),
                fraction(give),
                float(hd.mean() / 9.0),
                float(signed.mean() / 9.0),
                float(hd.max() / 9.0),
                fraction(receive & (hd == 2)),
                fraction(receive & (hd == 3)),
                fraction(receive & (hd >= 4)),
                fraction(give & (hd == 2)),
                fraction(give & (hd == 3)),
                fraction(give & (hd >= 4)),
            ],
            dtype=np.float32,
        )
    return result


def main() -> None:  # pylint: disable=too-many-statements
    """Writes exact per-game context and question-level structural features."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt"
        ),
    )
    parser.add_argument(
        "--h3-opp",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/opp_probe/h3_opp.csv"
        ),
    )
    parser.add_argument(
        "--holdout-root",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/features/holdout3"
        ),
    )
    parser.add_argument(
        "--h3-map",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/rank_probe/h3_map.npz"
        ),
        help="game_key -> training row index, as used by h3_dump.py",
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("data/training"),
        help="directory holding train_<rank>.csv with HD in sgf_content",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "rank_context/holdout3_context.npz"
        ),
    )
    args = parser.parse_args()

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    game_question = np.asarray(state["game_question"], dtype=np.int64)
    order = np.asarray(state["order"]).astype(str)
    tests = load_test_games(args.holdout_root)
    handicap_of_game = handicap_by_game(args.data_root, args.h3_map)

    with args.h3_opp.open(encoding="utf-8", newline="") as source:
        opponent_rows = list(csv.DictReader(source))
    if len(opponent_rows) != len(game_question):
        raise ValueError(
            f"h3_opp rows {len(opponent_rows)} != state games "
            f"{len(game_question)}"
        )

    handicap = np.zeros(len(opponent_rows), np.int8)
    own_black = np.zeros(len(opponent_rows), np.bool_)
    game_id = []
    for row_index, row in enumerate(opponent_rows):
        question_index = int(game_question[row_index])
        qid = str(order[question_index])
        if row["player_id"] != qid:
            raise ValueError(
                f"question mismatch at game {row_index}: "
                f"{row['player_id']!r} != {qid!r}"
            )
        index = game_index(row["game_id"], qid)
        info = tests[(qid, index)]
        own_color = info["own_color"]
        opponent_color = row["color"]
        expected_opponent = "W" if own_color == "B" else "B"
        if opponent_color != expected_opponent:
            raise ValueError(
                f"color mismatch for {row['game_id']}: "
                f"opponent={opponent_color}, own={own_color}"
            )
        handicap[row_index] = handicap_of_game[row["game_id"]]
        own_black[row_index] = own_color == "B"
        game_id.append(row["game_id"])

    # Reconstructed context must reproduce the coarse v47 kind exactly:
    # 0 even, 1 HD1 opponent White (= own Black),
    # 2 HD1 opponent Black (= own White), 3 HD>=2.
    derived_kind = np.where(
        handicap == 0,
        0,
        np.where(
            handicap == 1,
            np.where(own_black, 1, 2),
            3,
        ),
    ).astype(np.int64)
    cached_kind = np.asarray(state["kind"], dtype=np.int64)
    if not np.array_equal(derived_kind, cached_kind):
        mismatch = np.flatnonzero(derived_kind != cached_kind)
        raise ValueError(
            f"reconstructed handicap kind disagrees with v47 on "
            f"{len(mismatch)} games; first rows {mismatch[:10].tolist()}"
        )

    features = question_features(
        handicap, own_black, game_question, len(order)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        game_id=np.asarray(game_id),
        question_id=order,
        game_question=game_question,
        handicap=handicap,
        own_black=own_black,
        own_receives_handicap=(handicap >= 2) & own_black,
        question_features=features,
        feature_names=np.asarray(FEATURE_NAMES),
    )

    hd2 = handicap >= 2
    questions_with_hd2 = np.unique(game_question[hd2])
    all_hd2 = 0
    for question in range(len(order)):
        rows = game_question == question
        if rows.any() and np.all(handicap[rows] >= 2):
            all_hd2 += 1
    report = {
        "games": len(handicap),
        "questions": len(order),
        "handicap_histogram": {
            str(value): int((handicap == value).sum())
            for value in np.unique(handicap)
        },
        "hd2plus_games": int(hd2.sum()),
        "hd2plus_fraction": float(hd2.mean()),
        "questions_with_hd2plus": int(len(questions_with_hd2)),
        "questions_all_hd2plus": int(all_hd2),
        "feature_names": list(FEATURE_NAMES),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
