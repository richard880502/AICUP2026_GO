"""Game lists for the opponent-identity test.

Reproduces Competition 2 rank submission v47, step 4 of 8. Writes, in the
opponent-probe directory:

1. present.csv   : unused training games of labelled gallery players (own side)
2. test_opp.csv  : every official rank-test game, opponent side
3. test_own.csv  : every official rank-test game, own side

Their trunk tokens (`trunk_style.extract --samples 160 --local-only`) go to
`opp_probe/{present,test_opp,test_own}` for link_test.py; extract_rich.py
writes the 960-D tokens of test_opp.csv for h3_v7.py.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition2_rank_prediction.experiments.opponent_identity.link_inputs
"""
import csv
import random

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.trunk_style import train

csv.field_size_limit(10**9)
# The research machine's location for the opponent-identity intermediate files.
OUT = "/tmp2/richard/claude_scratch/opp_probe"
RANKS = dataset.cache.RANKS
FIELDS = ["game_id", "player_id", "sgf_content", "color", "note"]
ROOT = "data/aicup2026_rank_prediction_tests_20261030/tests"


def v4_gallery():
    """Class (owner) and corpus row of v4's training embeddings, and the corpus global indices.

    Shared with link_test.py and h3_dump.py.
    """
    everything = np.load(train.OPEN / "engine_inputs/global_indices.npy")
    with np.load(train.ROOT / "rerank/train_embeddings.npz") as saved:
        owners, rows = saved["owners"], saved["rows"]
    selected = np.concatenate(
        (everything[:120561], np.load(train.ROOT / "inputs/corpus2_global_indices.npy"))
    )
    return owners, rows, selected


def write_present() -> None:  # pylint: disable=too-many-locals
    """present.csv: 4,096 unused training games of labelled gallery players."""
    owners, rows, selected = v4_gallery()
    used = set(selected.tolist())
    gallery_global = selected[rows]
    # owners is an NpzFile array; pylint infers it as the np.array function.
    owner_of_global = dict(
        zip(gallery_global.tolist(), owners.tolist())  # pylint: disable=no-member
    )
    random.seed(11)
    owner_of_player, candidates = {}, []
    tables = {}
    for index, rank in enumerate(RANKS):
        with open(f"data/training/train_{rank}.csv", encoding="utf-8", newline="") as source:
            tables[index] = list(csv.DictReader(source))
        for position, row in enumerate(tables[index]):
            key = index * 100000 + position
            if key in owner_of_global:
                owner_of_player[row["player_id"]] = owner_of_global[key]
    for index, table in tables.items():
        for position, row in enumerate(table):
            key = index * 100000 + position
            if key not in used and row["player_id"] in owner_of_player:
                candidates.append((key, row))
    print(
        "gallery players", len(owner_of_player),
        "unused games of gallery players", len(candidates),
    )
    with open(f"{OUT}/present.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for key, row in sorted(random.sample(candidates, 4096)):
            writer.writerow({
                "game_id": row["game_id"], "player_id": row["player_id"],
                "sgf_content": row["sgf_content"], "color": row["color"],
                "note": f"{owner_of_player[row['player_id']]}:{key}",
            })


def write_official() -> None:
    """test_opp.csv and test_own.csv: both sides of every official rank-test game."""
    writers = {}
    for side in ("opp", "own"):
        handle = open(  # pylint: disable=consider-using-with
            f"{OUT}/test_{side}.csv", "w", encoding="utf-8", newline=""
        )
        writers[side] = csv.DictWriter(handle, fieldnames=FIELDS)
        writers[side].writeheader()
    total = 0
    for name in ("public", "private"):
        path = f"{ROOT}/rank_prediction_test_{name}.csv"
        with open(path, encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                for index in range(1, int(row["num_games"]) + 1):
                    color = row[f"color_{index}"]
                    base = {
                        "game_id": f"{row['question_id']}:{index}",
                        "player_id": row["question_id"],
                        "sgf_content": row[f"sgf_{index}"], "note": color,
                    }
                    writers["own"].writerow({**base, "color": color})
                    writers["opp"].writerow({**base, "color": "W" if color == "B" else "B"})
                    total += 1
    print("official games", total)


def main() -> None:
    """Writes present.csv, then the two official game lists."""
    write_present()
    write_official()


if __name__ == "__main__":
    main()
