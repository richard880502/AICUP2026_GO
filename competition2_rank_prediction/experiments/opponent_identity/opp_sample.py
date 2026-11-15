"""Writes a sample of training games with the colour flipped (opponent side).

Reproduces Competition 2 rank submission v47, step 3 of 8: writes
`opp_probe/opp_sample.csv`, the training-game opponent queries of
link_test.py (their trunk tokens go to `opp_probe/tokens`).

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition2_rank_prediction.experiments.opponent_identity.opp_sample 6144
"""
import csv
import random
import sys

csv.field_size_limit(10**9)
SOURCE = "production/open_set_v2_20261111/engine_inputs/games.csv"
# The research machine's location for the opponent-identity intermediate files.
OUT = "/tmp2/richard/claude_scratch/opp_probe"


def main() -> None:
    """Samples argv[1] training games and flips their colour."""
    count = int(sys.argv[1])
    with open(SOURCE, encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    random.seed(7)
    picks = sorted(random.sample(range(len(rows)), count))
    with open(f"{OUT}/opp_sample.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["game_id", "player_id", "sgf_content", "color", "corpus_row"]
        )
        writer.writeheader()
        for index in picks:
            row = dict(rows[index])
            row["color"] = "W" if row["color"] == "B" else "B"
            row["corpus_row"] = index
            writer.writerow(row)
    print(len(rows), "games;", count, "sampled")


if __name__ == "__main__":
    main()
