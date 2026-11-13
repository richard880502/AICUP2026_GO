"""Builds extraction lists: extra training games and flattened questions.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition1_player_identification.experiments.trunk_style.inputs

Extra games come only from open-set v2 training rows (held-out players and
protected game content stay excluded) and never repeat the 16-game corpus.
"""

import csv
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import cache
from competition1_player_identification.experiments.open_set_v2 import dataset

csv.field_size_limit(10**9)
OPEN = dataset.ROOT
OUT = pathlib.Path("production/trunk_style_20261113/inputs")
DATA = pathlib.Path(
    "data/aicup2026_player_identification_v2_tests_20261108_fix_readme"
)
EXTRA = 32


def choose_extra() -> np.ndarray:
    """Draws up to 32 unused training games per training player."""
    selected = set(np.load(OPEN / "engine_inputs/global_indices.npy").tolist())
    players = dataset.load_arrays(OPEN / "raw_cache", ("players",))["players"]
    with np.load(OPEN / "clean_splits.npz") as splits:
        train = np.asarray(splits["train"])
    generator = np.random.default_rng(20261114)
    chosen: list[int] = []
    for group in dataset.player_groups(players[train])[1]:
        rows = [int(row) for row in train[group] if int(row) not in selected]
        chosen.extend(generator.permutation(rows)[:EXTRA].tolist())
    return np.sort(np.asarray(chosen))


def extra_games() -> None:
    """Writes the chosen games in global row order, keyed by that row."""
    chosen = choose_extra()
    wanted = set(chosen.tolist())
    np.save(OUT / "corpus2_global_indices.npy", chosen)
    row_number = 0
    with (OUT / "corpus2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["game_id", "player_id", "sgf_content", "color"])
        for rank in cache.RANKS:
            with open(
                f"data/training/train_{rank}.csv", encoding="utf-8", newline=""
            ) as source:
                for row in csv.DictReader(source):
                    if row_number in wanted:
                        writer.writerow(
                            [
                                row_number,
                                row["player_id"],
                                row["sgf_content"],
                                row["color"],
                            ]
                        )
                    row_number += 1
    print("extra games", len(chosen))


def questions() -> None:
    """Flattens all four official question files into one game per row."""
    for split in ("public", "private"):
        for suffix in ("", "2"):
            name = f"{'pub' if split == 'public' else 'priv'}{suffix}"
            source_path = DATA / f"player_identification_test{suffix}_{split}.csv"
            with source_path.open(encoding="utf-8", newline="") as source, (
                OUT / f"{name}_queries.csv"
            ).open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["game_id", "sgf_content", "color"])
                for row in csv.DictReader(source):
                    for index in range(1, int(row["num_games"]) + 1):
                        writer.writerow(
                            [
                                f"{row['question_id']}:{index}",
                                row[f"sgf_{index}"],
                                row[f"color_{index}"],
                            ]
                        )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    questions()
    extra_games()
