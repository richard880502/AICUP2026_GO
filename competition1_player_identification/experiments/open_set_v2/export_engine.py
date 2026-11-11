"""Exports training-only, deduplicated SGFs for reusable engine extraction.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.export_engine

The extractor treats every selected row as an independent game. Query CSV is
header-only because retrieval grouping is defined by clean_splits.npz.
"""

import csv
import json
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import cache
from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset


def selected_games(root: pathlib.Path, limit: int) -> np.ndarray:
    """Selects up to sixteen training games plus all clean evaluation games."""
    arrays = dataset.load_arrays(root / "raw_cache", ("players",))
    with np.load(root / "clean_splits.npz") as splits:
        train = np.asarray(splits["train"])
        _, groups = dataset.player_groups(arrays["players"][train])
        selected = [
            np.asarray(splits["validation"]).ravel(),
            np.asarray(splits["final_holdout"]).ravel(),
        ]
        generator = np.random.default_rng(20261111)
        for indices in groups:
            selected.append(train[generator.permutation(indices)[:limit]])
    return np.unique(np.concatenate(selected))


def export(
    root: pathlib.Path,
    source: pathlib.Path,
    limit: int,
) -> dict:
    """Writes selected original records with validated global row positions."""
    output = root / "engine_inputs"
    output.mkdir(exist_ok=True)
    destination = output / "games.csv"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite extraction input: {destination}"
        )
    selected = selected_games(root, limit)
    wanted = set(selected.tolist())
    records = []
    offset = 0
    for rank in cache.RANKS:
        with (source / f"train_{rank}.csv").open(
            encoding="utf-8",
            newline="",
        ) as input_file:
            for row in csv.DictReader(input_file):
                if offset in wanted:
                    records.append(row)
                offset += 1
    if len(records) != len(selected):
        raise ValueError("Selected indices are outside the training CSV corpus")
    with destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "game_id",
                "player_id",
                "sgf_content",
                "color",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    (output / "questions.csv").write_text(
        "question_id,num_games,sgf_1,color_1\n",
        encoding="utf-8",
    )
    np.save(output / "global_indices.npy", selected)
    report = {
        "selected_games": len(selected),
        "training_game_cap_per_player": limit,
        "evaluation_games": 18000,
        "source": str(source.resolve()),
    }
    artifacts.write_report(output / "export.json", report)
    return report


def main() -> None:
    """Exports all evaluation rows and a bounded training sample per player."""
    parser = artifacts.root_parser(__doc__, dataset.ROOT)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default="data/training",
    )
    parser.add_argument(
        "--train-games",
        type=int,
        default=16,
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            export(
                arguments.root,
                arguments.source,
                arguments.train_games,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
