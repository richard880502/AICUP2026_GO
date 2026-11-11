"""Creates immutable player splits and v2 retrieval task manifests.

Run from the repository root:
    python -m competition1_player_identification.experiments.open_set_v2.prepare

Only the training SQLite index is read. Official candidate and test files are
excluded. SGF-content deduplication must run before training or evaluation.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sqlite3


QUERY_COUNTS = (1, 2, 3, 5)


def read_games(database: pathlib.Path) -> dict[str, list[dict]]:
    """Reads game pointers from an existing, read-only training index."""
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        for player, path, offset, color in connection.execute(
            "SELECT player_id, path, offset, color FROM games "
            "ORDER BY player_id, path, offset",
        ):
            if color not in ("B", "W"):
                raise ValueError(f"Unexpected player color: {color!r}")
            grouped[player].append({
                "path": path,
                "offset": offset,
                "color": color,
            })
    finally:
        connection.close()
    return dict(grouped)


def make_cohort(
    players: list[str],
    games: dict[str, list[dict]],
    generator: random.Random,
) -> dict:
    """Uses ten references and disjoint, nested queries for each player."""
    references = {}
    questions = []
    for player in players:
        selected = generator.sample(games[player], 15)
        references[player] = selected[:10]
        for count in QUERY_COUNTS:
            questions.append({
                "question_id": f"{player}_{count}",
                "player_id": player,
                "num_games": count,
                "games": selected[10:10 + count],
            })
    return {"references": references, "questions": questions}


def prepare(database: pathlib.Path, output: pathlib.Path, seed: int) -> dict:
    """Writes reproducible splits without consuming any official labels."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite split directory: {output}")
    games = read_games(database)
    eligible = []
    for player, records in games.items():
        counts = collections.Counter(record["color"] for record in records)
        if len(records) >= 15 and min(counts["B"], counts["W"]) >= 5:
            eligible.append(player)
    if len(eligible) < 1200:
        raise ValueError(f"Need 1200 eligible players; found {len(eligible)}")
    generator = random.Random(seed)
    generator.shuffle(eligible)
    splits = {
        "train": sorted(set(games) - set(eligible[:1200])),
        "validation": sorted(eligible[:600]),
        "final_holdout": sorted(eligible[600:1200]),
    }
    output.mkdir(parents=True)
    (output / "players.json").write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, players in (
        ("validation", splits["validation"]),
        ("final_holdout", splits["final_holdout"]),
    ):
        payload = make_cohort(players, games, generator)
        (output / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "seed": seed,
        "training_players": len(splits["train"]),
        "validation_players": len(splits["validation"]),
        "final_holdout_players": len(splits["final_holdout"]),
        "eligible_mixed_color_players": len(eligible),
        "total_games": sum(len(records) for records in games.values()),
        "candidate_count": 600,
        "reference_games": 10,
        "query_counts": QUERY_COUNTS,
        "questions_per_cohort": 2400,
        "query_sampling": "uniform games; nested 1/2/3/5; both colors allowed",
        "player_disjoint": True,
        "content_deduplication": "pending; must pass before model fitting",
        "manifest_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.glob("*.json"))
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Parses paths and writes the audit summary to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=pathlib.Path,
        default=pathlib.Path("data/games_with_color.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("production/open_set_v2_20261111/manifests"),
    )
    parser.add_argument("--seed", type=int, default=20261111)
    arguments = parser.parse_args()
    print(json.dumps(prepare(
        arguments.database,
        arguments.output,
        arguments.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
