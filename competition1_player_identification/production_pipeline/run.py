"""Produces and validates the four-pool player-identification v2 submission.

Example:
    python -m competition1_player_identification.production_pipeline.run \
        --pool pub2 --data-dir data/tests --output production/player_v2
"""

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.production_pipeline import data
from competition1_player_identification.production_pipeline import metric
from competition1_player_identification.production_pipeline import ranking


POOLS = {
    "pub": ("public", ""),
    "pub2": ("public", "2"),
    "priv": ("private", ""),
    "priv2": ("private", "2"),
}
HEADER = ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    """Reads published or generated UTF-8 CSV rows."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    """Atomically writes the exact official submission columns."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def input_paths(directory: pathlib.Path, pool: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolves a question file and its corresponding candidate library."""
    split, suffix = POOLS[pool]
    return (
        directory / f"player_identification_test{suffix}_{split}.csv",
        directory / f"player_identification_candidates{suffix}_{split}.csv",
    )


def pool_predictions(arguments: argparse.Namespace) -> pathlib.Path:
    """Fits only candidates and predicts one pool of unlabeled questions."""
    root = arguments.output / arguments.pool
    output = root / "models"
    output.mkdir(parents=True, exist_ok=True)
    pool = data.load_pool(root, arguments.workers)
    context = metric.make_context(pool, output, torch.device(arguments.device))
    retrieval = ranking.Context(context, metric.fit_branch(context, 512),
                                metric.fit_branch(context, 1024), {})
    retrieval.weights = ranking.select_weights(retrieval)
    fitted = ranking.fit_reranker(retrieval)
    indices = ranking.predict(retrieval, fitted)
    rows = []
    for identifier, candidates in zip(pool.queries.identifiers, indices):
        row = {"question_id": str(identifier)}
        for position, candidate in enumerate(candidates, start=1):
            row[f"top_{position}"] = str(pool.candidates.players[candidate])
        rows.append(row)
    path = root / "predictions.csv"
    write_csv(path, rows)
    print(f"Saved {len(rows)} questions: {path}", flush=True)
    return path


def expected_questions(directory: pathlib.Path) -> tuple[dict, dict]:
    """Loads candidate membership and question-to-pool mappings from inputs."""
    question_pools = {}
    candidate_pools = {}
    for pool in POOLS:
        questions, candidates = input_paths(directory, pool)
        candidate_pools[pool] = {row["player_id"] for row in read_csv(candidates)}
        for row in read_csv(questions):
            identifier = row["question_id"]
            if identifier in question_pools:
                raise ValueError(f"Duplicate source question ID: {identifier!r}")
            question_pools[identifier] = pool
    return question_pools, candidate_pools


def validate(rows: list[dict[str, str]], directory: pathlib.Path) -> dict:
    """Rejects missing, duplicate, cross-pool, or malformed Top-5 answers."""
    question_pools, candidate_pools = expected_questions(directory)
    identifiers = [row["question_id"] for row in rows]
    if len(identifiers) != 1600 or len(set(identifiers)) != 1600:
        raise ValueError("Submission must contain exactly 1,600 unique questions")
    if set(identifiers) != set(question_pools):
        raise ValueError("Submission question IDs do not match all four test files")
    for row in rows:
        if list(row) != HEADER:
            raise ValueError(f"Invalid submission columns: {list(row)!r}")
        pool = question_pools[row["question_id"]]
        answers = [row[key] for key in HEADER[1:]]
        if len(set(answers)) != 5:
            raise ValueError(f"Duplicate Top-5 answers: {row['question_id']!r}")
        for answer in answers:
            if answer not in candidate_pools[pool] or not answer.startswith(
                f"{pool}_p_",
            ):
                raise ValueError(f"Invalid {pool!r} candidate: {answer!r}")
    return {
        "verdict": "PASS", "questions": len(rows),
        "per_pool": {pool: identifiers_count(question_pools, pool)
                     for pool in POOLS},
        "candidate_players": {pool: len(players)
                              for pool, players in candidate_pools.items()},
        "query_labels_used": False,
    }


def identifiers_count(question_pools: dict, pool: str) -> int:
    """Counts official questions assigned to a pool."""
    return sum(value == pool for value in question_pools.values())


def file_hash(path: pathlib.Path) -> str:
    """Computes a streaming SHA-256 provenance digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(arguments: argparse.Namespace) -> None:
    """Writes the complete submission in the published sample's row order."""
    predictions = []
    for pool in POOLS:
        predictions.extend(read_csv(arguments.output / pool / "predictions.csv"))
    report = validate(predictions, arguments.data_dir)
    by_question = {row["question_id"]: row for row in predictions}
    sample = read_csv(arguments.data_dir / "sample_submission_player_v2.csv")
    rows = [by_question[row["question_id"]] for row in sample]
    validate(rows, arguments.data_dir)
    destination = arguments.output / "submission_player.csv"
    write_csv(destination, rows)
    validate(read_csv(destination), arguments.data_dir)
    report["sha256"] = file_hash(destination)
    report["input_sha256"] = {
        path.name: file_hash(path) for path in arguments.data_dir.glob("*.csv")
    }
    (arguments.output / "submission_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"UPLOAD FILE: {destination.resolve()}", flush=True)


def main() -> None:
    """Runs one pool or assembles all four validated prediction files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=tuple(POOLS) + ("aggregate",),
                        required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("Profile workers must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(20260927)
    if arguments.pool == "aggregate":
        aggregate(arguments)
    else:
        pool_predictions(arguments)


if __name__ == "__main__":
    main()
