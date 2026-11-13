"""Applies the locked open-set v2 recipe to official new 600-player pools.

Examples (repository root, CUDA_VISIBLE_DEVICES selects the GPU):
    python -m competition1_player_identification.production_pipeline.open_set_official --check
    python -m competition1_player_identification.production_pipeline.open_set_official --pool pub2

No weights, normalization or fusion coefficients are fitted. Query rows are
padded to the experiment layout [600 players, 10 references + 5 queries];
only the slice of each question's real game count is ever read.
"""

import argparse
import csv
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import bank
from competition1_player_identification.experiments.open_set_v2 import cache
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    decision_rank,
)
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.open_set_v2 import models
from competition1_player_identification.experiments.open_set_v2 import opening
from competition1_player_identification.experiments.open_set_v2 import pair_data
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)
from competition1_player_identification.experiments.open_set_v2 import prefix
from competition1_player_identification.experiments.open_set_v2 import replay
from competition1_player_identification.experiments.open_set_v2 import training

ROOT = dataset.ROOT
FEATURES = pathlib.Path("production/player_v2_20261111")
OUTPUT = pathlib.Path("production/player_v2_openset_20261113")
csv.field_size_limit(10**9)


def encode(name: str, games: dict) -> np.ndarray:
    """Encodes games with one frozen checkpoint and its saved normalization."""
    checkpoint = torch.load(
        ROOT / f"encoder_{name}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    specification = models.Specification(**checkpoint["specification"])
    if specification.kind == "sequence":
        values = games["tokens"][:, : specification.length]
    else:
        values = np.clip(
            (games[specification.kind] - checkpoint["mean"])
            / checkpoint["scale"],
            -10,
            10,
        ).astype(np.float32)
    data = training.GameData(
        values,
        games["colors"].astype(np.int64),
        [],
        np.full(len(values), -1, np.int64),
        np.zeros((0, 15), np.int64),
        checkpoint["mean"],
        checkpoint["scale"],
    )
    runtime = bank.frozen_runtime(checkpoint, data, "cuda:0")
    return training.encode(runtime, np.arange(len(values)))


def rank_probabilities(games: dict) -> np.ndarray:
    """Replays the three-seed decision rank prior with its fixed temperature."""
    with np.load(ROOT / "rank_engine/preprocessing.npz") as preprocessing:
        normalized = decision_rank.normalize(
            games["decisions"],
            games["masks"],
            preprocessing["mean"],
            preprocessing["scale"],
        )
    data = decision_rank.RankData(
        normalized,
        games["masks"],
        np.full(len(normalized), -1, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros((0, 15), np.int64),
    )
    predictions = decision_rank.ensemble(
        ROOT / "rank_engine", data, np.arange(len(normalized))
    )
    temperature = json.loads((ROOT / "rank_engine/completed.json").read_text())[
        "temperature"
    ]
    predictions = predictions.clip(min=1e-7) ** (1 / temperature)
    return predictions / predictions.sum(1, keepdims=True)


def matching(games: dict, positions: np.ndarray, recipe: dict) -> np.ndarray:
    """Applies the saved Top-20 pair reranker to both frozen engine branches."""
    reference = pair_features.build_bank(
        encode("engine_512", games), encode("engine_1024", games), positions
    )
    rows = []
    for count in evaluation.COUNTS:
        for query in range(600):
            task = pair_features.Task(query, count, np.arange(600))
            rows.append(
                pair_features.make_rows(
                    (reference, task, recipe["stage_weight_b"], "both")
                )
            )
    pairs = pair_data.assemble(rows, list(np.repeat(evaluation.COUNTS, 600)))
    stage = np.reshape(pairs.scores, (4, 600, 600))
    logits = infer.pair_logits(ROOT, pairs, recipe["branch"])
    return pair_features.reorder_scores(
        stage, pairs.top_ids, logits, recipe["blend"]
    )


def fused_scores(games: dict, positions: np.ndarray) -> tuple[np.ndarray, dict]:
    """Returns locked-fusion scores [4 counts, 600 rows, 600 candidates]."""
    recipe = json.loads((ROOT / "locked_configuration.json").read_text())
    required = set(recipe["weights"])
    shape = (600, 15, -1)
    matrices = {}
    for name, aggregation in recipe["aggregations"].items():
        if name in required:
            matrices[name] = evaluation.similarity_scores(
                encode(name, games)[positions], "cuda:0"
            )[aggregation]
    flat = positions.ravel()
    if "prefix" in required:
        hashes = prefix.fingerprints(
            games["tokens"][flat], games["colors"][flat]
        )
        matrices["prefix"] = prefix.score_prefixes(np.reshape(hashes, shape))
    if "rank_engine" in required:
        matrices["rank_engine"] = evaluation.rank_affinity(
            np.reshape(rank_probabilities(games)[flat], (600, 15, 10))
        )
    if "learned_pair" in required:
        matrices["learned_pair"] = matching(
            games, positions, recipe["matching"]
        )
    if set(matrices) != required:
        raise ValueError(f"Unsupported sources: {required - set(matrices)}")
    return replay.combined_scores(matrices, recipe["weights"]), matrices


def holdout_games() -> tuple[dict, np.ndarray]:
    """Rebuilds the sealed final cohort through the generic game interface."""
    global_rows, local_rows = infer.positions(ROOT, "final_holdout")
    raw = dataset.load_arrays(ROOT / "raw_cache", ("tokens", "colors"))
    decisions, masks = decision_rank.load_tokens(ROOT)
    local = local_rows.ravel()
    games = {
        "engine": np.load(ROOT / "engine_profiles.npy", mmap_mode="r")[local],
        "opening": np.load(ROOT / "opening_profiles.npy", mmap_mode="r")[local],
        "tokens": raw["tokens"][global_rows.ravel()],
        "colors": raw["colors"][global_rows.ravel()],
        "decisions": decisions[local],
        "masks": masks[local],
    }
    return games, np.arange(9000).reshape(600, 15)


def load_shards(directory: pathlib.Path, keys: tuple) -> dict:
    """Concatenates extraction shards in their immutable sorted order."""
    parts: dict = {key: [] for key in keys}
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as arrays:
            for key in keys:
                parts[key].append(np.asarray(arrays[key]))
    return {key: np.concatenate(value) for key, value in parts.items()}


def source_games(metadata: dict) -> tuple[dict, dict]:
    """Reads official SGF text and actor colors keyed like extraction ids."""
    content, colors = {}, {}
    with open(metadata["candidates"], encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            content[row["game_id"]] = row["sgf_content"]
            colors[row["game_id"]] = row["color"]
    with open(metadata["questions"], encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            for index in range(1, int(row["num_games"]) + 1):
                key = f"{row['question_id']}:{index}"
                content[key] = row[f"sgf_{index}"]
                colors[key] = row[f"color_{index}"]
    return content, colors


def pool_games(  # pylint: disable=too-many-locals
    pool: str,
) -> tuple[dict, np.ndarray, dict]:
    """Loads one official pool and pads questions into the cohort layout."""
    base = FEATURES / pool
    metadata = json.loads((base / "features/metadata.json").read_text())
    content, colors = source_games(metadata)
    parts = []
    for group in ("candidates", "queries"):
        features = load_shards(
            base / "features" / group,
            ("tokens", "masks", "group_indices", "record_ids"),
        )
        profiles = load_shards(
            base / "profiles" / group, ("profiles", "record_ids")
        )
        if not np.array_equal(features["record_ids"], profiles["record_ids"]):
            raise ValueError("Feature and profile shards are misaligned")
        features["profiles"] = profiles["profiles"]
        parts.append(features)
    candidates, queries = parts[0], parts[1]
    players, questions = metadata["player_ids"], metadata["question_ids"]
    counts = np.asarray(metadata["query_game_counts"])
    if (
        len(players) != 600
        or (np.bincount(candidates["group_indices"]) != 10).any()
        or not np.array_equal(np.bincount(queries["group_indices"]), counts)
        or not set(counts.tolist()) <= set(evaluation.COUNTS)
    ):
        raise ValueError("Pool is not 600 x 10 with 1/2/3/5-game questions")
    identifiers = np.concatenate(
        (candidates["record_ids"], queries["record_ids"])
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate game identifiers")
    tokens = np.stack(
        [cache.parse_game(content[key])[1] for key in identifiers]
    )
    color = np.asarray([int(colors[key] == "W") for key in identifiers])
    games = {
        "engine": np.concatenate((candidates["profiles"], queries["profiles"])),
        "opening": opening.position_histograms(tokens, color),
        "tokens": tokens,
        "colors": color,
        "decisions": np.concatenate((candidates["tokens"], queries["tokens"])),
        "masks": np.concatenate((candidates["masks"], queries["masks"])),
    }
    positions = np.zeros((600, 15), np.int64)
    for owner in range(600):
        positions[owner, :10] = np.flatnonzero(
            candidates["group_indices"] == owner
        )
    offset = len(candidates["record_ids"])
    for row in range(600):
        # Rows beyond the real questions reuse question 0 and are discarded.
        own = offset + np.flatnonzero(
            queries["group_indices"] == (row if row < len(questions) else 0)
        )
        positions[row, 10:] = np.resize(own, 5)
    names = {"players": players, "questions": questions, "counts": counts}
    return games, positions, names


def check() -> None:
    """Requires the generic path to reproduce the sealed final predictions."""
    games, positions = holdout_games()
    scores, _ = fused_scores(games, positions)
    with np.load(ROOT / "final_scores.npz") as arrays:
        expected = np.asarray(arrays["proposed"])
    report = {
        "score": evaluation.summarize(scores)["score"],
        "expected": evaluation.summarize(expected)["score"],
        "rank_disagreements": int(
            (
                evaluation.correct_ranks(scores)
                != evaluation.correct_ranks(expected)
            ).sum()
        ),
        "max_abs_error": float(np.abs(scores - expected).max()),
    }
    print(json.dumps(report, indent=2))


def predict(pool: str) -> None:
    """Writes Top-5 predictions and raw fused scores for one official pool."""
    games, positions, names = pool_games(pool)
    scores, matrices = fused_scores(games, positions)
    destination = OUTPUT / pool
    destination.mkdir(parents=True, exist_ok=True)
    slots = [evaluation.COUNTS.index(int(count)) for count in names["counts"]]
    rows = np.arange(len(slots))
    kept = scores[slots, rows]
    order = np.argsort(-kept, axis=1, kind="stable")[:, :5]
    np.savez_compressed(
        destination / "scores.npz",
        fused=kept,
        **{name: values[slots, rows] for name, values in matrices.items()},
    )
    with (destination / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["question_id"] + [f"top_{i}" for i in range(1, 6)])
        for question, chosen in zip(names["questions"], order):
            writer.writerow(
                [question] + [names["players"][index] for index in chosen]
            )
    print(json.dumps({"pool": pool, "questions": len(rows)}))


def main() -> None:
    """Runs the reproduction check or one official pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pool", choices=("pub2", "priv2"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    if arguments.check:
        check()
    if arguments.pool:
        predict(arguments.pool)


if __name__ == "__main__":
    main()
