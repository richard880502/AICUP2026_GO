"""Fuses the pool-trained ArcFace pipeline with the open-set v2 recipe.

Run from the repository root:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.production_pipeline.fusion_official

Original pools: the open-set engine ensemble is added to the pool-trained
Stage-1 score; its weight is chosen on candidate-held-out games only.
New pools: original-pool encoders are added to the locked open-set fusion;
their weight is chosen on the open-set validation cohort and reported once
on the final cohort. No official query answers exist or are used.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification import train_profile_metric
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import fusion
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.reranker import run_group_mlp
from competition1_player_identification.production_pipeline import data
from competition1_player_identification.production_pipeline import frozen
from competition1_player_identification.production_pipeline import metric
from competition1_player_identification.production_pipeline import (
    open_set_official as official,
)
from competition1_player_identification.production_pipeline import ranking
from competition1_player_identification.production_pipeline import run

INPUT = pathlib.Path("production/player_v2_20261111")
OPEN = pathlib.Path("production/player_v2_openset_20261113")
OUTPUT = pathlib.Path("production/player_v2_fusion_20261113")
DATA = pathlib.Path(
    "data/aicup2026_player_identification_v2_tests_20261108_fix_readme"
)
SEEDS = (20260927, 20260928, 20260929)
GRID = tuple(round(0.1 * step, 1) for step in range(0, 21))
DEVICE = torch.device("cuda:0")


def open_embeddings(profiles: np.ndarray) -> list[np.ndarray]:
    """Encodes black-or-mixed games with both frozen open-set style encoders."""
    flat = profiles.reshape(-1, profiles.shape[-1])
    games = {"engine": flat, "colors": np.zeros(len(flat), np.int64)}
    return [
        ranking.normalize(official.encode(name, games)).reshape(
            *profiles.shape[:-1], -1
        )
        for name in ("engine_512", "engine_1024")
    ]


def centroid_scores(queries: list, references: list) -> np.ndarray:
    """Averages both branches' centroid cosines, then standardizes each row."""
    parts = [
        ranking.normalize(query) @ ranking.normalize(reference).T
        for query, reference in zip(queries, references)
    ]
    return ranking.row_standardize(0.5 * np.sum(parts, axis=0))


def development_selection(  # pylint: disable=too-many-locals
    context: ranking.Context, candidates: list
) -> dict:
    """Chooses the open-set weight per query size on held-out candidate games."""
    positions = context.metric_context
    owners = np.arange(len(positions.reference_positions))[:, None]
    closed_reference = [
        ranking.take_positions(branch.development, positions.reference_positions)
        for branch in (context.first, context.second)
    ]
    closed_held = [
        ranking.take_positions(branch.development, positions.holdout_positions)
        for branch in (context.first, context.second)
    ]
    open_reference = [
        values[owners, positions.reference_positions].mean(1)
        for values in candidates
    ]
    open_held = [
        values[owners, positions.holdout_positions] for values in candidates
    ]
    generator = np.random.default_rng(20261113)
    report = {}
    for count in positions.counts:
        totals = np.zeros(len(GRID))
        for _ in range(8):
            columns = generator.permutation(open_held[0].shape[1])[:count]
            parts = [
                ranking.score_matrix(reference, held[:, :, columns])
                for reference, held in zip(closed_reference, closed_held)
            ]
            blend = context.weights[count]
            closed = ranking.row_standardize(
                (1 - blend) * parts[0] + blend * parts[1]
            )
            opened = centroid_scores(
                [held[:, columns].mean(1) for held in open_held], open_reference
            )
            alone = ranking.utility(opened)
            totals += [
                ranking.utility(closed + weight * opened) for weight in GRID
            ]
        totals /= 8
        best = int(np.argmax(totals))
        report[int(count)] = {
            "weight": GRID[best],
            "pool_trained_only": float(totals[0]),
            "fused": float(totals[best]),
            "open_set_only_last_draw": float(alone),
        }
    return report


def original_pool(name: str) -> dict:  # pylint: disable=too-many-locals
    """Adds open-set evidence inside the existing Top-20 reranking step."""
    source = INPUT / name / "models"
    pool = data.load_pool(INPUT / name, 2)
    context = ranking.Context(
        metric.make_context(pool, source, DEVICE),
        metric.fit_branch(metric.make_context(pool, source, DEVICE), 512),
        metric.fit_branch(metric.make_context(pool, source, DEVICE), 1024),
        {},
    )
    context.weights = ranking.select_weights(context)
    fitted = frozen.load_reranker(source, DEVICE)
    strength = float(
        torch.load(
            source / "reranker.pt", map_location="cpu", weights_only=False
        )["weight"]
    )
    candidates = open_embeddings(pool.candidates.profiles)
    selection = development_selection(context, candidates)
    query_games = open_embeddings(pool.queries.profiles)
    groups = pool.queries.groups
    question_count = len(pool.queries.counts)
    opened = centroid_scores(
        [
            np.stack(
                [values[groups == index].mean(0) for index in range(question_count)]
            )
            for values in query_games
        ],
        [values.mean(1) for values in candidates],
    )
    pairs = ranking.make_dataset(context, np.arange(question_count), False)
    logits = run_group_mlp.predict_production_pairs(
        fitted, pairs, DEVICE
    ).reshape(-1, 20)
    rows, unchanged, top1_same = [], 0, 0
    for index, order in enumerate(pairs.stage1_orders):
        top = order[:20]
        stage = ranking.row_standardize(pairs.stage1_scores[index])
        learned = strength * ranking.row_standardize(logits[index])
        previous = top[
            np.argsort(
                -(learned + (1 - strength) * ranking.row_standardize(stage[top])),
                kind="stable",
            )
        ][:5]
        weight = selection[int(pool.queries.counts[index])]["weight"]
        mixed = stage + weight * opened[index]
        chosen = top[
            np.argsort(
                -(learned + (1 - strength) * ranking.row_standardize(mixed[top])),
                kind="stable",
            )
        ][:5]
        unchanged += int(np.array_equal(previous, chosen))
        top1_same += int(previous[0] == chosen[0])
        row = {"question_id": str(pool.queries.identifiers[index])}
        for place, player in enumerate(chosen, start=1):
            row[f"top_{place}"] = str(pool.candidates.players[player])
        rows.append((row, previous))
    reference = {
        item["question_id"]: [item[f"top_{place}"] for place in range(1, 6)]
        for item in run.read_csv(INPUT / name / "predictions.csv")
    }
    reproduced = sum(
        reference[row["question_id"]]
        == [str(pool.candidates.players[player]) for player in previous]
        for row, previous in rows
    )
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    run.write_csv(OUTPUT / name / "predictions.csv", [row for row, _ in rows])
    return {
        "development_selection": selection,
        "reranker_strength": strength,
        "previous_pipeline_reproduced": f"{reproduced}/{question_count}",
        "top5_unchanged": unchanged,
        "top1_unchanged": top1_same,
    }


def frozen_cohort_scores(profiles: np.ndarray, sources: tuple) -> np.ndarray:
    """Scores a [600, 15] layout with original-pool encoders, no fitting."""
    parts = []
    flat = profiles.reshape(-1, profiles.shape[-1])
    for source in sources:
        for dimension in (512, 1024):
            for seed in SEEDS:
                checkpoint = torch.load(
                    INPUT / source / "models" / f"arc{dimension}_seed{seed}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = train_profile_metric.ArcFaceModel(
                    len(checkpoint["mean"]), dimension, len(checkpoint["players"])
                ).to(DEVICE)
                model.load_state_dict(checkpoint["state"])
                scaled = (
                    (flat - checkpoint["mean"]) / checkpoint["scale"]
                ).astype(np.float32)
                embeddings = np.reshape(
                    metric.encode(model, scaled), (600, 15, -1)
                )
                parts.append(
                    fusion.standardize(
                        evaluation.similarity_scores(embeddings, "cuda:0")[
                            "centroid"
                        ]
                    )
                )
    return fusion.standardize(np.mean(parts, axis=0))


def new_pool_selection() -> dict:
    """Selects one weight on validation and reports it once on final."""
    root = official.ROOT
    profiles = np.load(root / "engine_profiles.npy", mmap_mode="r")
    cohorts = {}
    for cohort in ("validation", "final_holdout"):
        _, local = infer.positions(root, cohort)
        cohorts[cohort] = np.asarray(profiles[local.ravel()]).reshape(600, 15, -1)
    base = {
        "validation": fusion.standardize(
            np.load(root / "fusion_validation_scores.npy")
        ),
        "final_holdout": fusion.standardize(
            np.load(root / "final_scores.npz")["proposed"]
        ),
    }
    report: dict = {"sources": {}}
    best: tuple[float, tuple[str, ...], float] = (-1.0, ("pub",), 0.0)
    for sources in (("pub",), ("priv",), ("pub", "priv")):
        scores = frozen_cohort_scores(cohorts["validation"], sources)
        trials = {
            weight: fusion.score_value(base["validation"] + weight * scores)
            for weight in GRID
        }
        weight = max(
            (score, -candidate) for candidate, score in trials.items()
        )[1] * -1
        report["sources"]["+".join(sources)] = {
            "validation_alone": fusion.score_value(scores),
            "validation_trials": trials,
            "weight": weight,
        }
        if trials[weight] > best[0]:
            best = (trials[weight], sources, weight)
    final = frozen_cohort_scores(cohorts["final_holdout"], best[1])
    mixed = base["final_holdout"] + best[2] * final
    report["selected"] = {"sources": list(best[1]), "weight": best[2]}
    report["validation"] = {
        "open_set": fusion.score_value(base["validation"]),
        "fused": best[0],
    }
    report["final_holdout"] = {
        "open_set": evaluation.summarize(base["final_holdout"]),
        "original_pool_encoders_alone": fusion.score_value(final),
        "fused": evaluation.summarize(mixed),
        "paired_delta": evaluation.paired_delta(mixed, base["final_holdout"]),
    }
    return report


def new_pool(name: str, selected: dict) -> dict:  # pylint: disable=too-many-locals
    """Adds original-pool encoder scores to the saved open-set fusion."""
    games, positions, names = official.pool_games(name)
    scores = frozen_cohort_scores(
        games["engine"][positions], tuple(selected["sources"])
    )
    slots = [evaluation.COUNTS.index(int(count)) for count in names["counts"]]
    rows = np.arange(len(slots))
    with np.load(OPEN / name / "scores.npz") as arrays:
        opened = fusion.standardize(np.asarray(arrays["fused"]))
    mixed = opened + selected["weight"] * scores[slots, rows]
    order = np.argsort(-mixed, axis=1, kind="stable")[:, :5]
    before = np.argsort(-opened, axis=1, kind="stable")[:, :5]
    output = []
    for question, chosen in zip(names["questions"], order):
        row = {"question_id": question}
        for place, player in enumerate(chosen, start=1):
            row[f"top_{place}"] = names["players"][player]
        output.append(row)
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    run.write_csv(OUTPUT / name / "predictions.csv", output)
    return {
        "top1_unchanged_vs_open_set": int((order[:, 0] == before[:, 0]).sum()),
        "top5_unchanged_vs_open_set": int((order == before).all(1).sum()),
    }


def main() -> None:
    """Runs both selections, all four pools and the validated assembly."""
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"new_pool_selection": new_pool_selection()}
    print(json.dumps(report, indent=1), flush=True)
    for name in ("pub2", "priv2"):
        report[name] = new_pool(name, report["new_pool_selection"]["selected"])
        print(name, json.dumps(report[name]), flush=True)
    for name in ("pub", "priv"):
        report[name] = original_pool(name)
        print(name, json.dumps(report[name], indent=1), flush=True)
    (OUTPUT / "fusion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    run.aggregate(argparse.Namespace(output=OUTPUT, data_dir=DATA))


if __name__ == "__main__":
    main()
