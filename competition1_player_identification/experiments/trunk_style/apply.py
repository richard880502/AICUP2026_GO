"""Evaluates the trained style encoder on the final cohort and official pools.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=0 \
      python -m competition1_player_identification.experiments.trunk_style.apply

The checkpoint and its normalization are loaded as saved; nothing is fitted.
The fusion weight comes from the validation trials stored in the checkpoint.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import fusion
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.trunk_style import train
from competition1_player_identification.production_pipeline import data
from competition1_player_identification.production_pipeline import metric
from competition1_player_identification.production_pipeline import (
    open_set_official as official,
)
from competition1_player_identification.production_pipeline import ranking
from competition1_player_identification.production_pipeline import run

ROOT = train.ROOT
INPUT = pathlib.Path("production/player_v2_20261111")
WEIGHTS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0)
OPEN = train.OPEN
DATA = pathlib.Path(
    "data/aicup2026_player_identification_v2_tests_20261108_fix_readme"
)


def load_model(name: str) -> tuple:
    """Restores the selected epoch with its training-only channel scaling."""
    checkpoint = torch.load(
        ROOT / name / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    model = train.StyleEncoder(checkpoint["classes"]).to(train.DEVICE)
    model.load_state_dict(checkpoint["state"])
    scaling = tuple(
        torch.from_numpy(checkpoint[key]).to(train.DEVICE)
        for key in ("mean", "scale")
    )
    trials = checkpoint["validation"]["fusion_trials"]
    weight = max(trials, key=lambda key: trials[key])
    return model, scaling, float(weight), checkpoint


def cohort_scores(model, scaling, games: dict, positions: np.ndarray):
    """Returns centroid scores [4, 600, 600] for a padded cohort layout."""
    embeddings = train.encode(model, games, scaling, positions.ravel())
    return evaluation.similarity_scores(
        np.reshape(embeddings, (600, 15, -1)), "cuda:0"
    )["centroid"]


def final_report(model, scaling, weight: float) -> dict:
    """Reads the untouched final cohort once with the frozen recipe."""
    corpus = train.load_corpus(0)
    _, local = infer.positions(OPEN, "final_holdout")
    scores = cohort_scores(model, scaling, corpus, local)
    base = fusion.standardize(np.load(OPEN / "final_scores.npz")["proposed"])
    mixed = base + weight * fusion.standardize(scores)
    return {
        "style_alone": evaluation.summarize(scores),
        "open_set": evaluation.summarize(base)["score"],
        "fused": evaluation.summarize(mixed),
        "paired_delta_vs_open_set": evaluation.paired_delta(mixed, base),
    }


def pool(  # pylint: disable=too-many-locals
    name: str, model, scaling, weight: float, output: pathlib.Path
):
    """Writes fused Top-5 answers for one official new pool."""
    games, positions, names = official.pool_games(name)
    trunk = np.concatenate(
        [
            train.load_shards(ROOT / f"{name}_{group}", "tokens", 0)
            for group in ("candidates", "queries")
        ]
    )
    masks = np.concatenate(
        [
            train.load_shards(ROOT / f"{name}_{group}", "masks", 0)
            for group in ("candidates", "queries")
        ]
    )
    if len(trunk) != len(games["masks"]):
        raise ValueError("Trunk and engine caches hold different game counts")
    agreement = float((masks == games["masks"]).all(1).mean())
    if agreement < 0.999:
        raise ValueError(f"Decision masks disagree: {agreement}")
    tokens = {
        "trunk": trunk,
        "engine": games["decisions"].astype(np.float16),
        "masks": masks & games["masks"],
        "colors": games["colors"].astype(np.int64),
    }
    scores = cohort_scores(model, scaling, tokens, positions)
    slots = [evaluation.COUNTS.index(int(count)) for count in names["counts"]]
    rows = np.arange(len(slots))
    style = fusion.standardize(scores[slots, rows])
    with np.load(
        official.OUTPUT / name / "scores.npz"
    ) as arrays:
        opened = fusion.standardize(np.asarray(arrays["fused"]))
    mixed = opened + weight * style
    order = np.argsort(-mixed, axis=1, kind="stable")[:, :5]
    alone = np.argsort(-style, axis=1, kind="stable")[:, :5]
    before = np.argsort(-opened, axis=1, kind="stable")[:, :5]
    (output / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / name / "scores.npz", style=style, fused=mixed)
    result = []
    for question, chosen in zip(names["questions"], order):
        row = {"question_id": question}
        for place, player in enumerate(chosen, start=1):
            row[f"top_{place}"] = names["players"][player]
        result.append(row)
    run.write_csv(output / name / "predictions.csv", result)
    margin = np.sort(style, axis=1)[:, ::-1]
    return {
        "mask_agreement": agreement,
        "style_top1_equals_open_set_top1": float(
            (alone[:, 0] == before[:, 0]).mean()
        ),
        "open_set_top1_in_style_top5": float(
            (alone == before[:, :1]).any(1).mean()
        ),
        "style_top1_margin_median_by_count": {
            int(count): float(
                np.median((margin[:, 0] - margin[:, 1])[names["counts"] == count])
            )
            for count in evaluation.COUNTS
        },
    }


def style_games(name: str, group: str, model, scaling) -> np.ndarray:
    """Encodes one original-pool game list (black only) in its file order."""
    features = official.load_shards(
        INPUT / name / "features" / group, ("tokens", "masks", "record_ids")
    )
    directory = ROOT / f"{name}_{group}"
    if not np.array_equal(
        train.load_shards(directory, "record_ids", 0), features["record_ids"]
    ):
        raise ValueError("Trunk and engine caches list different games")
    masks = train.load_shards(directory, "masks", 0)
    if float((masks == features["masks"]).all(1).mean()) < 0.999:
        raise ValueError("Decision masks disagree")
    games = {
        "trunk": train.load_shards(directory, "tokens", 0),
        "engine": features["tokens"].astype(np.float16),
        "masks": masks & features["masks"],
        "colors": np.zeros(len(masks), np.int64),
    }
    return train.encode(model, games, scaling, np.arange(len(masks)))


def held_out_table(  # pylint: disable=too-many-locals
    context: ranking.Context, style: np.ndarray
) -> dict:
    """Scores pool-trained, style and mixed Stage-1 on held-out candidate games."""
    positions = context.metric_context
    owners = np.arange(len(style))[:, None]
    references = [
        ranking.take_positions(branch.development, positions.reference_positions)
        for branch in (context.first, context.second)
    ]
    held = [
        ranking.take_positions(branch.development, positions.holdout_positions)
        for branch in (context.first, context.second)
    ]
    style_reference = ranking.normalize(
        style[owners, positions.reference_positions].mean(1)
    )
    style_held = style[owners, positions.holdout_positions]
    generator = np.random.default_rng(20261113)
    table = {}
    for count in positions.counts:
        totals = np.zeros(len(WEIGHTS) + 1)
        for _ in range(8):
            columns = generator.permutation(style_held.shape[1])[:count]
            parts = [
                ranking.score_matrix(reference, values[:, :, columns])
                for reference, values in zip(references, held)
            ]
            blend = context.weights[count]
            closed = ranking.row_standardize(
                (1 - blend) * parts[0] + blend * parts[1]
            )
            opened = ranking.row_standardize(
                ranking.normalize(style_held[:, columns].mean(1))
                @ style_reference.T
            )
            totals[:-1] += [
                ranking.utility(closed + weight * opened) for weight in WEIGHTS
            ]
            totals[-1] += ranking.utility(opened)
        table[int(count)] = (totals / 8).tolist()
    return table


def original_pool(  # pylint: disable=too-many-locals
    name: str, model, scaling, base: pathlib.Path, output: pathlib.Path
) -> dict:
    """Adds the style score to the pool-trained Stage-1 over the full pool.

    One weight is chosen for all query sizes on held-out candidate games.
    The Top-20 reranker is not applied: it was fitted to the pool-trained
    ordering, which the style score now dominates.
    """
    source = INPUT / name / "models"
    pool_data = data.load_pool(INPUT / name, 2)
    metric_context = metric.make_context(pool_data, source, torch.device("cuda:0"))
    context = ranking.Context(
        metric_context,
        metric.fit_branch(metric_context, 512),
        metric.fit_branch(metric_context, 1024),
        {},
    )
    context.weights = ranking.select_weights(context)
    features = official.load_shards(
        INPUT / name / "features/candidates", ("group_indices",)
    )
    games = style_games(name, "candidates", model, scaling)
    players = len(pool_data.candidates.players)
    style = np.stack(
        [games[features["group_indices"] == index] for index in range(players)]
    )
    table = held_out_table(context, style)
    means = np.mean(list(table.values()), axis=0)
    weight = WEIGHTS[int(np.argmax(means[:-1]))]
    questions = len(pool_data.queries.counts)
    query_games = ranking.normalize(style_games(name, "queries", model, scaling))
    query = np.stack(
        [
            query_games[pool_data.queries.groups == index].mean(0)
            for index in range(questions)
        ]
    )
    opened = ranking.row_standardize(
        ranking.normalize(query) @ ranking.normalize(style.mean(1)).T
    )
    pairs = ranking.make_dataset(context, np.arange(questions), False)
    closed = ranking.row_standardize(np.stack(pairs.stage1_scores))
    order = np.argsort(-(closed + weight * opened), axis=1, kind="stable")[:, :5]
    previous = {
        row["question_id"]: row["top_1"]
        for row in run.read_csv(base / name / "predictions.csv")
    }
    rows, same = [], 0
    for identifier, chosen in zip(pool_data.queries.identifiers, order):
        row = {"question_id": str(identifier)}
        for place, player in enumerate(chosen, start=1):
            row[f"top_{place}"] = str(pool_data.candidates.players[player])
        same += int(previous[row["question_id"]] == row["top_1"])
        rows.append(row)
    (output / name).mkdir(parents=True, exist_ok=True)
    run.write_csv(output / name / "predictions.csv", rows)
    return {
        "held_out_candidate_scores": {
            "weights": list(WEIGHTS) + ["style_only"],
            "by_query_games": table,
            "mean": means.tolist(),
        },
        "selected_weight": weight,
        "top1_same_as_previous_submission": f"{same}/{questions}",
        "style_only_top1_same_as_fused": float(
            (np.argmax(opened, 1) == order[:, 0]).mean()
        ),
    }


def main() -> None:
    """Runs the final-cohort check, both new pools and the assembly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="style_v1")
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=pathlib.Path("production/player_v2_fusion_20261113"),
    )
    arguments = parser.parse_args()
    model, scaling, weight, checkpoint = load_model(arguments.name)
    output = ROOT / f"submission_{arguments.name}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "epoch": checkpoint["epoch"],
        "fusion_weight": weight,
        "validation": checkpoint["validation"],
        "final_holdout": final_report(model, scaling, weight),
    }
    print(json.dumps(report, indent=1), flush=True)
    for name in ("pub2", "priv2"):
        report[name] = pool(name, model, scaling, weight, output)
        print(name, json.dumps(report[name]), flush=True)
    for name in ("pub", "priv"):
        report[name] = original_pool(
            name, model, scaling, arguments.base, output
        )
        print(name, json.dumps(report[name], indent=1), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run.aggregate(argparse.Namespace(output=output, data_dir=DATA))


if __name__ == "__main__":
    main()
