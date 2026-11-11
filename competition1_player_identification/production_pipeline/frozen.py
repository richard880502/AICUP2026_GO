"""Reuses original-pool encoders for the few-game v2 pools without training.

Run as a module with --input, --output, and --data-dir. Only candidate-held-out
games select the dimension blend and frozen reranker weight.
"""

import argparse
import dataclasses
import json
import pathlib
import shutil

import numpy as np
import torch

from competition1_player_identification.experiments.reranker import run_experiment
from competition1_player_identification.experiments.reranker import run_group_mlp
from competition1_player_identification import train_profile_metric
from competition1_player_identification.production_pipeline import data
from competition1_player_identification.production_pipeline import metric
from competition1_player_identification.production_pipeline import ranking
from competition1_player_identification.production_pipeline import run


def frozen_branch(pool: data.Pool, source: pathlib.Path,
                  dimension: int, device: torch.device) -> metric.Branch:
    """Loads source weights and encodes unseen games with source normalization."""
    candidates, queries = [], []
    for seed in (20260927, 20260928, 20260929):
        checkpoint = torch.load(source / f"arc{dimension}_seed{seed}.pt",
                                map_location="cpu", weights_only=False)
        model = train_profile_metric.ArcFaceModel(
            len(checkpoint["mean"]), dimension, len(checkpoint["players"]),
        ).to(device)
        model.load_state_dict(checkpoint["state"])
        profiles = pool.candidates.profiles
        scaled = ((profiles.reshape(-1, profiles.shape[-1])
                   - checkpoint["mean"]) / checkpoint["scale"]).astype(np.float32)
        candidates.append(metric.encode(model, scaled).reshape(
            *profiles.shape[:2], dimension,
        ))
        queries.append(metric.encode(model, ((pool.queries.profiles
                                               - checkpoint["mean"])
                                              / checkpoint["scale"]).astype(np.float32)))
    embeddings = np.stack(candidates)
    return metric.Branch(embeddings, embeddings, np.stack(queries), [])


def load_reranker(source: pathlib.Path, device: torch.device) -> run_group_mlp.ProductionFit:
    """Loads the source pool's frozen group-softmax model."""
    checkpoint = torch.load(source / "reranker.pt", map_location="cpu",
                            weights_only=False)
    model = run_group_mlp.GroupReranker(len(checkpoint["mean"])).to(device)
    model.load_state_dict(checkpoint["state"])
    return run_group_mlp.ProductionFit(
        model, run_group_mlp.Selection(checkpoint["state"], checkpoint["epoch"],
                                       0.0, 0.0, []),
        checkpoint["mean"], checkpoint["standard_deviation"],
    )


def validation_pairs(context: ranking.Context) -> run_experiment.PairDataset:
    """Creates candidate-only queries disjoint from their reference games."""
    tensors = ranking.make_tensors(context, True)
    rows = []
    for owner, positions in enumerate(context.metric_context.holdout_positions):
        for count in context.metric_context.counts:
            row = run_experiment.retrieval_rows(
                tensors, run_experiment.Task(owner, count, positions[:count]),
                context.weights[count],
            )
            rows.append(dataclasses.replace(
                row, labels=(row.candidate_ids == owner).astype(np.int64),
                owners=np.full(20, owner, dtype=np.int64),
            ))
    return ranking.concatenate_rows(rows)


def select_reranker(context: ranking.Context,
                     fitted: run_group_mlp.ProductionFit) -> dict:
    """Selects reuse strength without updating any neural network weights."""
    pairs = validation_pairs(context)
    logits = ranking.row_standardize(run_group_mlp.predict_production_pairs(
        fitted, pairs, context.metric_context.device,
    ).reshape(-1, 20))
    candidates = np.stack([order[:20] for order in pairs.stage1_orders])
    base = ranking.row_standardize(np.stack([
        scores[order[:20]] for scores, order in zip(
            pairs.stage1_scores, pairs.stage1_orders,
        )
    ]))
    owners = pairs.owners.reshape(-1, 20)[:, :1]
    best = (-1.0, 0.0)
    trials = {}
    for weight in np.linspace(0, 1, 21):
        order = np.argsort(-(weight * logits + (1 - weight) * base),
                           axis=1, kind="stable")
        matches = np.take_along_axis(candidates, order, axis=1) == owners
        ranks = np.argmax(matches, axis=1) + 1
        reward = np.where(matches.any(axis=1) & (ranks <= 5),
                          np.exp(-(ranks - 1)), 0.0)
        score = float(reward.mean())
        trials[str(float(weight))] = score
        if score > best[0]:
            best = (score, float(weight))
    fitted.selection = dataclasses.replace(fitted.selection, weight=best[1])
    return {"selected_weight": best[1], "candidate_validation_score": best[0],
            "stage1_candidate_validation_score": trials["0.0"], "trials": trials}


def predict_pool(arguments: argparse.Namespace, pool_name: str) -> dict:
    """Produces the new pool's predictions from same-split original models."""
    destination = arguments.output / pool_name
    destination.mkdir(parents=True, exist_ok=True)
    source = arguments.input / pool_name.replace("2", "") / "models"
    pool = data.load_pool(arguments.input / pool_name, 2)
    device = torch.device(arguments.device)
    context = ranking.Context(
        metric.make_context(pool, destination, device),
        frozen_branch(pool, source, 512, device),
        frozen_branch(pool, source, 1024, device), {},
    )
    context.weights = ranking.select_weights(context)
    fitted = load_reranker(source, device)
    report = select_reranker(context, fitted)
    report["dimension_weights"] = context.weights
    report["source_models"] = str(source)
    report["neural_weights_updated"] = False
    report["validation_note"] = "Candidate-only selection, not independent accuracy"
    print(pool_name, json.dumps(report), flush=True)
    rows = []
    for identifier, indices in zip(pool.queries.identifiers,
                                    ranking.predict(context, fitted)):
        row = {"question_id": str(identifier)}
        for index, player in enumerate(indices, start=1):
            row[f"top_{index}"] = str(pool.candidates.players[player])
        rows.append(row)
    run.write_csv(destination / "predictions.csv", rows)
    (destination / "selection.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    return report


def main() -> None:
    """Builds a complete alternative submission and its selection report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    reports = {}
    for pool in ("pub", "priv"):
        (arguments.output / pool).mkdir(parents=True, exist_ok=True)
        shutil.copy2(arguments.input / pool / "predictions.csv",
                     arguments.output / pool / "predictions.csv")
    for pool in ("pub2", "priv2"):
        reports[pool] = predict_pool(arguments, pool)
    run.aggregate(arguments)
    (arguments.output / "frozen_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
