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
from competition1_player_identification.production_pipeline import (
    open_set_official as official,
)
from competition1_player_identification.production_pipeline import run

ROOT = train.ROOT
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


def cohort_scores(model, scaling, data: dict, positions: np.ndarray):
    """Returns centroid scores [4, 600, 600] for a padded cohort layout."""
    embeddings = train.encode(model, data, scaling, positions.ravel())
    return evaluation.similarity_scores(
        np.reshape(embeddings, (600, 15, -1)), "cuda:0"
    )["centroid"]


def final_report(model, scaling, weight: float) -> dict:
    """Reads the untouched final cohort once with the frozen recipe."""
    data = train.load_corpus(0)
    _, local = infer.positions(OPEN, "final_holdout")
    scores = cohort_scores(model, scaling, data, local)
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
    data = {
        "trunk": trunk,
        "engine": games["decisions"].astype(np.float16),
        "masks": masks & games["masks"],
        "colors": games["colors"].astype(np.int64),
    }
    scores = cohort_scores(model, scaling, data, positions)
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
        (output / name).mkdir(exist_ok=True)
        (output / name / "predictions.csv").write_bytes(
            (arguments.base / name / "predictions.csv").read_bytes()
        )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run.aggregate(argparse.Namespace(output=output, data_dir=DATA))


if __name__ == "__main__":
    main()
