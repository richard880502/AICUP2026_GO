"""Runs frozen dual-encoder game-pair matching and feature-family ablations.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.matching

Training pairs use only training players. The fixed 600-player validation
selects ensemble, checkpoint and blend. Final-holdout scores are not computed.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import bank
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import pair_data
from competition1_player_identification.experiments.open_set_v2 import (
    pair_features,
)
from competition1_player_identification.experiments.open_set_v2 import (
    pair_model,
)


def save_pairs(path: pathlib.Path, pairs: pair_data.PairDataset) -> None:
    """Caches grouped features without exposing the data outside production."""
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            features=pairs.features,
            targets=pairs.targets,
            top_ids=pairs.top_ids,
            scores=pairs.scores,
            counts=pairs.counts,
            truth_positions=pairs.truth_positions,
        )


def load_pairs(path: pathlib.Path) -> pair_data.PairDataset:
    """Restores complete grouped features in the saved query order."""
    with np.load(path) as values:
        return pair_data.PairDataset(
            np.asarray(values["features"]),
            np.asarray(values["targets"]),
            np.asarray(values["top_ids"]),
            np.asarray(values["scores"]),
            np.asarray(values["counts"]),
            np.asarray(values["truth_positions"]),
        )


def prepare_pairs(arguments: argparse.Namespace) -> tuple:
    """Mines natural hard negatives once, checking frozen source provenance."""
    signature = {
        "checkpoints": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in arguments.checkpoints
        },
        "owners": arguments.owners,
        "bank_device": arguments.device,
        "split_sha256": hashlib.sha256(
            (arguments.root / "clean_splits.npz").read_bytes()
        ).hexdigest(),
        "feature_schema": "stage7_plus_13_per_branch_v1",
        "checkpoint_order": [str(path) for path in arguments.checkpoints],
    }
    manifest = arguments.output / "pair_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["signature"] != signature:
            raise ValueError(
                "Existing pair cache has different frozen sources or protocol"
            )
        return (
            load_pairs(arguments.output / "training_pairs.npz"),
            load_pairs(arguments.output / "validation_pairs.npz"),
            saved,
        )
    first = bank.encode_checkpoint(arguments.checkpoints[0], arguments.device)
    second = bank.encode_checkpoint(arguments.checkpoints[1], arguments.device)
    training_positions, validation_positions = pair_data.row_positions(
        arguments.root
    )
    training_bank = pair_features.build_bank(first, second, training_positions)
    validation_bank = pair_features.build_bank(
        first, second, validation_positions
    )
    weight, trials = pair_data.choose_weight(validation_bank)
    train_pairs = pair_data.synthetic_pairs(
        training_bank, weight, arguments.owners
    )
    validation = pair_data.validation_pairs(validation_bank, weight)
    save_pairs(arguments.output / "training_pairs.npz", train_pairs)
    save_pairs(arguments.output / "validation_pairs.npz", validation)
    report = {
        "signature": signature,
        "stage_weight_b": weight,
        "stage_weight_trials": trials,
        "training_players_eligible": len(training_positions),
        "training_query_owners": arguments.owners,
        "training_questions": len(train_pairs.targets),
        "training_misses_at_20": int((train_pairs.targets < 0).sum()),
        "training_recall": pair_data.stage_recall(train_pairs),
        "validation_recall": pair_data.stage_recall(validation),
        "validation_questions": len(validation.targets),
        "answers_forced_into_topk": False,
    }
    artifacts.write_report(manifest, report)
    return train_pairs, validation, report


def fit_branch(context: tuple) -> dict:
    """Fits a fixed feature family and retains zero-blend fallback scores."""
    train_pairs, validation, arguments, branch = context
    columns = pair_data.feature_columns(branch)
    fitted = pair_model.fit(
        (train_pairs, validation, columns, arguments.device)
    )
    stage = np.reshape(validation.scores, (4, 600, 600))
    scores = pair_features.reorder_scores(
        stage,
        validation.top_ids,
        fitted.logits,
        fitted.selection["blend"],
    )
    checkpoint = {
        "state": fitted.state,
        "mean": fitted.mean,
        "scale": fitted.scale,
        "selection": fitted.selection,
        "columns": columns,
        "branch": branch,
    }
    torch.save(checkpoint, arguments.output / f"pair_{branch}.pt")
    np.save(arguments.output / f"validation_{branch}.npy", scores)
    report = {
        "branch": branch,
        "feature_dimensions": len(columns),
        "selection": fitted.selection,
        "validation": evaluation.summarize(scores),
        "paired_vs_centroid_ensemble": evaluation.paired_delta(scores, stage),
        "history": fitted.history,
    }
    artifacts.write_report(arguments.output / f"ablation_{branch}.json", report)
    return report


def run(arguments: argparse.Namespace) -> None:
    """Records validation ablations without modifying either stage-one file."""
    arguments.output.mkdir(parents=True, exist_ok=True)
    if (arguments.output / "completed.json").exists():
        raise FileExistsError(
            f"Completed matching run exists: {arguments.output}"
        )
    train_pairs, validation, manifest = prepare_pairs(arguments)
    reports = {}
    for branch in ("b", "a", "both"):
        reports[branch] = fit_branch(
            (train_pairs, validation, arguments, branch)
        )
        print(
            {"branch": branch, "selection": reports[branch]["selection"]},
            flush=True,
        )
    selected = max(
        reports,
        key=lambda name: (reports[name]["validation"]["score"], name == "b"),
    )
    for path in arguments.checkpoints:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["signature"]["checkpoints"][str(path)]
        ):
            raise ValueError(
                "Frozen stage-one checkpoint changed during reranking"
            )
    stage = np.reshape(validation.scores, (4, 600, 600))
    np.save(arguments.output / "stage1_validation_scores.npy", stage)
    scores = np.load(arguments.output / f"validation_{selected}.npy")
    np.save(arguments.output / "reranked_validation_scores.npy", scores)
    artifacts.write_report(
        arguments.output / "completed.json",
        {
            "status": "validation_selected",
            "selected_branch": selected,
            "selection": reports[selected]["selection"],
            "stage1_validation": evaluation.summarize(stage),
            "reranked_validation": reports[selected]["validation"],
            "source_manifest": manifest,
            "source_checkpoints_unchanged": True,
            "final_holdout_evaluated": False,
            "reranker_adopted": reports[selected]["selection"]["blend"] > 0,
        },
        emit=True,
    )


def main() -> None:
    """Runs all three matching ablations with explicit checkpoint sources."""
    parser = artifacts.root_parser(__doc__, dataset.ROOT)
    parser.add_argument("--checkpoints", nargs=2, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--owners", type=int, default=1600)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.checkpoints is None:
        arguments.checkpoints = [
            arguments.root / f"encoder_engine_{dimension}/checkpoint.pt"
            for dimension in (512, 1024)
        ]
    if arguments.output is None:
        arguments.output = arguments.root / "matching"
    torch.set_num_threads(4)
    if arguments.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.14, arguments.device)
    run(arguments)


if __name__ == "__main__":
    main()
