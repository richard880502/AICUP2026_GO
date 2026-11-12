"""Checks frozen inference on validation before any final scores are computed.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.replay

The serialized models, preprocessing and reranker must reproduce source
rankings. Divergences are recorded and reject sealing; final data stays sealed.
"""

import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import fusion
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.open_set_v2 import seal


def combined_scores(matrices: dict, weights: dict) -> np.ndarray:
    """Combines fixed candidate-row standardized sources without refitting."""
    result = np.zeros_like(next(iter(matrices.values())), dtype=np.float32)
    for name, weight in weights.items():
        result += weight * fusion.standardize(matrices[name])
    return result


def source_differences(actual: dict, expected: dict) -> dict:
    """Reports numeric error and every correct-rank discrepancy per source."""
    return {
        name: {
            "maximum_absolute_error": float(
                np.abs(fusion.standardize(values) - expected[name]).max()
            ),
            "rank_disagreements": int(
                (
                    evaluation.correct_ranks(values)
                    != evaluation.correct_ranks(expected[name])
                ).sum()
            ),
        }
        for name, values in actual.items()
    }


def run(root: pathlib.Path) -> dict:
    """Replays validation without changing weights or touching final outcomes."""
    if (root / "locked_configuration.json").exists():
        raise FileExistsError(
            "Configuration is already sealed; do not regenerate replay"
        )
    recipe = seal.recipe_for_validation(root)
    matrices, baseline = infer.cohort_scores(root, recipe, "validation")
    expected, _ = fusion.load_sources(root)
    differences = source_differences(matrices, expected)
    with np.load(root / "fisher_validation_scores.npz") as values:
        expected_baseline = np.asarray(values["centroid"])
    differences["fisher_baseline"] = {
        "maximum_absolute_error": float(
            np.abs(baseline - expected_baseline).max()
        ),
        "rank_disagreements": int(
            (
                evaluation.correct_ranks(baseline)
                != evaluation.correct_ranks(expected_baseline)
            ).sum()
        ),
    }
    passed = all(row["rank_disagreements"] == 0 for row in differences.values())
    proposed = combined_scores(matrices, recipe["weights"])
    original = np.load(root / "fusion_validation_scores.npy")
    if not np.array_equal(
        evaluation.correct_ranks(proposed), evaluation.correct_ranks(original)
    ):
        passed = False
    paths = [
        path
        for path in seal.required_paths(root)
        if path.name != "validation_replay.json"
    ]
    paths.extend(
        pathlib.Path(source["path"])
        for source in json.loads((root / "fusion_validation.json").read_text())[
            "source_manifest"
        ].values()
    )
    report = {
        "status": "passed" if passed else "failed",
        "recipe": recipe,
        "source_differences": differences,
        "baseline": evaluation.summarize(baseline),
        "proposed": evaluation.summarize(proposed),
        "files": {str(path): artifacts.file_digest(path) for path in paths},
        "final_holdout_evaluated": False,
    }
    artifacts.write_report(root / "validation_replay.json", report, emit=True)
    if not passed:
        raise ValueError(
            "Serialized inference did not reproduce validation rankings"
        )
    return report


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    run(dataset.ROOT)
