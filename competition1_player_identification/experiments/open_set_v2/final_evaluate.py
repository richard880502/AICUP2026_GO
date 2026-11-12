"""Evaluates exactly one locked recipe on the untouched 600-player final cohort.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.final_evaluate

No hyperparameters, weights, preprocessing or checkpoints are fitted here.
Repeated invocation verifies the saved result and never selects a new model.
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
from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.open_set_v2 import replay
from competition1_player_identification.experiments.open_set_v2 import seal


def stored_metrics(path: pathlib.Path) -> dict:
    """Computes scores and paired uncertainty from serialized predictions."""
    with np.load(path) as arrays:
        baseline = np.asarray(arrays["baseline"])
        proposed = np.asarray(arrays["proposed"])
    if baseline.shape != (4, 600, 600) or proposed.shape != baseline.shape:
        raise ValueError(
            "Final prediction matrices have the wrong cohort shape"
        )
    if not np.isfinite(baseline).all() or not np.isfinite(proposed).all():
        raise ValueError("Final prediction matrices contain nonfinite values")
    return {
        "baseline": evaluation.summarize(baseline),
        "proposed": evaluation.summarize(proposed),
        "paired_delta": evaluation.paired_delta(proposed, baseline),
    }


def verify_completed(root: pathlib.Path, configuration: dict) -> dict:
    """Replays saved result metrics, preserving the one fixed final decision."""
    report = json.loads((root / "final_results.json").read_text())
    if report["configuration_sha256"] != artifacts.file_digest(
        root / "locked_configuration.json"
    ):
        raise ValueError(
            "Final report does not belong to the sealed configuration"
        )
    if report["prediction_sha256"] != artifacts.file_digest(
        root / "final_scores.npz"
    ):
        raise ValueError("Saved final predictions have changed")
    if report["results"] != stored_metrics(root / "final_scores.npz"):
        raise ValueError(
            "Saved final metrics do not reproduce from predictions"
        )
    seal.verify_files(configuration)
    return report


def run(root: pathlib.Path) -> dict:
    """Runs fixed inference once, only after a verified complete configuration."""
    lock = root / "locked_configuration.json"
    if not lock.exists():
        raise ValueError(
            "Final evaluation requires a prior validated configuration lock"
        )
    configuration = json.loads(lock.read_text())
    seal.verify_files(configuration)
    if (root / "final_results.json").exists():
        return verify_completed(root, configuration)
    attempt = root / "final_attempt.json"
    lock_digest = artifacts.file_digest(lock)
    if attempt.exists():
        previous = json.loads(attempt.read_text())
        if previous["configuration_sha256"] != lock_digest:
            raise ValueError(
                "Cannot replace a configuration after final inference started"
            )
    artifacts.write_report(
        attempt,
        {
            "configuration_sha256": lock_digest,
            "status": "started_fixed_inference",
            "selection_allowed": False,
        },
    )
    sources, baseline = infer.cohort_scores(
        root, configuration, "final_holdout"
    )
    proposed = replay.combined_scores(sources, configuration["weights"])
    temporary = root / "final_scores.tmp"
    with temporary.open("wb") as output:
        np.savez_compressed(output, baseline=baseline, proposed=proposed)
    temporary.replace(root / "final_scores.npz")
    global_rows, _ = infer.positions(root, "final_holdout")
    colors = dataset.load_arrays(root / "raw_cache", ("colors",))["colors"][
        global_rows[:, 10:]
    ]
    report = {
        "status": "complete",
        "configuration_sha256": lock_digest,
        "prediction_sha256": artifacts.file_digest(root / "final_scores.npz"),
        "results": stored_metrics(root / "final_scores.npz"),
        "query_color_subgroups": {
            "baseline": evaluation.color_subgroups(baseline, colors),
            "proposed": evaluation.color_subgroups(proposed, colors),
        },
        "players": 600,
        "candidates": 600,
        "references_per_candidate": 10,
        "questions": 2400,
        "query_counts": evaluation.COUNTS,
        "selected_on_final": False,
        "production_public_unchanged": 0.5236602,
    }
    artifacts.write_report(root / "final_results.json", report)
    artifacts.write_report(
        attempt,
        {
            "configuration_sha256": lock_digest,
            "status": "complete_fixed_inference",
            "selection_allowed": False,
        },
    )
    return verify_completed(root, configuration)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.14, "cuda:0")
    print(json.dumps(run(dataset.ROOT), indent=2))
