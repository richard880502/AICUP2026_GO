"""Locks a complete validation-selected experiment before final evaluation.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.seal

Every required stage must be complete. Existing locks are verified, never
silently replaced. This command never computes final-holdout retrieval scores.
"""

import json
import pathlib

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset

REQUIRED_BRANCHES = ("opening", "sequence", "engine_512", "engine_1024")


def required_paths(root: pathlib.Path) -> list[pathlib.Path]:
    """Enumerates source data, model, normalization and decision artifacts."""
    paths = [
        root / "source_audit.json",
        root / "split_audit.json",
        root / "clean_splits.npz",
        root / "engine_inputs/global_indices.npy",
        root / "engine_profile_audit.json",
        root / "engine_profiles.npy",
        root / "opening_profiles.npy",
        root / "fisher_parameters.npz",
        root / "fisher_validation_scores.npz",
        root / "fusion_validation.json",
        root / "validation_replay.json",
        root / "matching/completed.json",
        root / "rank_prior/checkpoint.pt",
        root / "rank_prior/completed.json",
        root / "rank_engine/completed.json",
        root / "rank_engine/preprocessing.npz",
    ]
    for branch in REQUIRED_BRANCHES:
        paths.extend(
            (
                root / f"encoder_{branch}/checkpoint.pt",
                root / f"encoder_{branch}/completed.json",
            )
        )
    for branch in ("a", "b", "both"):
        paths.extend(
            (
                root / f"matching/pair_{branch}.pt",
                root / f"matching/ablation_{branch}.json",
            )
        )
    paths.extend(sorted((root / "raw_cache").glob("*.npz")))
    paths.extend(
        root / f"rank_engine/seed_{seed}.pt"
        for seed in (20261111, 20261112, 20261113)
    )
    paths.extend(sorted(pathlib.Path(__file__).parent.glob("*.py")))
    return paths


def check_complete(root: pathlib.Path) -> tuple[dict, dict]:
    """Rejects missing ablations, source changes, and any already-opened final."""
    for name in ("source_audit", "split_audit", "engine_profile_audit"):
        if (
            json.loads((root / f"{name}.json").read_text())["status"]
            != "passed"
        ):
            raise ValueError(f"Required audit has not passed: {name}")
    fusion = json.loads((root / "fusion_validation.json").read_text())
    if (
        json.loads((root / "rank_engine/completed.json").read_text())["status"]
        != "complete"
    ):
        raise ValueError("Clean C2 decision-rank experiment is incomplete")
    if not fusion["all_engine_branches_available"]:
        raise ValueError(
            "Both engine dimensions must be evaluated before sealing"
        )
    matching = json.loads((root / "matching/completed.json").read_text())
    expected = [
        root / f"encoder_engine_{dimension}/checkpoint.pt"
        for dimension in (512, 1024)
    ]
    actual = matching["source_manifest"]["signature"]["checkpoint_order"]
    if [path.resolve() for path in expected] != [
        pathlib.Path(path).resolve() for path in actual
    ]:
        raise ValueError(
            "Primary matching did not use the two distinct engine branches"
        )
    for name, digest in matching["source_manifest"]["signature"][
        "checkpoints"
    ].items():
        if artifacts.file_digest(pathlib.Path(name)) != digest:
            raise ValueError(
                f"Stage-one checkpoint changed after matching: {name}"
            )
    if fusion["final_holdout_evaluated"] or matching["final_holdout_evaluated"]:
        raise ValueError(
            "Final outcomes were accessed before configuration sealing"
        )
    for name, source in fusion["source_manifest"].items():
        if (
            artifacts.file_digest(pathlib.Path(source["path"]))
            != source["sha256"]
        ):
            raise ValueError(f"Validation source changed after fusion: {name}")
    return fusion, matching


def verify_files(configuration: dict) -> None:
    """Rejects any changed model, source input or inference implementation."""
    for name, digest in configuration["files"].items():
        if artifacts.file_digest(pathlib.Path(name)) != digest:
            raise ValueError(f"Sealed experiment artifact changed: {name}")


def recipe_for_validation(root: pathlib.Path) -> dict:
    """Builds a replay recipe from completed validation decisions only."""
    fusion, matching = check_complete(root)
    aggregations = {}
    for name in REQUIRED_BRANCHES:
        aggregations[name] = json.loads(
            (root / f"encoder_{name}/completed.json").read_text()
        )["aggregation"]
    return {
        "weights": fusion["weights"],
        "aggregations": aggregations,
        "matching": {
            "branch": matching["selected_branch"],
            "blend": matching["selection"]["blend"],
            "stage_weight_b": matching["source_manifest"]["stage_weight_b"],
        },
        "fisher_baseline_aggregation": "centroid",
    }


def create_lock(root: pathlib.Path) -> dict:
    """Writes a one-time lock after the inference implementation has replayed."""
    destination = root / "locked_configuration.json"
    if destination.exists():
        configuration = json.loads(destination.read_text())
        verify_files(configuration)
        return configuration
    if (root / "final_results.json").exists():
        raise ValueError("Final results exist without the required prior lock")
    recipe = recipe_for_validation(root)
    replay = json.loads((root / "validation_replay.json").read_text())
    if replay["status"] != "passed" or replay["recipe"] != recipe:
        raise ValueError(
            "Validation replay has not passed for this exact recipe"
        )
    verify_files(replay)
    configuration = {
        "schema": "open_set_v2_locked_v1",
        "protocol": "600 candidates; 10 refs; nested 1/2/3/5 mixed-color queries",
        **recipe,
        "validation": replay["proposed"],
        "files": {
            **replay["files"],
            str(root / "validation_replay.json"): artifacts.file_digest(
                root / "validation_replay.json"
            ),
        },
        "final_evaluated_before_lock": False,
    }
    with destination.open("x", encoding="utf-8") as output:
        output.write(json.dumps(configuration, indent=2) + "\n")
    return configuration


def main() -> None:
    """Locks only complete experiment configurations, leaving final sealed."""
    configuration = create_lock(dataset.ROOT)
    print(
        json.dumps(
            {
                "status": "sealed",
                "weights": configuration["weights"],
                "validation": configuration["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
