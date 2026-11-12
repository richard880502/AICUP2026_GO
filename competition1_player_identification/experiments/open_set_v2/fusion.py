"""Selects score fusion on style validation while keeping final holdout sealed.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.fusion

Missing engine branches are recorded, not silently treated as evaluated. This
is an exploratory validation result until every planned ablation is finished.
"""

import hashlib
import json
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)

WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def standardize(scores: np.ndarray) -> np.ndarray:
    """Standardizes across candidates without labels or cross-question fitting."""
    return (
        (scores - scores.mean(-1, keepdims=True))
        / scores.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def score_value(scores: np.ndarray) -> float:
    """Computes exactly the contest utility without bootstrap overhead."""
    ranks = evaluation.correct_ranks(scores)
    return float(
        np.where(
            ranks <= 5,
            np.exp(-(ranks - 1)),
            0.0,
        ).mean()
    )


def load_sources(root: pathlib.Path) -> tuple[dict, dict]:
    """Loads only existing validation scores with explicit source-file hashes."""
    sources = {}
    manifest = {}
    paths = {
        "prefix": root / "prefix_validation_scores.npy",
        "rank_prior": root / "rank_prior/validation_scores.npy",
        "rank_engine": root / "rank_engine/validation_scores.npy",
    }
    matching_report = root / "matching/completed.json"
    if matching_report.exists():
        matching = json.loads(matching_report.read_text())
        paths["engine_ensemble"] = (
            root / "matching/stage1_validation_scores.npy"
        )
        if matching["reranker_adopted"]:
            paths["learned_pair"] = (
                root / "matching/reranked_validation_scores.npy"
            )
    for kind in ("opening", "sequence", "engine_512", "engine_1024"):
        directory = root / f"encoder_{kind}"
        if (directory / "completed.json").exists():
            selection = json.loads((directory / "completed.json").read_text())
            path = directory / "validation_scores.npz"
            with np.load(path) as arrays:
                sources[kind] = standardize(arrays[selection["aggregation"]])
            manifest[kind] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "aggregation": selection["aggregation"],
            }
    for name, path in paths.items():
        if path.exists():
            sources[name] = standardize(np.load(path))
            manifest[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return sources, manifest


def select_fusion(sources: dict) -> tuple[dict, np.ndarray, list]:
    """Runs a fixed two-pass forward search with one global branch weight."""
    primary = max(
        (score_value(value), name) for name, value in sources.items()
    )[1]
    selected = {primary: 1.0}
    current = sources[primary].copy()
    trials = []
    for _ in range(2):
        for name, value in sorted(sources.items()):
            if name in selected:
                continue
            candidates = [
                (score_value(current + weight * value), weight)
                for weight in WEIGHTS
            ]
            score, weight = max(
                candidates, key=lambda item: (item[0], -item[1])
            )
            trials.append(
                {"branch": name, "weight": weight, "validation_score": score}
            )
            if weight:
                selected[name] = weight
                current += weight * value
    return selected, current, trials


def main() -> None:
    """Records selected weights and validation-only paired uncertainty."""
    root = dataset.ROOT
    sources, manifest = load_sources(root)
    if not sources:
        raise ValueError("No completed validation branches are available")
    weights, scores, trials = select_fusion(sources)
    np.save(root / "fusion_validation_scores.npy", scores)
    baseline = "prefix" if "prefix" in sources else next(iter(sources))
    artifacts.write_report(
        root / "fusion_validation.json",
        {
            "status": "exploratory_validation_only",
            "weights": weights,
            "source_manifest": manifest,
            "trials": trials,
            "validation": evaluation.summarize(scores),
            "paired_vs_prefix": evaluation.paired_delta(
                scores, sources[baseline]
            ),
            "all_engine_branches_available": all(
                name in sources for name in ("engine_512", "engine_1024")
            ),
            "hard_candidate_pruning": False,
            "final_holdout_evaluated": False,
        },
        emit=True,
    )


if __name__ == "__main__":
    main()
