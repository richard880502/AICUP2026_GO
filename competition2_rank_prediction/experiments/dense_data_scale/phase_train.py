"""Fits matched fixed forests to mean/std versus richer phase summaries.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
This experimental auxiliary branch never trains on original validation players.
"""

import json
import pathlib
import pickle
import sys

import numpy as np
import sklearn.ensemble  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_evaluate,
)
from competition2_rank_prediction.experiments.dense_data_scale import (
    phase_features,
)

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error


def fit_arm(training: dict, validation: dict, rich: bool, directory) -> list:
    """Keeps players, weighting, forest recipe and seeds fixed between arms."""
    train_features = phase_features.summarize(
        training["tokens"], training["masks"], rich
    )
    val_features = phase_features.summarize(
        validation["tokens"], validation["masks"], rich
    )
    _, inverse, counts = np.unique(
        training["player_id"], return_inverse=True, return_counts=True
    )
    weights = 1 / counts[inverse]
    weights *= len(weights) / weights.sum()
    predictions = []
    for seed in range(3):
        model = sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=128,
            max_depth=20,
            min_samples_leaf=8,
            max_features=0.5,
            n_jobs=6,
            random_state=seed,
        )
        model.fit(
            train_features,
            rank_lib.rank_ids(training["rank"]),
            sample_weight=weights,
        )
        predictions.append(model.predict_proba(val_features))
        with (directory / f"rich{int(rich)}_seed{seed}.pkl").open(
            "wb"
        ) as handle:
            pickle.dump(model, handle)
        print(f"rich={rich} seed={seed} complete", flush=True)
    return predictions


def main() -> None:
    """Runs one fixed comparison, with no validation-driven parameter search."""
    root = pathlib.Path.cwd()
    directory = root / "production/phase_distribution_20261109"
    if directory.exists():
        raise FileExistsError(
            f"Refusing existing phase experiment: {directory}"
        )
    source = root / "runtime/competition2_research/features/feat96"
    training = rank_lib.load_features(source / "train")
    validation = rank_lib.load_features(source / "val")
    if set(training["player_id"]) & set(validation["player_id"]):
        raise ValueError("Training and validation players overlap")
    directory.mkdir(parents=True)
    predictions = {
        2: fit_arm(training, validation, False, directory),
        8: fit_arm(training, validation, True, directory),
    }
    metadata = {
        "player": validation["player_id"],
        "y": rank_lib.rank_ids(validation["rank"]),
    }
    result = coverage_evaluate.summarize(predictions, metadata)
    result["arms"] = {"2": "mean_std", "8": "quantiles_skew"}
    result["scope"] = "exploratory phase-distribution own branch, not Public"
    np.savez_compressed(
        directory / "predictions.npz",
        control=np.stack(predictions[2]),
        rich=np.stack(predictions[8]),
        game_key=validation["game_key"],
        player=metadata["player"],
        y=metadata["y"],
    )
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
