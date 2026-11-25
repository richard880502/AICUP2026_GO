"""Fits the preregistered train-only PCA/logistic probe on frozen HumanSL features.

Example: python -m competition2_rank_prediction.experiments.human_trunk.classifier
Requires complete train/validation extraction; no model selection on validation.
"""

import json
import pathlib
import pickle
import sys

import numpy as np
import sklearn.decomposition  # type: ignore[import-untyped]
import sklearn.linear_model  # type: ignore[import-untyped]
import sklearn.pipeline  # type: ignore[import-untyped]
import sklearn.preprocessing  # type: ignore[import-untyped]

from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import rank_lib  # pylint: disable=wrong-import-position,import-error

SEED = 20261109


def phase_statistics(projected: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Builds phase1/2/4 mean/std summaries, excluding padding from moments."""
    descriptors: list[np.ndarray] = []
    for phases in (1, 2, 4):
        for indices in np.array_split(np.arange(projected.shape[1]), phases):
            valid = masks[:, indices, None]
            count = np.maximum(valid.sum(axis=1), 1)
            values = projected[:, indices]
            mean = (values * valid).sum(axis=1) / count
            variance = (((values - mean[:, None]) ** 2) * valid).sum(
                axis=1
            ) / count
            descriptors.extend((mean, np.sqrt(variance)))
    return np.concatenate(descriptors, axis=1)


def game_features(
    payload: dict[str, np.ndarray],
    scaler: sklearn.preprocessing.StandardScaler,
    projection: sklearn.decomposition.PCA,
) -> np.ndarray:
    """Applies frozen training transforms and masked phase statistics in chunks."""
    pieces = []
    for start in range(0, len(payload["tokens"]), 512):
        tokens = payload["tokens"][start : start + 512]
        masks = payload["masks"][start : start + 512]
        projected = np.zeros((*masks.shape, 64), dtype=np.float32)
        if masks.any():
            projected[masks] = projection.transform(
                scaler.transform(tokens[masks].astype(np.float32))
            )
        summaries = phase_statistics(projected, masks)
        coverage = masks.mean(axis=1, keepdims=True)
        pass_fraction = (tokens[:, :, -1] * masks).sum(
            axis=1, keepdims=True
        ) / np.maximum(masks.sum(axis=1, keepdims=True), 1)
        pieces.append(
            np.concatenate((summaries, coverage, pass_fraction), axis=1)
        )
    return np.concatenate(pieces)


def fit_projection(training: dict[str, np.ndarray]):
    """Uses valid training positions only, including a fixed bounded PCA sample."""
    valid = training["tokens"][training["masks"]].astype(np.float32)
    if len(valid) < 64:
        raise ValueError("Insufficient valid training positions for PCA64")
    scaler = sklearn.preprocessing.StandardScaler().fit(valid)
    generator = np.random.default_rng(SEED)
    rows = generator.choice(
        len(valid), size=min(32768, len(valid)), replace=False
    )
    projection = sklearn.decomposition.PCA(
        n_components=64, svd_solver="randomized", random_state=SEED
    )
    projection.fit(scaler.transform(valid[rows]))
    return scaler, projection


def main() -> None:
    """Fits exactly one fixed probe and exports aligned validation probabilities."""
    args = common.parse_arguments(__doc__ or "", "human_trunk_20261109")
    if not (args.output / "EXTRACTION_COMPLETE").exists():
        raise RuntimeError(
            "Complete extraction is required before probe fitting"
        )
    if (args.output / "probe_model.pkl").exists():
        raise FileExistsError("Refusing to overwrite a fitted probe")
    training = rank_lib.load_features(args.output / "features/train")
    validation = rank_lib.load_features(args.output / "features/val")
    if set(training["player_id"]) & set(validation["player_id"]):
        raise ValueError("Probe train/validation player overlap")
    if len(training["tokens"]) != 15211 or len(validation["tokens"]) != 8000:
        raise ValueError("Unexpected original train/validation game coverage")
    scaler, projection = fit_projection(training)
    train_features = game_features(training, scaler, projection)
    val_features = game_features(validation, scaler, projection)
    valid = np.asarray(training["masks"].any(axis=1))
    model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED
        ),
    )
    model.fit(train_features[valid], rank_lib.rank_ids(training["rank"])[valid])
    if not np.array_equal(model.classes_, np.arange(10)):
        raise ValueError("Probe classifier class order does not match rank IDs")
    probabilities = model.predict_proba(val_features)
    missing = np.asarray(~validation["masks"].any(axis=1))
    probabilities[missing] = 0.1
    with (args.output / "probe_model.pkl").open("wb") as handle:
        pickle.dump(
            {"scaler": scaler, "projection": projection, "model": model}, handle
        )
    np.savez_compressed(
        args.output / "probe_validation.npz",
        probabilities=probabilities,
        game_key=validation["game_key"],
        player=validation["player_id"],
        y=rank_lib.rank_ids(validation["rank"]),
    )
    report = {
        "training_games": int(valid.sum()),
        "validation_games": len(missing),
        "invalid_training_games": int((~valid).sum()),
        "invalid_validation_games": int(missing.sum()),
        "feature_dimension": train_features.shape[1],
        "pca_explained_variance": float(
            projection.explained_variance_ratio_.sum()
        ),
        "seed": SEED,
        "C": 0.1,
        "sklearn_version": sklearn.__version__,
        "scope": "fixed train-only probe; conditional C1+C2 evaluation is separate",
    }
    (args.output / "probe_fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
