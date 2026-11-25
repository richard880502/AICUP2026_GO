# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Compares NLL and official-reward objectives for C1 residual calibration."""

import json
import pathlib

import numpy as np
import torch
from torch.nn import functional

import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.reward_alignment import (
    evaluate as fold_evaluate,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")
REWARD_WEIGHT = 0.2
EPOCHS = 300
LEARNING_RATE = 0.02
L2 = 0.01


def reward_matrix() -> np.ndarray:
    """Returns the official exact/adjacent/other reward matrix."""
    identity = np.eye(10)
    adjacent = np.eye(10, k=1) + np.eye(10, k=-1)
    return identity + np.exp(-1.0) * adjacent


def fit_utility_residual(
    data: dict[str, np.ndarray], weights: np.ndarray
) -> residual_shift.ShiftResidual:
    """Fits the unchanged C1 shift head with its fixed mixed objective."""
    logits, features, labels = calibration_inputs(data, weights)
    residual = residual_shift.ShiftResidual()
    residual.mu = features.mean(0)
    residual.sd = features.std(0) + 1e-6
    residual.module = fit_mixed_head(
        logits,
        (features - residual.mu) / residual.sd,
        labels,
    )
    return residual


def calibration_inputs(
    data: dict[str, np.ndarray], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the fixed C1 training questions and their frozen inputs."""
    logs = np.stack([np.log(data[name].clip(1e-6, 1)) for name in MEMBERS])
    fused = blend_eval.fuse(logs, weights)
    questions, labels = residual_shift.validation_questions(
        data["player"], data["y"]
    )
    logits, features = residual_shift.question_inputs(
        questions,
        fused,
        data["d96"],
        data["opp_scratch"],
        data["final"],
    )
    return logits, features, labels


def fit_mixed_head(
    logits: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> residual_shift.ShiftModule:
    """Optimizes the unchanged shift head with fixed mixed-loss weights."""
    module = residual_shift.ShiftModule(features.shape[1])
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    rewards = torch.as_tensor(reward_matrix(), dtype=torch.float32)
    target_rewards = rewards[:, labels_tensor].T
    optimizer = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        output = module(logits_tensor, features_tensor)
        probabilities = torch.softmax(output, dim=1)
        expected_reward = (probabilities * target_rewards).sum(1).mean()
        loss = functional.cross_entropy(output, labels_tensor)
        loss = loss + REWARD_WEIGHT * (1 - expected_reward)
        loss = loss + L2 * (module.f.weight ** 2).sum()
        loss.backward()
        optimizer.step()
    return module


def official_scores(
    probabilities: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Returns one official score per held-out query."""
    prediction = rank_lib.decode(probabilities)
    gap = np.abs(prediction - labels)
    return np.where(gap == 0, 1.0, np.where(gap == 1, np.exp(-1.0), 0.0))


def evaluate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs the fixed player-disjoint folds for both calibration objectives."""
    folds = fold_evaluate.game_folds(data)
    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("nll", "mixed", "y", "player", "count")
    }
    for fold in range(5):
        fitting = {key: value[folds != fold] for key, value in data.items()}
        testing = {key: value[folds == fold] for key, value in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("C1 calibration players overlap across folds")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        model = calibration.fit(fitting, MEMBERS)
        result["nll"].append(calibration.predict(model, testing, questions))
        utility_residual = fit_utility_residual(fitting, model.weights)
        mixed_model = calibration.FittedModel(
            MEMBERS, model.weights, utility_residual
        )
        result["mixed"].append(
            calibration.predict(mixed_model, testing, questions)
        )
        result["y"].append(labels)
        result["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        result["count"].append(np.asarray([len(rows) for rows in questions]))
        print(f"Reward calibration fold {fold + 1}/5 complete", flush=True)
    return {key: np.concatenate(values) for key, values in result.items()}


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Reports paired official utility, accuracy, NLL, rank and size deltas."""
    labels = arrays["y"]
    scores = {
        key: official_scores(arrays[key], labels) for key in ("nll", "mixed")
    }
    difference = scores["mixed"] - scores["nll"]
    player_delta = np.asarray(
        [
            difference[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    interval = statistics.bootstrap_players(player_delta, confidence=0.975)
    baseline_utility = float(scores["nll"].mean())
    utility_delta = float(difference.mean())
    return {
        "scope": "exploratory original400-player calibration CV; not Public",
        "utility": {key: float(value.mean()) for key, value in scores.items()},
        "utility_delta": utility_delta,
        "paired_player_97_5ci": interval,
        "exact": {
            key: float((rank_lib.decode(arrays[key]) == labels).mean())
            for key in ("nll", "mixed")
        },
        "question_nll": {
            key: float(-np.log(arrays[key][np.arange(len(labels)), labels]).mean())
            for key in ("nll", "mixed")
        },
        "by_rank_delta": [
            float(difference[labels == rank].mean()) for rank in range(10)
        ],
        "by_query_size_delta": {
            str(size): float(difference[arrays["count"] == size].mean())
            for size in np.unique(arrays["count"])
        },
        "changed": {
            "improved": int((difference > 0).sum()),
            "worsened": int((difference < 0).sum()),
        },
        "screen_pass": utility_delta >= 0.003 and interval["low"] > 0,
        "baseline_utility": baseline_utility,
    }


def main() -> None:
    """Runs the fixed screen and refuses to overwrite prior artifacts."""
    root = pathlib.Path.cwd()
    output = root / "production/reward_calibration_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing experiment output: {output}")
    output.mkdir(parents=True)
    with np.load(root / "production/rank_joint_20261108/prepared.npz") as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in ("y", "player", "final", *MEMBERS)
        }
    torch.set_num_threads(4)
    arrays = evaluate(data)
    report = summarize(arrays)
    np.savez_compressed(
        output / "predictions.npz", allow_pickle=False, **arrays
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
