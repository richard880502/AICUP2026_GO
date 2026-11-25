# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Tests a soft expected-reward action objective for C1 calibration."""

import pathlib

import numpy as np
import torch
from torch.nn import functional

import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.decision_alignment import artifacts
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.reward_calibration import run as base

TEMPERATURE = 0.1
REWARD_WEIGHT = 0.2
L2 = 0.01
EPOCHS = 300
LEARNING_RATE = 0.02


def fit_decision_head(
    logits: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> residual_shift.ShiftModule:
    """Fits the fixed C1 shift head using a soft decoded-rank action."""
    module = residual_shift.ShiftModule(features.shape[1])
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    rewards = torch.as_tensor(base.reward_matrix(), dtype=torch.float32)
    true_rewards = rewards[:, labels_tensor].T
    optimizer = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        output = module(logits_tensor, features_tensor)
        loss = decision_loss(output, labels_tensor, rewards, true_rewards)
        loss = loss + L2 * (module.f.weight ** 2).sum()
        loss.backward()
        optimizer.step()
    return module


def decision_loss(
    output: torch.Tensor,
    labels: torch.Tensor,
    rewards: torch.Tensor,
    true_rewards: torch.Tensor,
) -> torch.Tensor:
    """Returns cross-entropy plus the soft decoded-action reward loss."""
    class_probabilities = torch.softmax(output, dim=1)
    action_values = class_probabilities @ rewards.T
    action_probabilities = torch.softmax(action_values / TEMPERATURE, dim=1)
    expected_reward = (action_probabilities * true_rewards).sum(1).mean()
    loss = functional.cross_entropy(output, labels)
    return loss + REWARD_WEIGHT * (1 - expected_reward)


def fit_decision_residual(
    data: dict[str, np.ndarray], weights: np.ndarray
) -> residual_shift.ShiftResidual:
    """Fits a standardized C1 residual with the fixed decision objective."""
    logits, features, labels = base.calibration_inputs(data, weights)
    residual = residual_shift.ShiftResidual()
    residual.mu = features.mean(0)
    residual.sd = features.std(0) + 1e-6
    standardized = (features - residual.mu) / residual.sd
    residual.module = fit_decision_head(logits, standardized, labels)
    return residual


def evaluate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluates the fixed decision objective on held-out player folds."""
    folds = base.fold_evaluate.game_folds(data)
    result: dict[str, list[np.ndarray]] = {
        name: [] for name in ("nll", "decision", "y", "player", "count")
    }
    for fold in range(5):
        fitting = {key: value[folds != fold] for key, value in data.items()}
        testing = {key: value[folds == fold] for key, value in data.items()}
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("C1 calibration players overlap across folds")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        model = calibration.fit(fitting, base.MEMBERS)
        result["nll"].append(
            calibration.predict(model, testing, questions)
        )
        decision_model = calibration.FittedModel(
            base.MEMBERS,
            model.weights,
            fit_decision_residual(fitting, model.weights),
        )
        result["decision"].append(
            calibration.predict(decision_model, testing, questions)
        )
        result["y"].append(labels)
        result["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        result["count"].append(np.asarray([len(rows) for rows in questions]))
        print(f"Decision alignment fold {fold + 1}/5 complete", flush=True)
    return {key: np.concatenate(values) for key, values in result.items()}


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Uses the shared fixed comparison report and names the tested arm."""
    renamed = dict(arrays)
    renamed["mixed"] = renamed.pop("decision")
    report = base.summarize(renamed)
    report["objective"] = "soft expected-reward action, temperature0.1"
    report["screen_pass"] = bool(report["screen_pass"])
    return report


def main() -> None:
    """Runs the fixed screen without overwriting other experiment outputs."""
    root = pathlib.Path.cwd()
    output = root / "production/decision_alignment_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing experiment output: {output}")
    with np.load(root / "production/rank_joint_20261108/prepared.npz") as cache:
        data = {
            name: np.asarray(cache[f"val_{name}"])
            for name in ("y", "player", "final", *base.MEMBERS)
        }
    torch.set_num_threads(4)
    arrays = evaluate(data)
    report = summarize(arrays)
    artifacts.save(output, arrays, report)


if __name__ == "__main__":
    main()
