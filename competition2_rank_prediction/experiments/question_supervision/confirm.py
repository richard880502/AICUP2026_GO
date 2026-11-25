# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Checks question-supervised encoders inside frozen-member C1+C2.

Example: python -m competition2_rank_prediction.experiments.question_supervision.confirm
Uses original validation games only, not official test predictions or labels.
"""

import json

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.mining_submission import (
    joint_features,
)
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.reward_alignment import evaluate
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")


def prepare(args) -> dict[str, np.ndarray]:
    """Checks original validation exclusion and caches aligned A/B predictions."""
    cache = (
        args.research.parents[1] / "production/rank_joint_20261108/prepared.npz"
    )
    with np.load(cache) as payload:
        data = {
            name: np.asarray(payload[f"val_{name}"])
            for name in ("y", "player", "game_key", "final", *MEMBERS)
        }
    with np.load(args.output / "splits.npz") as split:
        excluded = set(split["train"]) | set(split["inner"])
        if excluded & set(data["player"]):
            raise ValueError(
                "Original validation overlaps neural training/selection players"
            )
    features = rank_lib.load_features(args.research / "features/feat96/val")
    lookup = {str(key): row for row, key in enumerate(features["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
    tokens = transformer_rank.decision_tokens(features["tokens"][rows])
    masks = features["masks"][rows]
    for arm in ("A", "B"):
        predictions = []
        for seed in range(3):
            predictions.append(
                joint_features.checkpoint_prediction(
                    args.output / f"seed_{seed}/{arm}.pt",
                    tokens,
                    masks,
                    torch.device(args.device),
                )
            )
        data[arm] = np.mean(predictions, axis=0)
    np.savez_compressed(
        args.output / "confirmation_inputs.npz", allow_pickle=False, **data
    )
    return data


def cross_validate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fits fusion/C1 only on the other four original validation player folds."""
    folds = evaluate.game_folds(data)
    output: dict[str, list[np.ndarray]] = {"y": [], "player": []}
    for fold in range(5):
        fitting = {key: value[folds != fold] for key, value in data.items()}
        testing = {key: value[folds == fold] for key, value in data.items()}
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        output["y"].append(labels)
        output["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        for method in ("original", "A", "B"):
            for subset, mask in (
                (fitting, folds != fold),
                (testing, folds == fold),
            ):
                subset["d96"] = (
                    data["d96"][mask]
                    if method == "original"
                    else data[method][mask]
                )
            model = calibration.fit(fitting, MEMBERS)
            output.setdefault(method, []).append(
                calibration.predict(model, testing, questions)
            )
        print(f"Full C1+C2 confirmation fold {fold + 1}/5", flush=True)
    return {key: np.concatenate(values) for key, values in output.items()}


def main() -> None:
    """Reports production reference and matched-control effects separately."""
    args = common.parse_arguments(
        __doc__ or "", "question_supervision_20261109", True
    )
    screening = json.loads((args.output / "report.json").read_text())
    if not screening["screen_pass"]:
        raise RuntimeError(
            "Own-branch pilot did not pass the preregistered gate"
        )
    if (args.output / "confirmation_report.json").exists():
        raise FileExistsError("Refusing to overwrite completed confirmation")
    torch.set_num_threads(4)
    results = cross_validate(prepare(args))
    rewards = {
        method: statistics.player_scores(results[method], results["y"])
        for method in ("original", "A", "B")
    }
    intervals = {}
    for reference in ("original", "A"):
        difference = rewards["B"] - rewards[reference]
        clustered = np.asarray(
            [
                difference[results["player"] == player].mean()
                for player in np.unique(results["player"])
            ]
        )
        intervals[f"B_minus_{reference}"] = statistics.bootstrap_players(
            clustered, confidence=0.975
        )
    report = {
        "scope": "exploratory original validation calibration-CV, not fresh confirmation or Public",
        "utility": {
            method: float(values.mean()) for method, values in rewards.items()
        },
        "paired_player_97_5ci": intervals,
        "original_replay_matches": abs(
            float(rewards["original"].mean()) - 0.7106057000201695
        )
        < 1e-12,
        "official_submission_created": False,
    }
    if not report["original_replay_matches"]:
        raise RuntimeError("Original C1+C2 calibration baseline did not replay")
    np.savez_compressed(
        args.output / "confirmation_predictions.npz",
        allow_pickle=False,
        **results,
    )
    (args.output / "confirmation_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
