# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Fits fold-safe geometric fusion and C1 for a mining submission candidate.

The original own/opponent inputs to C1 remain fixed for both comparisons.
"""

import dataclasses
from typing import TypedDict

import numpy as np

import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.semi_hard_mining import statistics


class CVReport(TypedDict):
    """Types the player-safe calibration comparison record."""

    folds: list[dict[str, object]]
    unique_players: int
    questions: int
    utility: dict[str, float]
    paired_player_95ci: dict[str, float]
    exact: dict[str, float]
    scope: str


@dataclasses.dataclass
class FittedModel:
    """Stores fusion and question correction fitted on validation players.

    Attributes:
        members: Named geometric-fusion members in fixed order.
        weights: Validation-fitted nonnegative member weights.
        residual: Fitted C1 shift correction.
    """

    members: tuple[str, ...]
    weights: np.ndarray
    residual: residual_shift.ShiftResidual


def fit(data: dict[str, np.ndarray], members: tuple[str, ...]) -> FittedModel:
    """Fits weights and C1 on the supplied training-validation players."""
    logs = np.stack([np.log(data[name].clip(1e-6, 1)) for name in members])
    weights = blend_eval.fit_weights(logs, data["y"])
    residual = residual_shift.fit_on_validation(
        logs,
        weights,
        (data["y"], data["player"]),
        (
            np.exp(logs[members.index("d96")]),
            np.exp(logs[members.index("opp_scratch")]),
        ),
        data["final"],
    )
    return FittedModel(members, weights, residual)


def predict(
    model: FittedModel,
    data: dict[str, np.ndarray],
    questions: list[np.ndarray],
) -> np.ndarray:
    """Returns corrected question probabilities with no test-label access."""
    logs = np.stack(
        [np.log(data[name].clip(1e-6, 1)) for name in model.members]
    )
    fused = blend_eval.fuse(logs, model.weights)
    logits, features = residual_shift.question_inputs(
        questions,
        fused,
        data["d96"],
        data["opp_scratch"],
        data["final"],
    )
    return model.residual.predict_proba(logits, features)


def _game_folds(data: dict[str, np.ndarray]) -> np.ndarray:
    generator = np.random.default_rng(20261108)
    folds = np.full(len(data["y"]), -1, dtype=int)
    for rank in range(10):
        players = np.unique(data["player"][data["y"] == rank])
        for index, player in enumerate(generator.permutation(players)):
            folds[data["player"] == player] = index % 5
    if (folds < 0).any():
        raise ValueError("Validation fold assignment is incomplete")
    return folds


def _scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    gap = np.abs(rank_lib.decode(probabilities) - labels)
    return np.where(gap == 0, 1, np.where(gap == 1, np.exp(-1), 0))


def cross_validate(
    data: dict[str, np.ndarray],
    members: tuple[str, ...],
) -> tuple[CVReport, dict[str, np.ndarray]]:
    """Evaluates repeated shared queries with player-disjoint calibration."""
    folds = _game_folds(data)
    arrays: dict[str, list[np.ndarray]] = {
        "baseline": [],
        "candidate": [],
        "y": [],
        "player": [],
    }
    records = []
    for fold in range(5):
        training = {key: value[folds != fold] for key, value in data.items()}
        testing = {key: value[folds == fold] for key, value in data.items()}
        if set(training["player"]) & set(testing["player"]):
            raise ValueError("Calibration train/evaluation player overlap")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        arrays["y"].append(labels)
        arrays["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        for method, selected in (
            ("baseline", members[:3]),
            ("candidate", members),
        ):
            model = fit(training, selected)
            probabilities = predict(model, testing, questions)
            arrays[method].append(probabilities)
            records.append(
                {
                    "fold": fold,
                    "method": method,
                    "weights": model.weights.tolist(),
                    "train_players": len(set(training["player"])),
                    "test_players": len(set(testing["player"])),
                }
            )
        print(f"Calibration fold {fold + 1}/5 complete", flush=True)
    aligned = {key: np.concatenate(parts) for key, parts in arrays.items()}
    return _report(aligned, records), aligned


def _report(arrays: dict[str, np.ndarray], records: list[dict]) -> CVReport:
    labels = arrays["y"]
    scores = {
        method: _scores(arrays[method], labels)
        for method in ("baseline", "candidate")
    }
    delta = scores["candidate"] - scores["baseline"]
    per_player = np.asarray(
        [
            delta[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    return {
        "folds": records,
        "unique_players": len(per_player),
        "questions": len(labels),
        "utility": {
            method: float(values.mean()) for method, values in scores.items()
        },
        "paired_player_95ci": statistics.bootstrap_players(per_player),
        "exact": {
            method: float((rank_lib.decode(arrays[method]) == labels).mean())
            for method in scores
        },
        "scope": "exploratory calibration CV; repeated queries aggregated within player",
    }
