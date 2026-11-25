# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Evaluates a player-safe nonlinear residual on frozen C1+C2 outputs."""

import json
import pathlib

import numpy as np
import torch

import blend_eval  # pylint: disable=import-error
import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.decision_alignment import (
    holdout3,
)
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.query_meta import model as meta_model
from competition2_rank_prediction.experiments.reward_alignment import (
    evaluate as fold_evaluate,
)
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

MEMBERS = ("d96", "resnet8c", "opp_scratch")


def question_logits_features(
    fitted: calibration.FittedModel,
    data: dict[str, np.ndarray],
    questions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns C1-corrected question logits and the unchanged C1 summaries."""
    logs = np.stack([np.log(data[name].clip(1e-6, 1)) for name in MEMBERS])
    fused = blend_eval.fuse(logs, fitted.weights)
    logits, features = residual_shift.question_inputs(
        questions,
        fused,
        data["d96"],
        data["opp_scratch"],
        data["final"],
    )
    return fitted.residual.logits(logits, features), features


def _player_fold_map(
    players: np.ndarray, row_folds: np.ndarray
) -> dict[str, int]:
    """Assigns each player the single fold used by its game rows."""
    mapping = {}
    for player in np.unique(players):
        folds = np.unique(row_folds[players == player])
        if len(folds) != 1:
            raise ValueError("A player's games span multiple meta folds")
        mapping[str(player)] = int(folds[0])
    return mapping


def _oof_fold_examples(
    data: dict[str, np.ndarray], row_folds: np.ndarray, meta_fold: int
) -> dict[str, np.ndarray]:
    """Builds one held-player fold of C1+C2 OOF query inputs."""
    training = {
        key: values[row_folds != meta_fold] for key, values in data.items()
    }
    testing = {
        key: values[row_folds == meta_fold] for key, values in data.items()
    }
    if set(training["player"]) & set(testing["player"]):
        raise ValueError("Meta OOF calibration player overlap")
    questions, labels = residual_shift.validation_questions(
        testing["player"], testing["y"]
    )
    fitted = calibration.fit(training, MEMBERS)
    logits, features = question_logits_features(fitted, testing, questions)
    return {
        "features": features,
        "logits": logits,
        "labels": labels,
        "players": np.asarray(
            [testing["player"][rows[0]] for rows in questions]
        ),
        "folds": np.full(len(labels), meta_fold, dtype=np.int64),
    }


def oof_examples(
    root: pathlib.Path,
) -> tuple[meta_model.Examples, dict[str, int]]:
    """Builds five-fold OOF query logits under both upstream model folds."""
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("features", "logits", "labels", "players", "folds")
    }
    player_folds: dict[str, int] = {}
    for upstream_fold in (0, 1):
        data = holdout3.validation_data(root, upstream_fold)
        row_folds = fold_evaluate.game_folds(data)
        local_player_folds = _player_fold_map(data["player"], row_folds)
        if player_folds and local_player_folds != player_folds:
            raise ValueError("Player folds differ across upstream model folds")
        player_folds = local_player_folds
        for meta_fold in range(5):
            fold_examples = _oof_fold_examples(data, row_folds, meta_fold)
            for name, values in fold_examples.items():
                collected[name].append(values)
            print(
                f"OOF calibration upstream={upstream_fold} "
                f"meta_fold={meta_fold}",
                flush=True,
            )
    arrays = {
        name: np.concatenate(parts) for name, parts in collected.items()
    }
    examples = meta_model.Examples(
        features=np.concatenate((arrays["logits"], arrays["features"]), 1),
        logits=arrays["logits"],
        labels=arrays["labels"],
        players=arrays["players"],
        folds=arrays["folds"],
    )
    return examples, player_folds


def _predict_outer(
    fitted: calibration.FittedModel,
    fitted_meta: meta_model.FittedModel,
    testing: dict[str, np.ndarray],
    questions: list[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns baseline and meta-corrected probabilities for fixed queries."""
    baseline = calibration.predict(fitted, testing, questions)
    logits, features = question_logits_features(fitted, testing, questions)
    np.testing.assert_allclose(
        torch.softmax(torch.as_tensor(logits), dim=1).numpy(),
        baseline,
        atol=1e-6,
        rtol=1e-6,
    )
    candidate_logits = meta_model.predict_logits(
        fitted_meta,
        np.concatenate((logits, features), axis=1),
        logits,
        device,
    )
    candidate = torch.softmax(
        torch.as_tensor(candidate_logits, dtype=torch.float32), dim=1
    ).numpy()
    return baseline, candidate


def evaluate_outer_fold(
    root: pathlib.Path,
    outer_fold: int,
    question_rows: list[dict[str, str]],
    fitted_meta: meta_model.FittedModel,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Fits baseline C1 on validation and predicts one unseen player half."""
    training = holdout3.validation_data(root, outer_fold)
    testing, questions, labels = holdout3.test_data(
        root, outer_fold, question_rows
    )
    fitted = calibration.fit(training, MEMBERS)
    baseline, candidate = _predict_outer(
        fitted, fitted_meta, testing, questions, device
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "y": labels,
        "player": np.asarray(
            [row["player"] for row in question_rows if int(row["fold"]) == outer_fold]
        ),
        "fold": np.full(len(labels), outer_fold, dtype=np.int64),
    }


def summarize(arrays: dict[str, np.ndarray], fit_info: dict[str, float]) -> dict:
    """Reports paired player utility, rank deltas and the fixed screen gate."""
    scores = {
        name: statistics.player_scores(arrays[name], arrays["y"])
        for name in ("baseline", "candidate")
    }
    difference = scores["candidate"] - scores["baseline"]
    per_player = np.asarray(
        [
            difference[arrays["player"] == player].mean()
            for player in np.unique(arrays["player"])
        ]
    )
    interval = statistics.bootstrap_players(per_player, confidence=0.95)
    fold_deltas = {
        str(fold): float(difference[arrays["fold"] == fold].mean())
        for fold in (0, 1)
    }
    delta = float(difference.mean())
    return {
        "scope": "historical2000-player cross-fit; nonlinear query meta-calibration",
        "utility": {name: float(value.mean()) for name, value in scores.items()},
        "delta": delta,
        "paired_player_95ci": interval,
        "outer_fold_deltas": fold_deltas,
        "exact": {
            name: float((rank_lib.decode(arrays[name]) == arrays["y"]).mean())
            for name in scores
        },
        "rank_deltas": [
            float(
                difference[arrays["y"] == rank].mean()
            )
            for rank in range(10)
        ],
        "questions": int(len(arrays["y"])),
        "players": int(len(per_player)),
        "meta_fit": fit_info,
        "screen_pass": bool(
            delta >= 0.003
            and interval["low"] > 0
            and all(value > 0 for value in fold_deltas.values())
        ),
    }


def main() -> None:
    """Trains on nested OOF validation queries and evaluates holdout3 once."""
    root = pathlib.Path.cwd()
    output = root / "production/query_meta_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing query-meta output: {output}")
    question_rows = holdout3.read_labels(
        root
        / "runtime/competition2_research/features/holdout3/question_folds.csv"
    )
    holdout3.validate_exclusion(root, question_rows)
    examples, player_folds = oof_examples(root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    fitted_meta = meta_model.fit(
        examples, player_folds, device
    )
    outer = [
        evaluate_outer_fold(
            root, fold, question_rows, fitted_meta, device
        )
        for fold in (0, 1)
    ]
    arrays = {
        key: np.concatenate([result[key] for result in outer])
        for key in outer[0]
    }
    report = summarize(
        arrays,
        {
            "best_epoch": float(fitted_meta.best_epoch),
            "valid_nll": fitted_meta.validation_nll,
        },
    )
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz", allow_pickle=False, **arrays
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
