"""Evaluation helpers shared by the experiment scripts that score fused members on questions.

Used by experiments/opponent_side, experiments/opponent_specific and experiments/fight_features.
Every helper reproduces the computation those scripts performed inline (same random seeds and the
same order of random draws), so the recorded results*/ files are unchanged.
"""
import dataclasses
import pathlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

import blend_eval
import rank_lib as lib
import residual_shift
import transformer_rank as tr

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORE_MATRIX = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
N_BOOTSTRAP = 5000
N_FOLDS = 5


def clipped_log(probs: np.ndarray) -> np.ndarray:
    """Log-probabilities with the probabilities clipped to [1e-6, 1]."""
    return np.log(np.clip(probs, 1e-6, 1))


def competition_score(probs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question competition score (1 exact, exp(-1) adjacent) of the expected-score decision."""
    gap = np.abs((probs @ SCORE_MATRIX).argmax(1) - truth)
    return np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))


def checkpoint_probs(paths: Iterable[pathlib.Path], tokens: np.ndarray, valid: np.ndarray,
                     device: torch.device,
                     predict_valid: np.ndarray | None = None) -> list[np.ndarray]:
    """Probabilities of each RankTransformer checkpoint in `paths` (one array per checkpoint).

    Tokens are standardised with the checkpoint statistics where `valid`; `predict_valid`
    (default `valid`) is the attention mask passed to the model.
    """
    probs = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hyper = ckpt["args"]
        model = tr.RankTransformer(tokens.shape[2], tokens.shape[1], hyper["d_model"],
                                   hyper["heads"], hyper["ffn"], hyper["layers"],
                                   hyper["dropout"], hyper.get("head", "clm")).to(device)
        model.load_state_dict(ckpt["state"])
        scaled = np.where(valid[..., None], (tokens - ckpt["mean"]) / ckpt["std"], 0)
        mask = valid if predict_valid is None else predict_valid
        probs.append(tr.predict(model, scaled.astype(np.float32), mask, device))
    return probs


def bootstrap_interval(diff: np.ndarray, units: np.ndarray,
                       rng: np.random.Generator) -> list[float]:
    """[mean of diff, 2.5%, 97.5%] where the interval resamples `units` 5,000 times."""
    boots = [units[rng.integers(0, len(units), len(units))].mean() for _ in range(N_BOOTSTRAP)]
    return [float(diff.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))]


def format_interval(interval: Sequence[float]) -> str:
    """'+mean  95% [+lo, +hi]' as printed by the evaluation scripts."""
    return f"{interval[0]:+.4f}  95% [{interval[1]:+.4f}, {interval[2]:+.4f}]"


# ---- untouched 400-player holdout

@dataclasses.dataclass
class Holdout:
    """Holdout questions: ids, true rank index, and the game rows of each question."""
    question_ids: list[str]
    truth: np.ndarray
    games_of: dict[str, list[int]]


def load_holdout(question_of_game: np.ndarray) -> Holdout:
    """Labels of the holdout questions; `question_of_game` is feats["player_id"] per game row."""
    labels = pd.read_csv(ROOT / "features/holdout/tests/labels.csv")
    truth = np.array([lib.LEVELS.index(r) for r in labels["rank"]])
    games_of: dict[str, list[int]] = {}
    for i, question in enumerate(question_of_game):
        games_of.setdefault(str(question), []).append(i)
    return Holdout(labels["question_id"].tolist(), truth, games_of)


def production_members(game_keys: np.ndarray) -> dict[str, np.ndarray]:
    """Production d96_h and resnet8c_h holdout probabilities, rows in the order of game_keys."""
    dump = np.load(ROOT / "features/holdout/probs_final.npz", allow_pickle=True)
    order = {k: i for i, k in enumerate(dump["game_key"])}
    rows = np.array([order[k] for k in game_keys])
    return {"d96_h": dump["d96_h"][rows], "resnet8c_h": dump["resnet8c_h"][rows]}


def validation_logs(models: Sequence[str], val_files: dict[str, tuple[pathlib.Path, str]]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[M, G, 10] clipped log-probabilities of the members on the validation cohort, y, player."""
    logs, y, player = [], None, None
    for model in models:
        path, key = val_files[model]
        val = np.load(path, allow_pickle=True)
        logs.append(clipped_log(val[key]))
        y, player = val["y"], val["player"]
    assert y is not None and player is not None
    return np.stack(logs), y, player


def validation_offsets(fused: np.ndarray, player: np.ndarray) -> np.ndarray:
    """Per-rank offsets fitted on 10 simulated 5-20 game questions per validation player
    (seed 0)."""
    rng = np.random.default_rng(0)
    questions = []
    for _ in range(10):
        for pid in np.unique(player):
            rows = np.flatnonzero(player == pid)
            size = min(int(rng.integers(5, 21)), len(rows))
            questions.append(lib.pool(fused[rng.choice(rows, size=size, replace=False)], "logmean"))
    return lib.fit_class_offsets(np.stack(questions))


def holdout_scores(models: Sequence[str], val_files: dict[str, tuple[pathlib.Path, str]],
                   members: dict[str, np.ndarray], holdout: Holdout,
                   fixed_w: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-question holdout scores of the fusion of `models`, and the fusion weights.

    Weights (unless fixed) and per-rank offsets are fitted on the validation cohort only;
    pooling = log-mean; expected-score decode.
    """
    val_stack, val_y, val_player = validation_logs(models, val_files)
    weights = (np.asarray(fixed_w, float) if fixed_w is not None
               else blend_eval.fit_weights(val_stack, val_y))
    offsets = validation_offsets(blend_eval.fuse(val_stack, weights), val_player)
    fused = blend_eval.fuse(np.stack([clipped_log(members[m]) for m in models]), weights)
    qprob = np.stack([lib.pool(fused[holdout.games_of[q]], "logmean")
                      for q in holdout.question_ids])
    return competition_score(lib.apply_class_offsets(qprob, offsets), holdout.truth), weights


# ---- simulated questions on the validation cohort

@dataclasses.dataclass
class ValidationQuestions:
    """The strength_axis validation questions (20 per player) with 5-fold player-disjoint folds."""
    games: list[np.ndarray]
    label: np.ndarray
    player: np.ndarray
    fold: np.ndarray
    players: np.ndarray
    fold_of: dict


def validation_questions(y: np.ndarray, player: np.ndarray) -> ValidationQuestions:
    """residual_shift.validation_questions plus the player folds (permutation seed 0)."""
    games, label = residual_shift.validation_questions(player, y)
    players = np.array(sorted(set(player)))
    fold_of = dict(zip(np.random.default_rng(0).permutation(players),
                       np.arange(len(players)) % N_FOLDS))
    q_player = np.array([player[g[0]] for g in games])
    fold = np.array([fold_of[p] for p in q_player])
    return ValidationQuestions(games, label, q_player, fold, players, fold_of)


def cv_question_scores(stack: np.ndarray, y: np.ndarray, player: np.ndarray,
                       questions: ValidationQuestions) -> np.ndarray:
    """Question scores with fusion weights and per-rank offsets fitted on 4 player folds and
    scored on the 5th. stack: [M, G, 10] member log-probabilities."""
    out = np.zeros(len(questions.games))
    for k in range(N_FOLDS):
        train_games = np.isin(player, [p for p in questions.players if questions.fold_of[p] != k])
        fused = blend_eval.fuse(stack, blend_eval.fit_weights(stack[:, train_games],
                                                              y[train_games]))
        qprob = np.stack([lib.pool(fused[g], "logmean") for g in questions.games])
        offsets = lib.fit_class_offsets(qprob[questions.fold != k])
        test = questions.fold == k
        out[test] = competition_score(lib.apply_class_offsets(qprob[test], offsets),
                                      questions.label[test])
    return out


def player_bootstrap(diff: np.ndarray, questions: ValidationQuestions,
                     rng: np.random.Generator) -> list[float]:
    """bootstrap_interval of per-question differences, resampling players, not questions."""
    per_player = np.array([diff[questions.player == p].mean() for p in questions.players])
    return bootstrap_interval(diff, per_player, rng)
