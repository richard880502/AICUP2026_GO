"""Shared pieces: feature loading, game profiles, question pooling and the competition metric."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "c1"))

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
ADJACENT = float(np.exp(-1.0))
PROFILE_SCALES = (1, 2, 4, 8)


def load_features(directory: pathlib.Path) -> dict[str, np.ndarray]:
    """Concatenates every feat_*.npz shard in a directory."""
    shards = sorted(directory.glob("feat_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no feature shards in {directory}")
    # allow_pickle: shards written by the first extraction run store string columns as object arrays.
    parts = [np.load(path, allow_pickle=True) for path in shards]
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("tokens", "masks", "game_key", "player_id", "rank")}
    for key in ("game_key", "player_id", "rank"):
        out[key] = out[key].astype(str)
    return out


def rank_ids(ranks: np.ndarray) -> np.ndarray:
    return np.asarray([LEVELS.index(r) for r in ranks], dtype=np.int64)


def channel_names(profiles: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{p}_{m}" for p in profiles for m in ("surprisal", "log_policy_rank", "policy_entropy"))


def game_features(tokens: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per-game feature vector: competition 1 multi-scale phase profile + likelihood summaries."""
    from evaluate_decision_features import ProfileConfig, make_feature_stat_profiles  # competition 1

    games, samples, models, per_model = tokens.shape
    flat = tokens.reshape(games, samples, models * per_model)
    names = tuple(f"ch{i}" for i in range(models * per_model))
    config = ProfileConfig(
        representation="multi_scale_stats", phase_scales=PROFILE_SCALES, include_phase_std=True,
        include_robust_stats=False, include_interactions=False,
    )
    profile = make_feature_stat_profiles(flat, masks, names, config)
    surprisal = tokens[:, :, 1:, 0]  # [G, S, 10 human profiles]
    valid = masks[..., None].astype(np.float32)
    count = np.maximum(valid.sum(1), 1.0)
    mean_surprisal = (surprisal * valid).sum(1) / count
    centered = mean_surprisal - mean_surprisal.mean(1, keepdims=True)
    log_post = -surprisal - np.log(np.exp(-surprisal).sum(2, keepdims=True))  # per-decision posterior
    mean_post = (np.exp(log_post) * valid).sum(1) / count
    extra = np.concatenate([centered, mean_post, count, mean_surprisal - tokens[:, :, 0, 0:1].mean(1)], axis=1)
    return np.concatenate([profile, extra], axis=1).astype(np.float32)


def competition_score(pred: np.ndarray, truth: np.ndarray) -> float:
    gap = np.abs(pred - truth)
    return float(np.mean(np.where(gap == 0, 1.0, np.where(gap == 1, ADJACENT, 0.0))))


def decode(prob: np.ndarray) -> np.ndarray:
    """Score-optimal class for each row: maximises p_c + e^-1 (p_{c-1} + p_{c+1})."""
    weight = np.eye(len(LEVELS)) + ADJACENT * (np.eye(len(LEVELS), k=1) + np.eye(len(LEVELS), k=-1))
    return (prob @ weight).argmax(1)


def pool(prob: np.ndarray, how: str = "logmean") -> np.ndarray:
    """Pools the per-game class probabilities of one question into one distribution."""
    if how == "mean":
        return prob.mean(0)
    log = np.log(np.clip(prob, 1e-6, 1.0)).mean(0)
    p = np.exp(log - log.max())
    return p / p.sum()


def fit_class_offsets(question_probs: np.ndarray, iters: int = 200) -> np.ndarray:
    """Log-offsets b so that, on questions whose true ranks are balanced, the predicted marginal becomes uniform.

    Learned on the validation cohort only: it corrects the models' systematic per-rank bias (e.g. predicting A far
    too often) without assuming anything about the test set's rank distribution.
    """
    b = np.zeros(question_probs.shape[1])
    for _ in range(iters):
        p = question_probs * np.exp(b)
        p /= p.sum(1, keepdims=True)
        b -= 0.5 * np.log(p.mean(0) * question_probs.shape[1] + 1e-9)
    return b


def apply_class_offsets(question_probs: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = question_probs * np.exp(b)
    return p / p.sum(1, keepdims=True)
