"""Is the rank information already on one strength axis, or does the 128-d player vector carry more?

Methods (all fitted with player-disjoint out-of-fold questions from the validation cohort only):
  A  production best2 (d96_h + cosine ResNet, log-mean pooling, per-rank offsets,
     expected-score decode)
  B  strength score s + 9 hard cut points chosen to maximise the competition score
  C  strength score s (+ log k) + ordinal logistic calibration, expected-score decode
  D  player-mean 128-d vector (+ log k) + L2-regularised ordinal logistic classifier,
     expected-score decode
The untouched holdout is scored ONCE at the end with calibrators fitted on all validation players
(no holdout tuning).

Run from the project root: c1/.venv/bin/python experiments/strength_axis/run.py
"""
import dataclasses
import json
import pathlib
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
import blend_eval  # noqa: E402  pylint: disable=wrong-import-position,import-error
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error
import residual_shift  # noqa: E402  pylint: disable=wrong-import-position,import-error
import transformer_rank as tr  # noqa: E402  pylint: disable=wrong-import-position,import-error

OUT = ROOT / "experiments/strength_axis/results"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
W = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
METHODS = ["A production best2", "A2 dense Transformer only", "B s + 9 hard cut points",
           "C s + ordinal calibration", "D 128-d mean + ordinal (l2)"]
L2S = (1e-3, 1e-2, 1e-1)


def score_vec(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-question competition score: 1 for exact, exp(-1) for an adjacent rank, else 0."""
    gap = np.abs(pred - truth)
    return np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))


# -------------------------------------------------- per-game strength scores and vectors
def strength_outputs(tokens: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strength score of every d96_h seed ([seeds, G]) and the seed-0 pooled 128-d game vectors."""
    x = tr.decision_tokens(tokens)
    s_all, v0 = [], None
    for path in sorted((ROOT / "results").glob("d96_h_seed*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hyper = ck["args"]
        model = tr.RankTransformer(x.shape[2], x.shape[1], hyper["d_model"], hyper["heads"],
                                   hyper["ffn"], hyper["layers"], hyper["dropout"],
                                   hyper.get("head", "clm")).to(DEV).eval()
        model.load_state_dict(ck["state"])
        xs = np.where(masks[..., None], (x - ck["mean"]) / ck["std"], 0).astype(np.float32)
        scores, vectors = [], []
        with torch.no_grad():
            for i in range(0, len(xs), 1024):
                game = model.encode(torch.from_numpy(xs[i:i + 1024]).to(DEV),
                                    torch.from_numpy(masks[i:i + 1024]).to(DEV))
                scores.append(model.score(game).squeeze(-1).cpu().numpy())
                vectors.append(game.cpu().numpy())
        s_all.append(np.concatenate(scores))
        if v0 is None:
            v0 = np.concatenate(vectors)
    assert v0 is not None, "no d96_h checkpoints found"
    return np.stack(s_all), v0


def game_outputs(split: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Features of one split, per-seed strength scores [seeds, G] and seed-0 game vectors."""
    feats = lib.load_features(ROOT / f"features/feat96/{split}")
    s_all, v0 = strength_outputs(feats["tokens"], feats["masks"])
    return feats, s_all, v0


def member_logs(val_keys: np.ndarray,
                hold_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Production members' per-game log-probabilities on validation and holdout (method A)."""
    vp = {m: np.load(ROOT / f"results/val_probs_{m}.npz", allow_pickle=True)
          for m in ("d96_h", "resnet8c_h")}
    assert (vp["d96_h"]["game_key"] == val_keys).all()
    val_logs = np.stack([np.log(np.clip(vp["d96_h"]["transformer"], 1e-6, 1)),
                         np.log(np.clip(vp["resnet8c_h"]["resnet8c_h"], 1e-6, 1))])
    dump = np.load(ROOT / "features/holdout/probs_final.npz", allow_pickle=True)
    order = {k: i for i, k in enumerate(dump["game_key"])}
    rows = np.array([order[k] for k in hold_keys])
    hold_logs = np.stack([np.log(np.clip(dump["d96_h"][rows], 1e-6, 1)),
                          np.log(np.clip(dump["resnet8c_h"][rows], 1e-6, 1))])
    return val_logs, hold_logs


@dataclasses.dataclass
class Questions:
    """Per-question summaries: mean strength s, number of games k, mean game vector v, label y."""
    s: np.ndarray
    k: np.ndarray
    v: np.ndarray
    y: np.ndarray

    @classmethod
    def from_games(cls, games: list[np.ndarray], s_game: np.ndarray, v_game: np.ndarray,
                   labels: np.ndarray) -> "Questions":
        """Averages the per-game strength scores and vectors over each question's games."""
        return cls(s=np.array([s_game[g].mean() for g in games]),
                   k=np.array([len(g) for g in games]),
                   v=np.stack([v_game[g].mean(0) for g in games]), y=labels)

    def subset(self, mask: np.ndarray) -> "Questions":
        """The questions selected by a boolean mask."""
        return Questions(self.s[mask], self.k[mask], self.v[mask], self.y[mask])

    def features(self, use_vec: bool) -> np.ndarray:
        """[s, log k] (+ the mean 128-d vector when use_vec)."""
        base = np.stack([self.s, np.log(self.k)], 1)
        return np.concatenate([base, self.v], 1) if use_vec else base


@dataclasses.dataclass
class Validation:
    """Validation games and the simulated questions on them."""
    logs: np.ndarray          # [members, G, 10] production member log-probabilities
    y: np.ndarray             # [G] rank ids
    player: np.ndarray        # [G] player ids
    games: list[np.ndarray]   # game indices of each question
    q_player: np.ndarray      # player of each question
    players: np.ndarray       # sorted unique players


# -------------------------------------------------- ordinal logistic (cumulative link) with L2
def fit_ordinal(features: np.ndarray, y: np.ndarray,
                l2: float) -> Callable[[np.ndarray], np.ndarray]:
    """Fits a cumulative-link model (L2 on all but the first two weights); returns a predictor."""
    x = torch.tensor(features, dtype=torch.float32)
    mean, std = x.mean(0), x.std(0) + 1e-6
    scaled = (x - mean) / std
    labels = torch.tensor(y)
    w = torch.zeros(x.shape[1], requires_grad=True)
    first = torch.tensor(-4.0, requires_grad=True)
    gaps = torch.full((8,), 0.0, requires_grad=True)
    opt = torch.optim.LBFGS([w, first, gaps], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

    def logp(z: torch.Tensor) -> torch.Tensor:
        """Class log-probabilities of standardised inputs."""
        th = first + torch.cat([torch.zeros(1), torch.nn.Softplus()(gaps + 0.5).cumsum(0)])
        cum = torch.sigmoid((z @ w)[:, None] - th[None])
        up = torch.cat([torch.ones_like(cum[:, :1]), cum], 1)
        lo = torch.cat([cum, torch.zeros_like(cum[:, :1])], 1)
        return torch.log((up - lo).clamp_min(1e-7))

    def closure() -> torch.Tensor:
        """Penalised negative log-likelihood for LBFGS."""
        opt.zero_grad()
        loss = torch.nn.functional.nll_loss(logp(scaled), labels)
        if x.shape[1] > 2:
            loss = loss + l2 * (w[2:] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)  # type: ignore[arg-type]

    def predict(new: np.ndarray) -> np.ndarray:
        """Class probabilities of raw (unstandardised) inputs."""
        with torch.no_grad():
            return logp((torch.tensor(new, dtype=torch.float32) - mean) / std).exp().numpy()
    return predict


def fit_cutpoints(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """9 increasing hard thresholds on s that maximise the competition score
    (coordinate ascent on quantile grid)."""
    grid = np.quantile(s, np.linspace(0.005, 0.995, 400))
    # start at class-frequency quantiles
    th = np.quantile(s, np.cumsum(np.bincount(y, minlength=10))[:-1] / len(y))

    def total(t: np.ndarray) -> float:
        """Mean score of the hard decode with thresholds t."""
        return score_vec(np.searchsorted(np.sort(t), s, side="right"), y).mean()
    best = total(th)
    for _ in range(4):
        for j in range(9):
            lo = th[j - 1] if j > 0 else -np.inf
            hi = th[j + 1] if j < 8 else np.inf
            for g in grid[(grid > lo) & (grid < hi)]:
                trial = th.copy()
                trial[j] = g
                value = total(trial)
                if value > best + 1e-12:
                    best, th = value, trial
    return np.sort(th)


def production_predict(val: Validation, train: tuple[np.ndarray, list[np.ndarray]],
                       test_games: list[np.ndarray], logs: np.ndarray,
                       members: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """A: weights on training-fold games, offsets on training-fold questions, applied to test
    questions. train = (training-game mask, game indices of the training questions).
    members=(0,) gives A2: the same pipeline with the dense Transformer only."""
    m = list(members)
    train_games, train_q = train
    w = blend_eval.fit_weights(val.logs[m][:, train_games], val.y[train_games])
    vf = blend_eval.fuse(val.logs[m], w)
    b = lib.fit_class_offsets(np.stack([lib.pool(vf[i], "logmean") for i in train_q]))
    fused = blend_eval.fuse(logs[m], w)
    probs = lib.apply_class_offsets(np.stack([lib.pool(fused[i], "logmean") for i in test_games]),
                                    b)
    return (probs @ W).argmax(1)


# -------------------------------------------------- 5-fold player-disjoint CV on validation
def fold_scores(val: Validation, questions: Questions, test_mask: np.ndarray,
                train_games: np.ndarray) -> tuple[list[np.ndarray], dict[float, np.ndarray]]:
    """Scores of methods A, A2, B, C and of D for every l2 on the test questions of one fold."""
    train_mask = ~test_mask
    train = (train_games, [val.games[i] for i in np.flatnonzero(train_mask)])
    test_games = [val.games[i] for i in np.flatnonzero(test_mask)]
    tr_q, te_q = questions.subset(train_mask), questions.subset(test_mask)
    scores = [score_vec(production_predict(val, train, test_games, val.logs), te_q.y),
              score_vec(production_predict(val, train, test_games, val.logs, (0,)), te_q.y)]
    th = fit_cutpoints(tr_q.s, tr_q.y)
    scores.append(score_vec(np.searchsorted(th, te_q.s, side="right"), te_q.y))
    calibrate = fit_ordinal(tr_q.features(False), tr_q.y, 0.0)
    scores.append(score_vec((calibrate(te_q.features(False)) @ W).argmax(1), te_q.y))
    by_l2 = {}
    for l2 in L2S:
        classify = fit_ordinal(tr_q.features(True), tr_q.y, l2)
        by_l2[l2] = score_vec((classify(te_q.features(True)) @ W).argmax(1), te_q.y)
    return scores, by_l2


def cross_validate(val: Validation, questions: Questions, fold_of: dict[Any, int]
                   ) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray]]:
    """Out-of-fold per-question scores of every method and of D for every l2."""
    q_fold = np.array([fold_of[p] for p in val.q_player])
    cv = {m: np.zeros(len(val.games)) for m in METHODS}
    cv_d = {l2: np.zeros(len(val.games)) for l2 in L2S}
    for f in range(5):
        te_q = q_fold == f
        train_players = set(val.players[[fold_of[p] != f for p in val.players]])
        train_games = np.isin(val.player, list(train_players))
        scores, by_l2 = fold_scores(val, questions, te_q, train_games)
        for m, score in zip(METHODS, scores):
            cv[m][te_q] = score
        for l2, score in by_l2.items():
            cv_d[l2][te_q] = score
        print(f"fold {f} done", flush=True)
    return cv, cv_d


def player_boot(a: np.ndarray, b: np.ndarray, q_player: np.ndarray, players: np.ndarray,
                rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired bootstrap over validation players of mean(a) - mean(b)."""
    idx = {p: np.flatnonzero(q_player == p) for p in players}
    sa = np.array([a[idx[p]].sum() for p in players])
    sb = np.array([b[idx[p]].sum() for p in players])
    n = np.array([len(idx[p]) for p in players])
    d = []
    for _ in range(3000):
        k = rng.integers(0, len(players), len(players))
        d.append(sa[k].sum() / n[k].sum() - sb[k].sum() / n[k].sum())
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cv_report(cv: dict[str, np.ndarray], val: Validation) -> dict[str, dict[str, Any]]:
    """Prints and summarises the CV scores (A2 vs A, every other method vs A2)."""
    boot_rng = np.random.default_rng(1)
    out = {}
    print("\n5-fold player-disjoint CV on validation questions "
          "(20 questions per player, k = 5..20)")
    for m in METHODS:
        exact = float((cv[m] == 1).mean())
        if m == METHODS[0]:
            d: tuple[float, float, float] = (0, 0, 0)
        else:
            ref = cv[METHODS[0]] if m == METHODS[1] else cv[METHODS[1]]
            d = player_boot(cv[m], ref, val.q_player, val.players, boot_rng)
        out[m] = {"score": float(cv[m].mean()), "exact": exact, "minus_A": d}
        versus = ("" if m == METHODS[0] else f"   vs {'A' if m == METHODS[1] else 'A2'} "
                  f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        print(f"  {m:30s} score {cv[m].mean():.4f}  exact {exact:.3f}" + versus)
    return out


# -------------------------------------------------- holdout: scored once, fitted on validation
def holdout_questions(hold: dict[str, np.ndarray], s_hold: np.ndarray,
                      v_hold: np.ndarray) -> tuple[list[np.ndarray], Questions]:
    """The 400 holdout questions (games, summaries and true ranks) in labels.csv order."""
    lab = pd.read_csv(ROOT / "features/holdout/tests/labels.csv")
    truth = np.array([lib.LEVELS.index(r) for r in lab["rank"]])
    by_q: dict[str, list[int]] = {}
    for i, q in enumerate(hold["player_id"]):
        by_q.setdefault(str(q), []).append(i)
    h_games = [np.array(by_q[q]) for q in lab["question_id"].tolist()]
    return h_games, Questions.from_games(h_games, s_hold, v_hold, truth)


def holdout_scores(val: Validation, questions: Questions,
                   holdout: tuple[list[np.ndarray], Questions], hold_logs: np.ndarray,
                   best_l2: float) -> dict[str, np.ndarray]:
    """Per-question holdout scores of every method, everything fitted on all validation."""
    h_games, h_q = holdout
    train = (np.ones(len(val.y), bool), val.games)
    scores = {
        METHODS[0]: score_vec(production_predict(val, train, h_games, hold_logs), h_q.y),
        METHODS[1]: score_vec(production_predict(val, train, h_games, hold_logs, (0,)), h_q.y),
        METHODS[2]: score_vec(np.searchsorted(fit_cutpoints(questions.s, questions.y), h_q.s,
                                              side="right"), h_q.y),
    }
    calibrate = fit_ordinal(questions.features(False), questions.y, 0.0)
    scores[METHODS[3]] = score_vec((calibrate(h_q.features(False)) @ W).argmax(1), h_q.y)
    classify = fit_ordinal(questions.features(True), questions.y, best_l2)
    scores[METHODS[4]] = score_vec((classify(h_q.features(True)) @ W).argmax(1), h_q.y)
    return scores


def standardised_strength() -> tuple[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray], ...]:
    """(features, strength score, seed-0 vectors) of the validation and holdout games; every
    seed's score is standardised on validation before averaging over seeds."""
    val_feats, s_val_raw, v_val = game_outputs("val")
    hold, s_hold_raw, v_hold = game_outputs("holdout")
    # per-seed standardisation on validation
    mu, sd = s_val_raw.mean(1, keepdims=True), s_val_raw.std(1, keepdims=True)
    s_val, s_hold = ((s_val_raw - mu) / sd).mean(0), ((s_hold_raw - mu) / sd).mean(0)
    return (val_feats, s_val, v_val), (hold, s_hold, v_hold)


def load_inputs() -> tuple[Validation, Questions, tuple[list[np.ndarray], Questions], np.ndarray]:
    """Validation data and questions, holdout questions and holdout member log-probabilities."""
    (val_feats, s_val, v_val), (hold, s_hold, v_hold) = standardised_strength()
    y_val = lib.rank_ids(val_feats["rank"])
    val_logs, hold_logs = member_logs(val_feats["game_key"], hold["game_key"])
    # simulated validation questions: 20 per player, 5-20 games each
    games, q_y = residual_shift.validation_questions(val_feats["player_id"], y_val)
    players = np.array(sorted(set(val_feats["player_id"])))
    val = Validation(val_logs, y_val, val_feats["player_id"], games,
                     np.array([p for _ in range(20) for p in players]), players)
    return (val, Questions.from_games(games, s_val, v_val, q_y),
            holdout_questions(hold, s_hold, v_hold), hold_logs)


def holdout_report(hold_scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Prints and summarises the holdout scores."""
    print("\nholdout (400 questions, scored once; nothing fitted on it)")
    out = {}
    for m, s in hold_scores.items():
        se = s.std(ddof=1) / np.sqrt(len(s))
        out[m] = {"score": float(s.mean()), "exact": float((s == 1).mean()), "se": float(se)}
        print(f"  {m:30s} score {s.mean():.4f}  exact {(s == 1).mean():.3f}  +-{se:.3f}")
    return out


def main() -> None:
    """Runs the CV comparison, scores the holdout once and writes results/."""
    torch.manual_seed(0)
    val, questions, holdout, hold_logs = load_inputs()
    fold_of = {p: i % 5 for i, p in enumerate(np.random.default_rng(7).permutation(val.players))}
    cv, cv_d = cross_validate(val, questions, fold_of)
    best_l2 = max(L2S, key=lambda l2: cv_d[l2].mean())   # chosen by CV, never by the holdout
    cv[METHODS[4]] = cv_d[best_l2]
    cv_by_l2 = {str(l2): float(cv_d[l2].mean()) for l2 in L2S}
    cv_summary = cv_report(cv, val)
    print(f"  (D's l2 chosen by CV: {best_l2}; CV by l2: {cv_by_l2})")

    hold_scores = holdout_scores(val, questions, holdout, hold_logs, best_l2)
    report = {"cv": cv_summary, "holdout": holdout_report(hold_scores), "chosen_l2_for_D": best_l2,
              "cv_D_by_l2": cv_by_l2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    # per-question arrays for experiments/strength_axis/ranking_ability.py (same questions as above)
    h_q = holdout[1]
    arrays: dict[str, Any] = {
        "q_s": questions.s, "q_y": questions.y, "q_k": questions.k, "q_player": val.q_player,
        "q_fold": np.array([fold_of[p] for p in val.q_player]), "h_s": h_q.s, "h_y": h_q.y,
        "h_k": h_q.k}
    arrays.update({"cv_" + m.split()[0]: cv[m] for m in METHODS})
    arrays.update({"hold_" + m.split()[0]: s for m, s in hold_scores.items()})
    np.savez(OUT / "questions.npz", **arrays)
    print("saved", OUT / "questions.npz")


if __name__ == "__main__":
    main()
