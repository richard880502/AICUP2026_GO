"""Three score-raising experiments on top of the new default (best2 + opponent member), as RESIDUAL
corrections of its logits.

Base: per question, z = log of the log-mean pooled, validation-fitted fusion of d96 / resnet8c /
d96-on-opponent probabilities (exactly the default pipeline before its per-rank offsets).  Every
model below predicts  z' = z + b + f(x)  and decodes with the expected competition score, so f = 0
reproduces the default (M0 = z + validation-fitted offsets).

Question features x (per question, from its 5-20 games; residual_shift.feature_blocks):
  E1 opponent distribution: expected rank of the d96 member on the OPPONENT's moves - mean / std /
     q10 / q50 / q90 / min / max, mean class probabilities, mean entropy, argmax agreement,
     log #games.
  E2 player x opponent pairing: expected rank on the player's own moves (mean / std), d = own -
     opponent per game (std, q10, q90, min, max, mean |d|), correlation of own and opponent across
     games.
  E3 game-result proxy (SGF has no result/handicap/komi): KataGo final-position win rate / score
     lead for the player (scripts/extract_final_result.py), win share, an Elo-like performance =
     mean(opponent rank) + (win share - 0.5), mean(win * opponent rank), game length, share of
     games where the player moved last.
Models f: 'shift' (one scalar per question moving the distribution along the rank axis, linear in
x), 'linear' (10 x |x|), 'mlp' (one hidden layer, last layer initialised at 0), and a non-residual
HistGradientBoosting on [z, x] for comparison.

Protocol: all fitting on the validation cohort (400 players, 20 simulated 5-20 game questions
each); 5-fold player-disjoint CV picks the configuration; the fresh cohort (features/holdout2,
models *_h2) is scored once with every configuration fitted on the whole cohort, the CV-selected
one being the pre-registered candidate; the old holdout (*_h) is reported as secondary.
Run from the project root: c1/.venv/bin/python experiments/opponent_meta/run.py
"""
import json
import pathlib
import sys
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
import blend_eval  # noqa: E402  pylint: disable=wrong-import-position,import-error
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error
import residual_shift  # noqa: E402  pylint: disable=wrong-import-position,import-error

HERE = ROOT / "experiments/opponent_meta"
RES = ROOT / "results"
W_SCORE = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
BASE = "M0 default (offsets)"
SETS = {"E1": ("E1",), "E1+E2": ("E1", "E2"), "E1+E2+E3": ("E1", "E2", "E3"), "E3": ("E3",),
        "E2+E3": ("E2", "E3")}
Predictor = Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
Config = tuple[str, Optional[float], Optional[str]]


def load_final(split: str) -> dict[str, np.ndarray]:
    """game_key -> final-result features of features/final_result/<split>."""
    return residual_shift.load_final(ROOT / f"features/final_result/{split}")


def build(qgames: list[np.ndarray], fused: np.ndarray, p_own: np.ndarray, p_opp: np.ndarray,
          fin: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """qgames: list of game-index arrays; fused: fused per-game probabilities.
    Returns base logits z [Q, 10] and feature blocks."""
    z = np.log(np.stack([lib.pool(fused[g], "logmean") for g in qgames]))
    blocks: dict[str, list[np.ndarray]] = {"E1": [], "E2": [], "E3": []}
    for g in qgames:
        for name, block in zip(("E1", "E2", "E3"),
                               residual_shift.feature_blocks(p_own[g], p_opp[g], fin[g])):
            blocks[name].append(block)
    return z, {k: np.stack(v) for k, v in blocks.items()}


def xset(blocks: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Concatenation of the feature blocks of set `name`."""
    return np.concatenate([blocks[b] for b in SETS[name]], 1)


def decode_score(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-question competition score of the expected-score decode."""
    p = torch.softmax(torch.as_tensor(logits), 1).numpy()
    gap = np.abs((p @ W_SCORE).argmax(1) - y)
    return np.where(gap == 0, 1.0, np.where(gap == 1, lib.ADJACENT, 0.0))


class Residual(torch.nn.Module):
    """z + b + f(x) with f 'shift' (scalar along the rank axis), 'linear' or 'mlp';
    the last layer of f starts at zero."""

    def __init__(self, n_in: int, kind: str, hidden: int = 32):
        super().__init__()
        self.kind = kind
        self.b = torch.nn.Parameter(torch.zeros(10))
        self.f: torch.nn.Module
        if kind == "shift":
            self.f = last = torch.nn.Linear(n_in, 1)
        elif kind == "linear":
            self.f = last = torch.nn.Linear(n_in, 10)
        else:
            first = torch.nn.Linear(n_in, hidden)   # created before the last layer, as before
            last = torch.nn.Linear(hidden, 10)
            self.f = torch.nn.Sequential(first, torch.nn.GELU(), torch.nn.Dropout(0.1), last)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.register_buffer("axis", torch.linspace(-1.5, 1.5, 10))

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Corrected logits."""
        out = self.f(x)
        return z + self.b + (out * self.axis if self.kind == "shift" else out)


def train(m: Residual, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], lam: float,
          epochs: int) -> None:
    """Full-batch Adam (lr 0.02) on cross-entropy + lam * L2 on the weights of f; leaves m in eval
    mode."""
    zt, xt, yt = data
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        reg = sum((p ** 2).sum() for n, p in m.named_parameters()
                  if n.startswith("f.") and "weight" in n)
        loss = torch.nn.functional.cross_entropy(m(zt, xt), yt) + lam * reg
        loss.backward()
        opt.step()
    m.eval()


def fit_residual(z: np.ndarray, x: np.ndarray, y: np.ndarray, spec: tuple[str, float],
                 epochs: int = 300) -> Predictor:
    """Fits Residual(kind) with cross-entropy + lam * L2 on the weights of f; spec = (kind, lam)."""
    kind, lam = spec
    mu, sd = x.mean(0), x.std(0) + 1e-6
    torch.manual_seed(0)
    m = Residual(x.shape[1], kind)
    zt, xt, yt = (torch.as_tensor(a, dtype=torch.float32) for a in (z, (x - mu) / sd, y))
    train(m, (zt, xt, yt.long()), lam, epochs)

    def predict(z2: np.ndarray, x2: Optional[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            return m(torch.as_tensor(z2, dtype=torch.float32),
                     torch.as_tensor((x2 - mu) / sd, dtype=torch.float32)).numpy()
    return predict


def fit_offsets(z: np.ndarray) -> Predictor:
    """M0: validation-fitted per-rank offsets on the base logits."""
    offsets = lib.fit_class_offsets(np.exp(z) / np.exp(z).sum(1, keepdims=True))
    return lambda z2, x2: z2 + offsets


def fit_hgb(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Predictor:
    """Non-residual HistGradientBoosting on [z, x]."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(np.concatenate([z, x], 1), y)
    return lambda z2, x2: np.log(np.clip(clf.predict_proba(np.concatenate([z2, x2], 1)), 1e-9, 1))


def make_configs() -> dict[str, Config]:
    """Configuration name -> (kind, lam, feature set), M0 first."""
    configs: dict[str, Config] = {BASE: ("offsets", None, None)}
    for s in SETS:
        for kind, lams in (("shift", (0.01, 0.1)), ("linear", (0.1, 1.0)), ("mlp", (0.01,))):
            for lam in lams:
                configs[f"{kind} lam={lam} [{s}]"] = (kind, lam, s)
    configs["HGB on [z, x] [E1+E2+E3]"] = ("hgb", None, "E1+E2+E3")
    return configs


CONFIGS = make_configs()


def fit_config(cfg: Config, z: np.ndarray, blocks: dict[str, np.ndarray],
               y: np.ndarray) -> Predictor:
    """Fits one configuration on base logits z and feature blocks."""
    kind, lam, s = cfg
    if kind == "offsets":
        return fit_offsets(z)
    assert s is not None
    x = xset(blocks, s)
    if kind == "hgb":
        return fit_hgb(z, x, y)
    assert lam is not None
    return fit_residual(z, x, y, (kind, lam))


def features_of(cfg: Config, blocks: dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The configuration's feature matrix (None for M0)."""
    return xset(blocks, cfg[2]) if cfg[2] else None


def boot(diff: np.ndarray, groups: np.ndarray, reps: int = 5000, seed: int = 1) -> list[float]:
    """Mean of diff and a 95% bootstrap interval over groups."""
    ug = np.unique(groups)
    per = np.array([diff[groups == g].mean() for g in ug])
    r = np.random.default_rng(seed)
    bs = [per[r.integers(0, len(per), len(per))].mean() for _ in range(reps)]
    return [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


class Validation(NamedTuple):
    """Validation members and the simulated questions they are fitted on."""
    vlogs: np.ndarray            # [3, G, 10] member log-probabilities
    p_own: np.ndarray            # d96 member probabilities per game
    p_opp: np.ndarray            # d96-on-opponent member probabilities per game
    fin: np.ndarray              # final-result features per game
    y: np.ndarray                # label per game
    player: np.ndarray           # player per game
    questions: list[np.ndarray]  # game rows of each question
    qy: np.ndarray               # label per question
    qp: np.ndarray               # player per question


def load_validation(sfx: str) -> Validation:
    """Validation probabilities of the three members *sfx and 20 questions per player."""
    members = [f"d96{sfx}", f"resnet8c{sfx}", f"d96{sfx}_opp"]
    files = {f"d96{sfx}": "transformer", f"resnet8c{sfx}": f"resnet8c{sfx}",
             f"d96{sfx}_opp": "transformer"}
    vz = {m: np.load(RES / f"val_probs_{m}.npz", allow_pickle=True) for m in members}
    keys = vz[members[0]]["game_key"]
    for m in members:
        assert (vz[m]["game_key"] == keys).all()
    vp = {m: np.clip(vz[m][files[m]], 1e-6, 1) for m in members}
    y, player = vz[members[0]]["y"], vz[members[0]]["player"]
    fv = load_final("val")
    questions, qy = residual_shift.validation_questions(player, y)
    return Validation(np.log(np.stack([vp[m] for m in members])), vp[f"d96{sfx}"],
                      vp[f"d96{sfx}_opp"], np.stack([fv[k] for k in keys]), y, player, questions,
                      qy, np.array([player[g[0]] for g in questions]))


def cross_validate(val: Validation, sfx: str) -> dict[str, np.ndarray]:
    """5-fold player-disjoint CV on the validation questions: per-question score per config."""
    players = np.array(sorted(set(val.player)))
    fold_of = dict(zip(np.random.default_rng(0).permutation(players),
                       np.arange(len(players)) % 5))
    qf = np.array([fold_of[p] for p in val.qp])
    cv = {c: np.zeros(len(val.questions)) for c in CONFIGS}
    for k in range(5):
        trg = np.isin(val.player, players[[fold_of[p] != k for p in players]])
        z, blocks = build(val.questions,
                          blend_eval.fuse(val.vlogs,
                                          blend_eval.fit_weights(val.vlogs[:, trg], val.y[trg])),
                          val.p_own, val.p_opp, val.fin)
        te = qf == k
        for name, cfg in CONFIGS.items():
            pred = fit_config(cfg, z[~te], {b: v[~te] for b, v in blocks.items()}, val.qy[~te])
            x_te = features_of(cfg, blocks)
            cv[name][te] = decode_score(pred(z[te], x_te[te] if x_te is not None else None),
                                        val.qy[te])
        print(f"  [{sfx}] fold {k} done", flush=True)
    return cv


class Test(NamedTuple):
    """Test-cohort questions and member probabilities."""
    questions: list[np.ndarray]  # game rows of each question
    probs: dict[str, np.ndarray]  # member -> clipped probabilities per game
    fin: np.ndarray               # final-result features per game
    y: np.ndarray                 # label per question


def load_test(sfx: str, test_dump: pathlib.Path, test_split: str, test_labels: str) -> Test:
    """Per-game member probabilities from the dump, final-result features and labels."""
    td = np.load(test_dump, allow_pickle=True)
    tp = {m: np.clip(td[m], 1e-6, 1) for m in (f"d96{sfx}", f"resnet8c{sfx}", f"d96{sfx}_opp")}
    ft = load_final(test_split)
    lab = pd.read_csv(ROOT / test_labels)
    by_q: dict[str, list[int]] = {}
    for i, q in enumerate(td["player_id"]):
        by_q.setdefault(str(q), []).append(i)
    return Test([np.array(by_q[q]) for q in lab["question_id"]], tp,
                np.stack([ft[k] for k in td["game_key"]]),
                np.array([lib.LEVELS.index(r) for r in lab["rank"]]))


def score_test(sfx: str, val: Validation, tst: Test) -> dict[str, np.ndarray]:
    """Fits every config on the whole validation cohort and scores the test cohort once."""
    w = blend_eval.fit_weights(val.vlogs, val.y)
    z, blocks = build(val.questions, blend_eval.fuse(val.vlogs, w), val.p_own, val.p_opp, val.fin)
    tz, tblocks = build(tst.questions,
                        blend_eval.fuse(np.log(np.stack(list(tst.probs.values()))), w),
                        tst.probs[f"d96{sfx}"], tst.probs[f"d96{sfx}_opp"], tst.fin)
    ty = tst.y
    test = {}
    for name, cfg in CONFIGS.items():
        pred = fit_config(cfg, z, blocks, val.qy)
        test[name] = decode_score(pred(tz, features_of(cfg, tblocks)), ty)
    return test


def run(sfx: str, test_dump: pathlib.Path, test_split: str,
        test_labels: str) -> dict[str, dict[str, Any]]:
    """CV on validation and a single test score for every configuration (members *sfx)."""
    val = load_validation(sfx)
    cv = cross_validate(val, sfx)
    tst = load_test(sfx, test_dump, test_split, test_labels)
    test = score_test(sfx, val, tst)
    n_test = len(tst.y)
    out = {}
    for name in CONFIGS:
        out[name] = {"val_cv": float(cv[name].mean()),
                     "val_cv_minus_M0": boot(cv[name] - cv[BASE], val.qp),
                     "test": float(test[name].mean()),
                     "test_minus_M0": boot(test[name] - test[BASE], np.arange(n_test))}
    return out


def print_cohort(title: str, sfx: str, r: dict[str, dict[str, Any]]) -> str:
    """Console table of one cohort; returns the CV-selected configuration."""
    sel = max((n for n in r if n != BASE), key=lambda n: r[n]["val_cv"])
    print(f"\n=== {title} (validation-fitted, models *{sfx});  CV-selected candidate: {sel}")
    print(f"{'config':34s} {'val CV':>7s} {'vs M0 [95%]':>26s}   {'test':>7s} {'vs M0 [95%]':>26s}")
    for n, v in r.items():
        a, b = v["val_cv_minus_M0"], v["test_minus_M0"]
        mark = "  <= selected" if n == sel else ""
        print(f"{n:34s} {v['val_cv']:.4f} {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]   "
              f"{v['test']:.4f} {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]{mark}")
    return sel


def main() -> None:
    """Both cohorts; writes results/report.json."""
    torch.set_num_threads(8)
    report: dict[str, Any] = {}
    for sfx, dump, split, labels, title in (
            ("_h2", HERE / "probs_h2.npz", "holdout2", "features/holdout2/tests/labels.csv",
             "fresh cohort"),
            ("_h", HERE / "probs_h.npz", "holdout", "features/holdout/tests/labels.csv",
             "old holdout")):
        r = run(sfx, dump, split, labels)
        report[title] = r
        report[title + " selected"] = print_cohort(title, sfx, r)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/report.json").write_text(json.dumps(report, indent=2))
    print("saved", HERE / "results/report.json")


if __name__ == "__main__":
    main()
