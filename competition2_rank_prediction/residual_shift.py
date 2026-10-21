"""Question-level 'shift' residual on top of the fused pipeline (confirmed 2026-10-21, C1).

z' = z + b + axis * (w . x): z is the log of the log-mean pooled fused probabilities of a question,
b a per-rank bias, and one scalar per question (linear in the standardised question features x)
moves the distribution along the rank axis.  x = E1 (the opponent member's distribution over the
question's games) + E2 (player x opponent pairing) + E3 (KataGo final-position result proxy,
scripts/extract_final_result.py).  Fitted on the validation cohort (20 simulated 5-20 game questions
per player), L2 = 0.01 on w, 300 full-batch Adam steps, seed 0: exactly the configuration that was
pre-registered and confirmed in experiments/crossfit_cohort.
"""
import pathlib

import numpy as np
import torch

import blend_eval
import rank_lib as lib

RANKS = np.arange(10)
LAMBDA = 0.01


def load_final(directory: pathlib.Path) -> dict[str, np.ndarray]:
    """game_key -> [final_winrate, final_score_lead, n_moves, last_move_by_player,
    ended_two_passes, leading_black_stones] from scripts/extract_final_result.py."""
    files = sorted(pathlib.Path(directory).glob("feat_*.npz"))
    if not files:
        raise FileNotFoundError(f"no final-result features in {directory} "
                                "(run scripts/extract_final_result.py)")
    out: dict[str, np.ndarray] = {}
    for path in files:
        data = np.load(path)
        out.update(zip(data["game_key"], data["values"]))
    return out


def feature_blocks(p_own: np.ndarray, p_opp: np.ndarray,
                   fin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E1, E2, E3 feature blocks of one question.

    p_own / p_opp: [k, 10] per-game probabilities of the player member / the opponent member;
    fin: [k, 6] final-position features of the same games.
    """
    e_own, e_opp = p_own @ RANKS, p_opp @ RANKS
    gap = e_own - e_opp
    k = len(e_own)
    entropy = -(p_opp * np.log(np.clip(p_opp, 1e-9, 1))).sum(1)
    agreement = np.bincount(p_opp.argmax(1), minlength=10).max() / k
    e1 = [e_opp.mean(), e_opp.std(), *np.quantile(e_opp, [.1, .5, .9]), e_opp.min(), e_opp.max(),
          *p_opp.mean(0), entropy.mean(), agreement, np.log(k)]
    varied = k >= 3 and e_own.std() > 1e-6 and e_opp.std() > 1e-6
    corr = np.corrcoef(e_own, e_opp)[0, 1] if varied else 0.0
    e2 = [e_own.mean(), e_own.std(), gap.std(), *np.quantile(gap, [.1, .9]), gap.min(), gap.max(),
          np.abs(gap).mean(), corr]
    fin = np.nan_to_num(fin, nan=0.5)
    win = (fin[:, 0] > 0.5).astype(float)
    e3 = [win.mean(), fin[:, 0].mean(), np.clip(fin[:, 1], -60, 60).mean() / 60,
          e_opp.mean() + win.mean() - 0.5, (win * e_opp).mean(), fin[:, 2].mean() / 300,
          fin[:, 3].mean()]
    return np.array(e1), np.array(e2), np.array(e3)


def question_features(p_own: np.ndarray, p_opp: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """E1 + E2 + E3 concatenated (the confirmed feature set)."""
    return np.concatenate(feature_blocks(p_own, p_opp, fin))


def question_inputs(questions: list[np.ndarray], fused: np.ndarray, p_own: np.ndarray,
                    p_opp: np.ndarray, fin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """questions: game-index arrays. Returns base logits z [Q, 10] and features x [Q, n]."""
    z = np.log(np.stack([lib.pool(fused[g], "logmean") for g in questions]))
    x = np.stack([question_features(p_own[g], p_opp[g], fin[g]) for g in questions])
    return z, x


class ShiftModule(torch.nn.Module):
    """z + b + axis * f(x), with f linear and initialised at zero (so it starts as the identity)."""

    def __init__(self, n_in: int):
        super().__init__()
        self.b = torch.nn.Parameter(torch.zeros(10))
        self.f = torch.nn.Linear(n_in, 1)
        torch.nn.init.zeros_(self.f.weight)
        torch.nn.init.zeros_(self.f.bias)
        self.register_buffer("axis", torch.linspace(-1.5, 1.5, 10))

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Corrected logits."""
        return z + self.b + self.f(x) * self.axis


class ShiftResidual:
    """Standardises x, fits ShiftModule with cross-entropy + L2 on the shift weights."""

    def __init__(self) -> None:
        self.mu: np.ndarray = np.zeros(0)
        self.sd: np.ndarray = np.ones(0)
        self.module: ShiftModule | None = None

    def fit(self, z: np.ndarray, x: np.ndarray, y: np.ndarray, lam: float = LAMBDA,
            epochs: int = 300) -> "ShiftResidual":
        """Full-batch Adam (lr 0.02), seed 0."""
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        torch.manual_seed(0)
        module = ShiftModule(x.shape[1])
        z_t = torch.as_tensor(z, dtype=torch.float32)
        x_t = torch.as_tensor((x - self.mu) / self.sd, dtype=torch.float32)
        y_t = torch.as_tensor(y).long()
        optimiser = torch.optim.Adam(module.parameters(), lr=0.02)
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(module(z_t, x_t), y_t)
            loss = loss + lam * (module.f.weight ** 2).sum()
            loss.backward()
            optimiser.step()
        self.module = module
        return self

    def logits(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Corrected logits for new questions."""
        assert self.module is not None, "fit() first"
        with torch.no_grad():
            out = self.module(torch.as_tensor(z, dtype=torch.float32),
                              torch.as_tensor((x - self.mu) / self.sd, dtype=torch.float32))
        return out.numpy()

    def predict_proba(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Corrected class probabilities for new questions."""
        return torch.softmax(torch.as_tensor(self.logits(z, x)), 1).numpy()


def validation_questions(player: np.ndarray,
                         y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """The 20 simulated 5-20 game questions per validation player used since
    experiments/strength_axis (seed 20261020). Returns game-index arrays and their labels."""
    rng = np.random.default_rng(20261020)
    players = np.array(sorted(set(player)))
    rows = {p: np.flatnonzero(player == p) for p in players}
    questions = [rng.choice(rows[p], size=min(int(rng.integers(5, 21)), len(rows[p])),
                            replace=False)
                 for _ in range(20) for p in players]
    return questions, np.array([y[g[0]] for g in questions])


def fit_on_validation(val_logs: np.ndarray, weights: np.ndarray,
                      labels: tuple[np.ndarray, np.ndarray],
                      members: tuple[np.ndarray, np.ndarray], fin: np.ndarray) -> ShiftResidual:
    """Fits the residual on the validation questions.

    val_logs: [M, G, 10] member log-probabilities; weights: fusion weights; labels: (y, player) per
    game; members: (player-member, opponent-member) probabilities per game; fin: [G, 6].
    """
    y, player = labels
    questions, q_y = validation_questions(player, y)
    z, x = question_inputs(questions, blend_eval.fuse(val_logs, weights), members[0], members[1],
                           fin)
    return ShiftResidual().fit(z, x, q_y)
