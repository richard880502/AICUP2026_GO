# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""P1: does a rank class-count correction help v47 itself? (holdout3 cross-fit)

`count_prior_check.py` found that v47-style soft pooling, which sums identity
probability over every gallery class of a rank, pulls opponents toward
populous ranks (A has 1,434 gallery classes, 1D has 358), and that averaging
within a rank (beta = 1) helps on a near rank-balanced known-player control.
v47's learned transition tables and bias may already absorb a constant
per-rank skew, so the only valid test is inside v47.

Preregistered before running:

* arm ``beta1``: posterior_r  proportional to  n_r ** -1 * sum_{c in r} softmax_c,
  beta fixed at 1 from the independent control (not fitted on holdout3);
* arm ``beta_learned``: beta = softplus(raw), raw initialised at -5 (beta about
  0.007), fitted with every other v47 parameter on the training half;
* reference ``beta0``: the identical code path at beta = 0, which must replay
  the production v47 cross-fit (0.765302 under deterministic algorithms);
* n_r counts the classes that stay in the softmax, i.e. excluding the masked
  holdout3 classes;
* primary gate on ``beta1 - beta0``: delta >= +0.003, player-bootstrap 95%
  lower bound > 0, both halves >= 0, and the mean over five jittered refits
  (base times exp(1e-6 N(0,1)), jitter shared across arms) >= +0.003;
* secondary: the 1,587 official-compatible questions (no HD>=2 game), per-rank
  utility and adjacent-boundary AUCs.

Exploratory on a repeatedly inspected cohort; a pass would justify a fresh
nested evaluation, not adoption. Run from ``runtime/competition2_research``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F


def _load_p2() -> Any:
    """p2_evaluate, loaded by path; it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts."""
    path = pathlib.Path(__file__).resolve().parents[1] / "ten_profile_production" / "p2_evaluate.py"
    spec = importlib.util.spec_from_file_location("p2_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p2 = _load_p2()

import rank_lib as lib  # pylint: disable=wrong-import-position,import-error

GATE = 0.003
SEED = 20261120


_OpponentIdentity: Any = p2.OpponentIdentity


class CountCorrected(_OpponentIdentity):
    """v47 with posterior_r scaled by n_r ** -beta before the unknown mixture."""

    def __init__(self, ranks: torch.Tensor, log_counts: torch.Tensor, beta: float | None):
        super().__init__(ranks)
        self.register_buffer("log_counts", log_counts)
        self.fixed_beta = beta
        if beta is None:
            self.beta_raw = torch.nn.Parameter(torch.tensor(-5.0))

    def beta(self) -> torch.Tensor:
        """The fixed or learned exponent."""
        if self.fixed_beta is not None:
            return torch.tensor(self.fixed_beta, device=self.log_counts.device)
        return F.softplus(self.beta_raw)  # pylint: disable=not-callable

    def forward(  # pylint: disable=arguments-differ
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        game_question: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """p2.OpponentIdentity.forward with the count correction inserted."""
        usable = kind < 3
        posterior = (scores / self.log_temperature.exp()).softmax(1) @ self.ranks
        if self.fixed_beta != 0.0:
            posterior = posterior * torch.exp(-self.beta() * self.log_counts)[None, :]
            posterior = posterior / posterior.sum(1, keepdim=True)
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * self.prior.softmax(0)
        transition = self.offsets[:, self.offset_index].log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None] + transition[kind.clamp(max=2)], dim=1
        )
        per_game = per_game * usable[:, None]
        total = torch.zeros(len(base), 10, device=base.device).index_add_(
            0, game_question, per_game
        )
        used = torch.zeros(len(base), device=base.device).index_add_(
            0, game_question, usable.float()
        ).clamp(min=1)
        logits = self.log_alpha.exp() * total / used[:, None] ** self.gamma + self.bias
        return logits + torch.sigmoid(self.base_weight) * 2 * base.clamp(min=1e-9).log()


def crossfit(inputs: dict, base: torch.Tensor, make_model, steps: int) -> tuple[np.ndarray, list]:
    """Fit on one holdout3 half, predict the other; also return learned betas."""
    arguments = (inputs["sims"], inputs["kind"], inputs["game_question"], base)
    truth = inputs["truth"]
    output = torch.zeros(len(truth), 10, device=base.device)
    betas = []
    for half in (0, 1):
        mask = torch.from_numpy(inputs["fold"] != half).to(base.device)
        model = make_model().to(base.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(steps):
            loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            output[~mask] = model(*arguments).softmax(1)[~mask]
        betas.append(float(model.beta()) if isinstance(model, CountCorrected) else 0.0)
    return output.double().cpu().numpy(), betas


def setup(state: pathlib.Path, labels_path: pathlib.Path, device: torch.device) -> dict:
    """v47 inputs, kept-class counts per rank, players and the HD>=2 question mask."""
    inputs = p2.load_v47(state, device)
    holdout = torch.load(state, map_location="cpu", weights_only=False)["holdout_classes"]
    kept = np.ones(inputs["ranks"].shape[0], dtype=bool)
    kept[np.asarray(holdout)] = False
    counts = inputs["ranks"].cpu().numpy()[kept].sum(0)
    labels = pd.read_csv(labels_path).set_index("question_id").loc[inputs["order"]]
    truth = inputs["truth"].cpu().numpy()
    if not np.array_equal(lib.rank_ids(labels["rank"].to_numpy()), truth):
        raise ValueError("question_folds.csv ranks disagree with the v47 cache")
    kind = inputs["kind"].cpu().numpy()
    question = inputs["game_question"].cpu().numpy()
    has_hd2 = np.bincount(question[kind == 3], minlength=len(truth)) > 0
    return {
        "inputs": inputs,
        "counts": counts,
        "log_counts": torch.log(torch.tensor(counts, dtype=torch.float32, device=device)),
        "players": labels["player"].to_numpy().astype(str),
        "truth": truth,
        "fold": inputs["fold"],
        "official_compatible": ~has_hd2,
    }


def compare(arms: dict, ctx: dict, mask: np.ndarray) -> dict:
    """Paired deltas against beta0 on a question subset."""
    truth = ctx["truth"][mask]
    scores = {name: p2.utility(q[mask], truth) for name, q in arms.items()}
    output = {}
    for name in arms:
        if name == "beta0":
            continue
        result = p2.paired(scores[name], scores["beta0"], ctx["players"][mask],
                           ctx["fold"][mask], 5000)
        result["changed_decisions"] = int(
            ((arms[name][mask] @ p2.GAIN).argmax(1)
             != (arms["beta0"][mask] @ p2.GAIN).argmax(1)).sum()
        )
        result["per_rank_delta"] = {
            r: float((scores[name] - scores["beta0"])[truth == i].mean())
            for i, r in enumerate(lib.LEVELS)
        }
        output[f"{name}-beta0"] = result
    return output


def main() -> None:  # pylint: disable=too-many-locals
    """Runs every arm, the sanity replays, jittered refits and the gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=pathlib.Path,
        default=pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt"),
    )
    parser.add_argument("--labels", type=pathlib.Path,
                        default=pathlib.Path("features/holdout3/question_folds.csv"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    ctx = setup(args.state, args.labels, device)
    inputs, base = ctx["inputs"], ctx["inputs"]["base"]
    ranks, log_counts = inputs["ranks"], ctx["log_counts"]
    factories = {
        "beta0": lambda: CountCorrected(ranks, log_counts, 0.0),
        "beta1": lambda: CountCorrected(ranks, log_counts, 1.0),
        "beta_learned": lambda: CountCorrected(ranks, log_counts, None),
    }

    production = p2.v47_crossfit(inputs, base, args.steps)
    arms, learned = {}, {}
    for name, factory in factories.items():
        arms[name], learned[name] = crossfit(inputs, base, factory, args.steps)
        print(f"{name}: {p2.utility(arms[name], ctx['truth']).mean():.6f}", flush=True)
    sanity = {
        "production_v47": float(p2.utility(production, ctx["truth"]).mean()),
        "beta0_vs_production_max_abs_diff": float(np.abs(arms["beta0"] - production).max()),
    }

    generator = torch.Generator().manual_seed(SEED)
    jitter: dict[str, list[float]] = {name: [] for name in factories}
    for _ in range(args.replicates):
        noise = torch.exp(1e-6 * torch.randn(base.shape, generator=generator)).to(device)
        jittered = base * noise
        jittered = jittered / jittered.sum(1, keepdim=True)
        for name, factory in factories.items():
            output, _ = crossfit(inputs, jittered, factory, args.steps)
            jitter[name].append(float(p2.utility(output, ctx["truth"]).mean()))
        print("jitter", {k: round(v[-1], 6) for k, v in jitter.items()}, flush=True)
    jitter_delta = np.asarray(jitter["beta1"]) - np.asarray(jitter["beta0"])

    everything = np.ones(len(ctx["truth"]), dtype=bool)
    paired_all = compare(arms, ctx, everything)
    primary = paired_all["beta1-beta0"]
    report = {
        "gallery_kept_classes_per_rank": dict(zip(lib.LEVELS, ctx["counts"].astype(int).tolist())),
        "sanity": sanity,
        "learned_beta_by_half": learned["beta_learned"],
        "arms": {name: p2.describe(q, ctx["truth"], ctx["fold"]) for name, q in arms.items()},
        "paired_all": paired_all,
        "paired_official_compatible": compare(arms, ctx, ctx["official_compatible"]),
        "official_compatible_questions": int(ctx["official_compatible"].sum()),
        "jitter_refits": {
            "utility": jitter,
            "beta1-beta0": jitter_delta.tolist(),
            "beta1-beta0_mean": float(jitter_delta.mean()),
            "beta1-beta0_sd": float(jitter_delta.std(ddof=1)),
        },
        "gate_primary_beta1": {
            "delta": primary["delta"],
            "ci95_lower": primary["ci95"][0],
            "by_half": primary["by_half"],
            "jitter_mean": float(jitter_delta.mean()),
            "passes": bool(
                primary["delta"] >= GATE and primary["ci95"][0] > 0
                and all(v >= 0 for v in primary["by_half"].values())
                and jitter_delta.mean() >= GATE
            ),
        },
        "status": "exploratory; holdout3 repeatedly inspected; a pass needs a fresh nested test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    keys = ("sanity", "learned_beta_by_half", "gate_primary_beta1")
    print(json.dumps({k: report[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
