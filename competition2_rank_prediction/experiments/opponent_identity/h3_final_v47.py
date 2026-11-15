"""Leak check on holdout3, then the opponent-identity model on the official questions.

Reproduces Competition 2 rank submission v47 (`rank_opponent_identity_v47.csv`),
step 8 of 8 (after h3_v7.py). Reads h3_sims_v47.pt, test_opp_scores.npz
(link_test.py), test_opp_sims_v7.npy (h3_v7.py), h3_opp.csv, h3_map.npz
(map_h3.py), the style_v4 checkpoint and the official C1+C2 probabilities.

1. Leak check: v4 was trained on some holdout3 players' own moves in the very
   games asked about. Compare the gain on questions none of whose games were
   in v4's training corpus with the rest.
2. Fit the ~45-parameter model on all 2,000 holdout3 questions (known-player
   list without any holdout3 player, which matches the official share of
   unknown opponents best) and apply it to the 800 official questions with
   the full list. Writes a candidate file in the scratch directory only.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 python -m \
      competition2_rank_prediction.experiments.opponent_identity.h3_final_v47
"""
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda:0"
# The research machine's locations for the opponent-identity and rank-probe files.
PROBE = "/tmp2/richard/claude_scratch/opp_probe"
RANK = "/tmp2/richard/claude_scratch/rank_probe"
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
GAIN = torch.tensor(
    [[1.0 if a == b else (np.exp(-1) if abs(a - b) == 1 else 0.0) for b in range(10)]
     for a in range(10)],
    device=DEVICE, dtype=torch.float32,
)
OFFSET = (torch.arange(10)[None] - torch.arange(10)[:, None]).clamp(-3, 3).to(DEVICE) + 3


class Model(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """Per-game opponent-rank evidence, multiplied over a question, plus the C1+C2 log-prior."""

    def __init__(self, ranks):
        super().__init__()
        self.ranks = ranks
        self.log_temperature = torch.nn.Parameter(torch.tensor(np.log(0.04)))
        self.unknown = torch.nn.Parameter(torch.full((3,), -1.0))
        self.prior = torch.nn.Parameter(torch.zeros(10))
        self.offsets = torch.nn.Parameter(torch.zeros(3, 7))
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))
        self.bias = torch.nn.Parameter(torch.zeros(10))
        self.base_weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, scores, kind, game_question, base):
        """Question logits from per-game class scores, game kinds and base probabilities."""
        questions = len(base)
        usable = kind < 3
        posterior = (scores / self.log_temperature.exp()).softmax(1) @ self.ranks
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * self.prior.softmax(0)
        transition = self.offsets[:, OFFSET].log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None] + transition[kind.clamp(max=2)], dim=1
        )
        per_game = per_game * usable[:, None]
        total = torch.zeros(questions, 10, device=DEVICE).index_add_(0, game_question, per_game)
        used = torch.zeros(questions, device=DEVICE).index_add_(
            0, game_question, usable.float()
        ).clamp(min=1)
        logits = self.log_alpha.exp() * total / used[:, None] ** self.gamma + self.bias
        return logits + torch.sigmoid(self.base_weight) * 2 * base.clamp(min=1e-9).log()


def decode(probabilities):
    """Expected-score decoding."""
    return (probabilities @ GAIN).argmax(1)


def fit(model, arguments, truth, mask, steps=300):
    """Adam on the cross-entropy of the masked questions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def leak_check(ranks, arguments, truth, fold):  # pylint: disable=too-many-locals
    """Cross-fitted gain on questions whose games v4 did or did not see in training."""
    game_question = arguments[2]
    everything = np.load("production/open_set_v2_20261111/engine_inputs/global_indices.npy")
    selected = set(np.concatenate((
        everything[:120561],
        np.load("production/trunk_style_20261113/inputs/corpus2_global_indices.npy"),
    )).tolist())
    with np.load(f"{RANK}/h3_map.npz") as mapping:
        # pylint: disable-next=no-member  # NpzFile arrays; pylint infers the np.array function.
        row_of_game = dict(zip(mapping["game_key"].tolist(), mapping["row"].tolist()))
    with open(f"{PROBE}/h3_opp.csv", encoding="utf-8", newline="") as source:
        game_ids = [row["game_id"] for row in csv.DictReader(source)]
    seen = torch.tensor([row_of_game[g] in selected for g in game_ids], device=DEVICE).float()
    count = torch.bincount(game_question, minlength=len(truth)).float()
    seen_share = torch.zeros(len(truth), device=DEVICE).index_add_(0, game_question, seen) / count
    crossfit = torch.zeros(len(truth), 10, device=DEVICE)
    for half in (0, 1):
        mask = torch.from_numpy(fold != half).to(DEVICE)
        model = fit(Model(ranks).to(DEVICE), arguments, truth, mask)
        with torch.no_grad():
            crossfit[~mask] = model(*arguments).softmax(1)[~mask]
    base = arguments[3]
    new, old = GAIN[decode(crossfit), truth], GAIN[decode(base), truth]
    print("cross-fit score", round(new.mean().item(), 4), "C1+C2", round(old.mean().item(), 4))
    for label, mask in (
        ("no game seen by v4", seen_share == 0),
        ("some games seen", (seen_share > 0) & (seen_share < 1)),
        ("all games seen", seen_share == 1),
    ):
        print(
            f"  {label:20s} n {int(mask.sum()):4d}  C1+C2 {old[mask].mean().item():.4f}"
            f"  new {new[mask].mean().item():.4f}  delta {(new - old)[mask].mean().item():+.4f}"
        )


def official_inputs():  # pylint: disable=too-many-locals
    """Official v4+v7 opponent similarities, game kinds, question index and base probabilities."""
    # The NpzFile values below are arrays; pylint infers them as the np.array function
    # (no-member, comparison-with-callable, not-an-iterable on the next lines).
    state = torch.load(
        "production/trunk_style_20261113/style_v4/checkpoint.pt",
        map_location=DEVICE, weights_only=False,
    )["state"]
    classes = F.normalize(state["class_weights"].float(), dim=-1)
    with np.load(f"{PROBE}/test_opp_scores.npz") as official:
        # pylint: disable-next=no-member
        vectors = torch.from_numpy(official["vectors"].astype(np.float32)).to(DEVICE)
        question = official["question"]
        own_color, handicap = official["own_color"], official["handicap"]
    test_sims = (
        (F.normalize(vectors, dim=-1) @ classes.T)
        + torch.from_numpy(np.load(f"{PROBE}/test_opp_sims_v7.npy")).float().to(DEVICE)
    ) / 2
    opp_white = own_color == "B"  # pylint: disable=comparison-with-callable
    test_kind = torch.from_numpy(
        # pylint: disable-next=comparison-with-callable
        np.where(handicap == 0, 0, np.where(handicap == 1, np.where(opp_white, 1, 2), 3))
    ).to(DEVICE)
    path = "runtime/competition2_research/experiments/test_prior/results/test_qprob.npz"
    with np.load(path, allow_pickle=True) as cached:
        order = cached["question_id"].tolist()  # pylint: disable=no-member
        test_base = torch.from_numpy(cached["qprob"]).float().to(DEVICE)
    test_base = test_base / test_base.sum(1, keepdim=True)
    index_of = {q: i for i, q in enumerate(order)}
    test_question = torch.from_numpy(
        np.asarray([index_of[q] for q in question])  # pylint: disable=not-an-iterable
    ).to(DEVICE)
    return order, (test_sims, test_kind, test_question, test_base)


def write_official(model) -> None:  # pylint: disable=too-many-locals
    """Applies the fitted model to the 800 official questions and writes the v47 files."""
    order, test_arguments = official_inputs()
    test_kind, test_base = test_arguments[1], test_arguments[3]
    with torch.no_grad():
        probabilities = model(*test_arguments).softmax(1)
    before, after = decode(test_base).cpu().numpy(), decode(probabilities).cpu().numpy()
    report = {
        "fitted_parameters": {
            "temperature": round(model.log_temperature.exp().item(), 4),
            "unknown_share": [round(v, 3) for v in torch.sigmoid(model.unknown).tolist()],
            "base_weight": round(2 * torch.sigmoid(model.base_weight).item(), 3),
        },
        "official_kinds": np.bincount(test_kind.cpu().numpy(), minlength=4).tolist(),
    }
    for name, prefix in (("public", "pub_"), ("private", "priv_")):
        mask = np.asarray([q.startswith(prefix) for q in order])
        diff = after[mask] - before[mask]
        report[name] = {
            "questions": int(mask.sum()),
            "changed": int((diff != 0).sum()), "up": int((diff > 0).sum()),
            "down": int((diff < 0).sum()),
            "changed_by_two_or_more": int((np.abs(diff) >= 2).sum()),
            "histogram_before": np.bincount(before[mask], minlength=10).tolist(),
            "histogram_after": np.bincount(after[mask], minlength=10).tolist(),
            "mean_top_probability_before": round(
                test_base[torch.from_numpy(mask).to(DEVICE)].max(1).values.mean().item(), 3
            ),
            "mean_top_probability_after": round(
                probabilities[torch.from_numpy(mask).to(DEVICE)].max(1).values.mean().item(), 3
            ),
        }
    with open(f"{PROBE}/rank_opponent_identity_v47.csv", "w", encoding="utf-8",
              newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["question_id", "pred_rank"])
        for q, value in zip(order, after):
            writer.writerow([q, RANKS[value]])
    np.savez(f"{PROBE}/official_probabilities_v47.npz", question=np.asarray(order),
             base=test_base.cpu().numpy(), new=probabilities.cpu().numpy())
    print(json.dumps(report, indent=1))


def main() -> None:
    """Leak check, then the fit on all of holdout3 and the official answers."""
    saved = torch.load(f"{PROBE}/h3_sims_v47.pt", weights_only=False)
    # torch builtin; pylint cannot see that it is callable.
    ranks = F.one_hot(  # pylint: disable=not-callable
        torch.from_numpy(saved["player_rank"]), 10
    ).float().to(DEVICE)
    sims = saved["sims"].float().to(DEVICE)
    sims[:, torch.tensor(saved["holdout_classes"], device=DEVICE)] = -5.0
    kind = torch.from_numpy(saved["kind"]).to(DEVICE)
    game_question = saved["game_question"].to(DEVICE)
    truth, fold, base = saved["truth"].to(DEVICE), saved["fold"], saved["base"].to(DEVICE)
    arguments = (sims, kind, game_question, base)

    # ---- 1. leak check
    leak_check(ranks, arguments, truth, fold)

    # ---- 2. official questions
    model = fit(
        Model(ranks).to(DEVICE), arguments, truth,
        torch.ones(len(truth), dtype=torch.bool, device=DEVICE),
    )
    write_official(model)


if __name__ == "__main__":
    main()
