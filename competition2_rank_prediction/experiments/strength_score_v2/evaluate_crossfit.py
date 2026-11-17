"""Evaluate two Strength Score folds and fuse the current opponent baseline."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
ADJACENT = float(np.exp(-1.0))
GAIN = np.eye(10, dtype=np.float32) + ADJACENT * (
    np.eye(10, k=1, dtype=np.float32)
    + np.eye(10, k=-1, dtype=np.float32)
)
WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0)


def decode(probabilities: np.ndarray) -> np.ndarray:
    return (probabilities @ GAIN).argmax(axis=1)


def score(probabilities: np.ndarray, truth: np.ndarray) -> float:
    pred = decode(probabilities)
    gap = np.abs(pred - truth)
    return float(
        np.mean(
            np.where(
                gap == 0,
                1.0,
                np.where(gap == 1, ADJACENT, 0.0),
            )
        )
    )


def load_strength(results: pathlib.Path) -> dict[str, np.ndarray]:
    parts = []
    for fold in (0, 1):
        with np.load(results / f"fold{fold}" / "predictions.npz") as saved:
            parts.append({key: np.asarray(saved[key]) for key in saved.files})
    return {
        key: np.concatenate([part[key] for part in parts])
        for key in parts[0]
    }


def adjacent_auc(values: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    result = {}
    for lower in range(9):
        mask = (truth == lower) | (truth == lower + 1)
        y = (truth[mask] == lower + 1).astype(np.int64)
        result[f"{LEVELS[lower]}/{LEVELS[lower + 1]}"] = float(
            roc_auc_score(y, values[mask])
        )
    return result


class OpponentIdentityModel(torch.nn.Module):
    """Matches experiments/opponent_identity/h3_final_v47.py."""

    def __init__(self, ranks: torch.Tensor, device: torch.device):
        super().__init__()
        self.ranks = ranks
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(np.log(0.04))
        )
        self.unknown = torch.nn.Parameter(torch.full((3,), -1.0))
        self.prior = torch.nn.Parameter(torch.zeros(10))
        self.offsets = torch.nn.Parameter(torch.zeros(3, 7))
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))
        self.bias = torch.nn.Parameter(torch.zeros(10))
        self.base_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "offset_index",
            (
                torch.arange(10, device=device)[None]
                - torch.arange(10, device=device)[:, None]
            ).clamp(-3, 3)
            + 3,
        )

    def forward(
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        game_question: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        usable = kind < 3
        posterior = (
            scores / self.log_temperature.exp()
        ).softmax(1) @ self.ranks
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (
            (1 - unknown) * posterior
            + unknown * self.prior.softmax(0)
        )
        transition = self.offsets[:, self.offset_index].log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None]
            + transition[kind.clamp(max=2)],
            dim=1,
        )
        per_game = per_game * usable[:, None]
        questions = len(base)
        total = torch.zeros(
            questions, 10, device=base.device
        ).index_add_(0, game_question, per_game)
        used = torch.zeros(
            questions, device=base.device
        ).index_add_(
            0, game_question, usable.float()
        ).clamp(min=1)
        logits = (
            self.log_alpha.exp()
            * total
            / used[:, None] ** self.gamma
            + self.bias
        )
        return (
            logits
            + torch.sigmoid(self.base_weight)
            * 2
            * base.clamp(min=1e-9).log()
        )


def fit_opponent(
    model: OpponentIdentityModel,
    arguments: tuple[torch.Tensor, ...],
    truth: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
) -> OpponentIdentityModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def opponent_crossfit(
    path: pathlib.Path,
    device: torch.device,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    player_rank = np.asarray(saved["player_rank"], dtype=np.int64)
    ranks = F.one_hot(
        torch.from_numpy(player_rank).to(device),
        num_classes=10,
    ).float()
    sims = saved["sims"].float().to(device)
    holdout_classes = torch.as_tensor(
        saved["holdout_classes"],
        dtype=torch.long,
        device=device,
    )
    sims[:, holdout_classes] = -5.0
    kind = torch.as_tensor(
        saved["kind"], dtype=torch.long, device=device
    )
    game_question = saved["game_question"].long().to(device)
    truth = saved["truth"].long().to(device)
    base = saved["base"].float().to(device)
    fold = np.asarray(saved["fold"], dtype=np.int64)
    order = np.asarray(saved["order"]).astype(str)
    arguments = (sims, kind, game_question, base)

    output = torch.zeros(len(truth), 10, device=device)
    for target_fold in (0, 1):
        train_mask = torch.from_numpy(
            fold != target_fold
        ).to(device)
        model = fit_opponent(
            OpponentIdentityModel(ranks, device).to(device),
            arguments,
            truth,
            train_mask,
            steps,
        )
        with torch.no_grad():
            test_mask = ~train_mask
            output[test_mask] = model(*arguments).softmax(1)[test_mask]
    return (
        order,
        truth.cpu().numpy(),
        fold,
        output.cpu().numpy(),
    )


def fuse(
    base: np.ndarray,
    strength: np.ndarray,
    weight: float,
) -> np.ndarray:
    logits = (
        np.log(np.clip(base, 1e-8, 1.0))
        + weight * np.log(np.clip(strength, 1e-8, 1.0))
    )
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def select_crossfit_weights(
    base: np.ndarray,
    strength: np.ndarray,
    truth: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.empty_like(base)
    chosen: dict[str, float] = {}
    for target_fold in (0, 1):
        tune = fold != target_fold
        trials = {
            weight: score(
                fuse(base[tune], strength[tune], weight),
                truth[tune],
            )
            for weight in WEIGHTS
        }
        best = max(trials, key=trials.get)
        test = fold == target_fold
        output[test] = fuse(base[test], strength[test], best)
        chosen[f"fold{target_fold}"] = float(best)
    return output, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--opponent-state", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opponent-steps", type=int, default=300)
    parser.add_argument("--gate", type=float, default=0.008)
    args = parser.parse_args()

    strength = load_strength(args.results)
    qids = strength["question_id"].astype(str)
    truth = strength["truth"].astype(np.int64)
    fold = strength["fold"].astype(np.int64)
    strength_prob = strength["probabilities"].astype(np.float64)
    raw = strength["strength"].astype(np.float64)
    strength_pred = decode(strength_prob)

    report: dict[str, object] = {
        "questions": int(len(qids)),
        "strength_only": {
            "score": score(strength_prob, truth),
            "exact": float((strength_pred == truth).mean()),
            "spearman": float(
                spearmanr(raw, truth).statistic
            ),
            "adjacent_auc": adjacent_auc(raw, truth),
            "rank_mean_strength": {
                LEVELS[label]: float(raw[truth == label].mean())
                for label in range(10)
                if np.any(truth == label)
            },
        },
    }

    if args.opponent_state is not None:
        base_qids, base_truth, base_fold, base_prob = opponent_crossfit(
            args.opponent_state,
            torch.device(args.device),
            args.opponent_steps,
        )
        position = {
            qid: index for index, qid in enumerate(base_qids)
        }
        missing = [qid for qid in qids if qid not in position]
        if missing:
            raise ValueError(
                f"{len(missing)} strength questions missing from opponent state"
            )
        order = np.asarray(
            [position[qid] for qid in qids],
            dtype=np.int64,
        )
        base = base_prob[order]
        if not np.array_equal(base_truth[order], truth):
            raise ValueError("truth arrays disagree")
        if not np.array_equal(base_fold[order], fold):
            raise ValueError("fold arrays disagree")

        fused, weights = select_crossfit_weights(
            base, strength_prob, truth, fold
        )
        baseline_score = score(base, truth)
        fused_score = score(fused, truth)
        expected_base = (
            base * np.arange(10, dtype=np.float64)[None, :]
        ).sum(axis=1)
        report["current_baseline"] = {
            "score": baseline_score,
            "exact": float((decode(base) == truth).mean()),
        }
        report["independence"] = {
            "pearson_strength_vs_baseline_expected_rank": float(
                pearsonr(raw, expected_base).statistic
            ),
            "spearman_strength_vs_baseline_expected_rank": float(
                spearmanr(raw, expected_base).statistic
            ),
        }
        report["fusion"] = {
            "weights_selected_on_opposite_fold": weights,
            "score": fused_score,
            "exact": float((decode(fused) == truth).mean()),
            "delta_vs_current_baseline": (
                fused_score - baseline_score
            ),
            "gate": args.gate,
            "passes_gate": bool(
                fused_score - baseline_score >= args.gate
            ),
        }

    output = args.results / "crossfit_report.json"
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
