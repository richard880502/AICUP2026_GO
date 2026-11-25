# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Build a question-level feature table for hard-boundary analysis.

The table is intentionally diagnostic: no new rank model is trained. It reuses
the existing holdout3 C1+C2 probabilities and v4+v7 opponent-identity cache,
re-fits v47 cross-fold exactly as production does, and exports interpretable
question-level summaries.

Optional extra NPZ files can add new feature families (for example a future
multi-skill HumanSL likelihood profile) without changing the atlas analysis.

Run from repository root:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      competition2_rank_prediction/c1/.venv/bin/python -m \
      competition2_rank_prediction.experiments.hard_boundary_atlas.build_features
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

RANK_NAMES = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
GAIN = np.eye(10, dtype=np.float32)
GAIN += np.exp(-1.0) * (
    np.eye(10, k=1, dtype=np.float32) + np.eye(10, k=-1, dtype=np.float32)
)


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-instance-attributes
class V47(torch.nn.Module):
    """Device-agnostic copy of the final opponent-identity model."""

    def __init__(self, ranks: torch.Tensor):
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
        axis = torch.arange(10)
        self.register_buffer(
            "offset_index",
            (axis[None] - axis[:, None]).clamp(-3, 3).long() + 3,
        )

    def forward(
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        game_question: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """Question logits."""
        questions = len(base)
        usable = kind < 3
        posterior = (scores / self.log_temperature.exp()).softmax(1) @ self.ranks
        unknown = torch.sigmoid(self.unknown)[kind.clamp(max=2)][:, None]
        posterior = (1 - unknown) * posterior + unknown * self.prior.softmax(0)
        transition = self.offsets[:, self.offset_index].log_softmax(2)
        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None]
            + transition[kind.clamp(max=2)],
            dim=1,
        )
        per_game = per_game * usable[:, None]
        total = torch.zeros(
            questions, 10, device=scores.device
        ).index_add_(0, game_question, per_game)
        used = torch.zeros(
            questions, device=scores.device
        ).index_add_(0, game_question, usable.float()).clamp(min=1)
        logits = self.log_alpha.exp() * total / used[:, None] ** self.gamma + self.bias
        return logits + torch.sigmoid(self.base_weight) * 2 * base.clamp(min=1e-9).log()


def fit(
    model: V47,
    arguments: tuple[torch.Tensor, ...],
    truth: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
) -> V47:
    """Same lightweight Adam fit as h3_final_v47."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(model(*arguments)[mask], truth[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def probability_summaries(prefix: str, probability: np.ndarray) -> dict[str, np.ndarray]:
    """Interpretable summaries of one 10-class question probability vector."""
    p = np.clip(probability.astype(np.float64), 1e-9, 1)
    expected = p @ np.arange(10, dtype=np.float64)
    entropy = -(p * np.log(p)).sum(1)
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    top = ordered[:, -1]
    return {
        f"{prefix}__expected_rank": expected,
        f"{prefix}__entropy": entropy,
        f"{prefix}__top_probability": top,
        f"{prefix}__top_margin": margin,
    }


def aggregate(
    values: np.ndarray,
    game_question: np.ndarray,
    questions: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Mean/std/q25/q75 of one scalar game-level feature per question."""
    out = {
        f"{prefix}_mean": np.zeros(questions, np.float64),
        f"{prefix}_std": np.zeros(questions, np.float64),
        f"{prefix}_q25": np.zeros(questions, np.float64),
        f"{prefix}_q75": np.zeros(questions, np.float64),
    }
    for q in range(questions):
        x = values[game_question == q]
        # len() is intentional: the operand may be a numpy array.
        # pylint: disable-next=use-implicit-booleaness-not-len
        if not len(x):
            continue
        out[f"{prefix}_mean"][q] = float(np.mean(x))
        out[f"{prefix}_std"][q] = float(np.std(x))
        out[f"{prefix}_q25"][q] = float(np.quantile(x, 0.25))
        out[f"{prefix}_q75"][q] = float(np.quantile(x, 0.75))
    return out


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def game_features(
    scores: torch.Tensor,
    kind: torch.Tensor,
    game_question: torch.Tensor,
    model: V47,
    ranks: torch.Tensor,
    question_mask: np.ndarray,
    batch: int = 1024,
) -> dict[str, np.ndarray]:
    """Opponent-identity/rank summaries for games belonging to one target half."""
    device = scores.device
    gq = game_question.cpu().numpy()
    selected_questions = torch.from_numpy(question_mask).to(device)
    selected_games = selected_questions[game_question]
    indices = torch.nonzero(selected_games).flatten()
    rows: dict[str, list[np.ndarray]] = {
        "game": [],
        "top_similarity": [],
        "top_identity_probability": [],
        "identity_entropy": [],
        "expected_opponent_rank": [],
        "opponent_rank_std": [],
        "opponent_rank_entropy": [],
        "top_opponent_rank_probability": [],
    }
    rank_axis = torch.arange(10, device=device, dtype=torch.float32)
    prior = model.prior.softmax(0)
    for start in range(0, len(indices), batch):
        idx = indices[start : start + batch]
        s = scores[idx]
        class_prob = (s / model.log_temperature.exp()).softmax(1)
        rank_prob = class_prob @ ranks
        k = kind[idx]
        unknown = torch.sigmoid(model.unknown)[k.clamp(max=2)][:, None]
        rank_prob = (1 - unknown) * rank_prob + unknown * prior
        exp_rank = rank_prob @ rank_axis
        # Per-game variance sum_k p_k (k - mu)^2. A matrix product here would
        # fail for most batch sizes and silently mix games when batch == 10.
        variance = (rank_prob * (rank_axis[None] - exp_rank[:, None]).square()).sum(1)
        rows["game"].append(idx.cpu().numpy())
        rows["top_similarity"].append(s.max(1).values.detach().cpu().numpy())
        rows["top_identity_probability"].append(
            class_prob.max(1).values.detach().cpu().numpy()
        )
        rows["identity_entropy"].append(
            (-(class_prob.clamp_min(1e-12) * class_prob.clamp_min(1e-12).log()).sum(1))
            .detach().cpu().numpy()
        )
        rows["expected_opponent_rank"].append(exp_rank.detach().cpu().numpy())
        rows["opponent_rank_std"].append(variance.sqrt().detach().cpu().numpy())
        rows["opponent_rank_entropy"].append(
            (-(rank_prob.clamp_min(1e-12) * rank_prob.clamp_min(1e-12).log()).sum(1))
            .detach().cpu().numpy()
        )
        rows["top_opponent_rank_probability"].append(
            rank_prob.max(1).values.detach().cpu().numpy()
        )

    games = np.concatenate(rows.pop("game")) if rows["top_similarity"] else np.zeros(0, int)
    result: dict[str, np.ndarray] = {}
    for name, parts in rows.items():
        values = np.concatenate(parts) if parts else np.zeros(0)
        full = np.full(len(gq), np.nan, np.float64)
        full[games] = values
        result[name] = full
    return result


def decode(probability: np.ndarray) -> np.ndarray:
    """Expected official-utility decoder."""
    return (probability @ GAIN).argmax(1)


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals,too-many-branches
def add_extra_npz(
    feature_map: dict[str, np.ndarray],
    order: list[str],
    paths: Iterable[pathlib.Path],
) -> None:
    """Align optional numeric question-level features by question id."""
    target = {qid: i for i, qid in enumerate(order)}
    reserved = {
        "question_id", "qid", "player_id", "truth", "label", "rank", "fold",
        "feature_names",
    }
    for path in paths:
        with np.load(path, allow_pickle=True) as saved:
            id_key = next(
                (key for key in ("question_id", "qid", "player_id") if key in saved),
                None,
            )
            if id_key is None:
                raise ValueError(f"{path}: needs question_id/qid/player_id")
            ids = saved[id_key].astype(str).tolist()
            positions = np.asarray([target.get(value, -1) for value in ids])
            keep = positions >= 0
            prefix = path.stem
            if "features" in saved:
                matrix = np.asarray(saved["features"])
                if matrix.ndim != 2 or len(matrix) != len(ids):
                    raise ValueError(f"{path}: features must be [questions, features]")
                if "feature_names" in saved:
                    # pylint false positive: numpy/scipy objects are not inferred.
                    # pylint: disable-next=no-member
                    names = saved["feature_names"].astype(str).tolist()
                else:
                    names = [f"f{i}" for i in range(matrix.shape[1])]
                for column, name in enumerate(names):
                    values = np.full(len(order), np.nan, np.float64)
                    values[positions[keep]] = matrix[keep, column]
                    feature_map[f"{prefix}__{name}"] = values
            for key in saved.files:
                if key in reserved or key == "features":
                    continue
                value = np.asarray(saved[key])
                if (
                    value.ndim == 0
                    or not np.issubdtype(value.dtype, np.number)
                    or len(value) != len(ids)
                ):
                    continue
                if value.ndim == 1:
                    aligned = np.full(len(order), np.nan, np.float64)
                    aligned[positions[keep]] = value[keep]
                    feature_map[f"{prefix}__{key}"] = aligned
                elif value.ndim == 2 and value.shape[1] <= 64:
                    for column in range(value.shape[1]):
                        aligned = np.full(len(order), np.nan, np.float64)
                        aligned[positions[keep]] = value[keep, column]
                        feature_map[f"{prefix}__{key}_{column:02d}"] = aligned


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Builds the feature table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=pathlib.Path,
        default=pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--extra-npz", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/hard_boundary_atlas/"
            "feature_table.npz"
        ),
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    saved = torch.load(args.state, map_location="cpu", weights_only=False)
    order = [str(value) for value in saved["order"]]
    truth = saved["truth"].numpy().astype(np.int64)
    fold = np.asarray(saved["fold"], dtype=np.int64)
    base = saved["base"].numpy().astype(np.float64)
    base /= base.sum(1, keepdims=True)
    kind_np = np.asarray(saved["kind"], dtype=np.int64)
    game_question_np = saved["game_question"].numpy().astype(np.int64)

    # pylint false positive: torch callables are not inferred.
    # pylint: disable-next=not-callable
    ranks = F.one_hot(
        torch.from_numpy(np.asarray(saved["player_rank"], dtype=np.int64)), 10
    ).float().to(device)
    scores = saved["sims"].float().to(device)
    holdout_classes = torch.tensor(saved["holdout_classes"], device=device)
    scores[:, holdout_classes] = -5.0
    kind = torch.from_numpy(kind_np).to(device)
    game_question = saved["game_question"].to(device)
    truth_t = saved["truth"].to(device)
    base_t = saved["base"].float().to(device)
    arguments = (scores, kind, game_question, base_t)

    v47 = np.zeros_like(base)
    per_game = {
        name: np.full(len(kind_np), np.nan, np.float64)
        for name in (
            "top_similarity", "top_identity_probability", "identity_entropy",
            "expected_opponent_rank", "opponent_rank_std",
            "opponent_rank_entropy", "top_opponent_rank_probability",
        )
    }
    fit_report = {}
    for target_half in (0, 1):
        train_mask_np = fold != target_half
        train_mask = torch.from_numpy(train_mask_np).to(device)
        model = fit(V47(ranks).to(device), arguments, truth_t, train_mask, args.steps)
        target = fold == target_half
        with torch.no_grad():
            v47[target] = model(*arguments).softmax(1)[
                torch.from_numpy(target).to(device)
            ].cpu().numpy()
        gf = game_features(
            scores, kind, game_question, model, ranks, target
        )
        target_games = target[game_question_np]
        for name in per_game:
            per_game[name][target_games] = gf[name][target_games]
        fit_report[str(target_half)] = {
            "temperature": float(model.log_temperature.exp().item()),
            "unknown_share": torch.sigmoid(model.unknown).detach().cpu().tolist(),
            "base_weight": float(2 * torch.sigmoid(model.base_weight).item()),
        }
        print("finished target half", target_half, fit_report[str(target_half)], flush=True)

    feature_map: dict[str, np.ndarray] = {}
    feature_map.update(probability_summaries("base", base))
    feature_map.update(probability_summaries("v47", v47))
    feature_map["v47__expected_rank_shift_vs_base"] = (
        feature_map["v47__expected_rank"] - feature_map["base__expected_rank"]
    )

    questions = len(order)
    game_count = np.bincount(game_question_np, minlength=questions)
    feature_map["structure__game_count"] = game_count.astype(np.float64)
    for game_kind, label in enumerate(("even", "hd1_opp_white", "hd1_opp_black", "hd2plus")):
        count = np.bincount(
            game_question_np[kind_np == game_kind], minlength=questions
        )
        feature_map[f"structure__{label}_fraction"] = count / np.maximum(game_count, 1)

    for name, values in per_game.items():
        feature_map.update(
            aggregate(
                values,
                game_question_np,
                questions,
                f"opponent__{name}",
            )
        )

    # Distribution of inferred opponent ranks across games, useful for matchmaking context.
    expected = per_game["expected_opponent_rank"]
    for rank in range(10):
        closeness = np.exp(-0.5 * (expected - rank) ** 2)
        feature_map[f"opponent__soft_rank_{rank}_mass"] = np.asarray(
            [
                np.nanmean(closeness[game_question_np == q])
                for q in range(questions)
            ]
        )

    add_extra_npz(feature_map, order, args.extra_npz)

    names = sorted(feature_map)
    matrix = np.stack([feature_map[name] for name in names], axis=1)
    # Median-fill only for the exported diagnostic table; missingness itself is
    # not allowed to become a label feature.
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        question_id=np.asarray(order),
        truth=truth,
        fold=fold,
        base_prob=base.astype(np.float32),
        v47_prob=v47.astype(np.float32),
        feature_names=np.asarray(names),
        features=matrix.astype(np.float32),
    )
    metadata = {
        "questions": questions,
        "games": int(len(kind_np)),
        "features": len(names),
        "rank_names": list(RANK_NAMES),
        "fit": fit_report,
        "v47_score": float(
            np.mean(
                np.where(
                    np.abs(decode(v47) - truth) == 0,
                    1.0,
                    np.where(
                        np.abs(decode(v47) - truth) == 1,
                        np.exp(-1.0),
                        0.0,
                    ),
                )
            )
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
