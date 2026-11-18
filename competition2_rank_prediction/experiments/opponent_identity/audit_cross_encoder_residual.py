"""Cross-encoder audit of identity-specific matchmaking residuals.

Follow-up to audit_identity_matchmaking_graph.py. The exact graph premise failed
because the one-million-game rank corpus stores each physical game only once.
This audit asks whether the underlying identity-specific residual hypothesis can
still be tested without trusting one encoder's self-consistency.

Core safeguard:
  * estimate per-opponent residuals with v4 and validate them with v7;
  * estimate with v7 and validate with v4;
  * split by SOURCE PLAYER, not by game, so the asked player never crosses the
    estimation/evaluation halves;
  * compare class-level cross-encoder stability with a within-registered-rank
    permutation null.

The audit is exploratory and trains no neural network. It uses only labelled
training games for residual estimation/validation. Official inputs are read
only for coverage, never for fitting.

Prerequisite token folders are created by run_cross_encoder_residual_audit.sh.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from competition1_player_identification.experiments.trunk_style import apply, train
from competition1_player_identification.experiments.trunk_style import rich_full
from competition2_rank_prediction.experiments.opponent_identity import link_test

TEMPERATURE = 0.04
RANKS = 10
SEED = 20261118


@dataclass
class HalfEstimate:
    """One encoder's residual estimate on one source-player half."""

    delta: np.ndarray
    mass: np.ndarray
    baseline: np.ndarray
    game_residual: np.ndarray


def stable_half(player_id: str) -> int:
    """Deterministic source-player split; all games of one player stay together."""
    return hashlib.sha256(player_id.encode()).digest()[0] & 1


def parse_kind(sgf: str, opponent_color: str) -> int:
    """0 even, 1 HD1 opponent White, 2 HD1 opponent Black, 3 HD>=2."""
    match = re.search(r"HD\[(\d+)\]", sgf)
    if match is None:
        raise ValueError("SGF is missing HD[...]")
    handicap = int(match.group(1))
    if handicap == 0:
        return 0
    if handicap == 1:
        return 1 if opponent_color == "W" else 2
    return 3


def load_sample(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Training opponent sample metadata in its extraction order."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    corpus_row = np.asarray([int(row["corpus_row"]) for row in rows], dtype=np.int64)
    return {
        "game_id": np.asarray([row["game_id"] for row in rows]),
        "player_id": np.asarray([row["player_id"] for row in rows]),
        "color": np.asarray([row["color"] for row in rows]),
        "corpus_row": corpus_row,
        "own_rank": corpus_row // 100000,
        "kind": np.asarray(
            [parse_kind(row["sgf_content"], row["color"]) for row in rows],
            dtype=np.int64,
        ),
        "half": np.asarray([stable_half(row["player_id"]) for row in rows], dtype=np.int64),
    }


def token_data(directory: pathlib.Path, rows: dict[str, np.ndarray]) -> dict:
    """StyleEncoder input from pre-extracted trunk shards."""
    masks = train.load_shards(directory, "masks", 0)
    if len(masks) != len(rows["game_id"]):
        raise ValueError(f"{directory}: token rows {len(masks)} != sample rows {len(rows['game_id'])}")
    record_ids = train.load_shards(directory, "record_ids", 0).astype(str)
    if not np.array_equal(record_ids, rows["game_id"].astype(str)):
        raise ValueError(f"{directory}: record_ids disagree with opp_sample.csv")
    return {
        "trunk": train.load_shards(directory, "tokens", 0),
        "engine": np.zeros((len(masks), masks.shape[1], 42), np.float16),
        "masks": masks,
        "colors": (rows["color"] == "W").astype(np.int64),
        "has_engine": np.zeros(len(masks), bool),
    }


def class_tables(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Registered rank per v47 class and own class per sampled corpus row."""
    _, player_rank, owner_of_row = link_test.gallery()
    own_class = owner_of_row[sample["corpus_row"]].astype(np.int64)
    return player_rank.astype(np.int64), own_class


def use_v4_layout() -> None:
    """The genuine 384-D v4 layout.

    Importing `rich_full` rebinds `train.use_all_moves` to the 960-D v7 layout
    for the whole process, and `apply.load_model` calls `train.use_all_moves()`
    for any `all_moves` checkpoint. Without this, style_v4 is built with a
    1002-wide input projection and its 426-wide checkpoint cannot load.
    """
    train.LAYOUT.length, train.LAYOUT.trunk, train.LAYOUT.engine = 160, 384, 42
    train.LAYOUT.suffix = "_all"


def load_v4() -> tuple:
    """style_v4 model and scaling, built with the 384-D layout it was trained on."""
    patched = train.use_all_moves
    train.use_all_moves = use_v4_layout
    try:
        use_v4_layout()
        model, scaling, _, _ = apply.load_model("style_v4")
    finally:
        train.use_all_moves = patched
    return model, scaling


def encode_v4(
    directory: pathlib.Path,
    rows: dict[str, np.ndarray],
    own_class: np.ndarray,
) -> np.ndarray:
    """Full v4 class-cosine matrix with the source player's own class masked."""
    model, scaling = load_v4()
    # torch Parameter method; pylint cannot infer callability.
    classes = F.normalize(model.class_weights.detach().float(), dim=-1)  # pylint: disable=not-callable
    data = token_data(directory, rows)
    vectors = torch.from_numpy(
        train.encode(model, data, scaling, np.arange(len(rows["game_id"])))
    ).float().to(train.DEVICE)
    sims = vectors @ classes.T
    own = torch.from_numpy(own_class).to(train.DEVICE)
    keep = own >= 0
    sims[torch.arange(len(sims), device=train.DEVICE)[keep], own[keep]] = -5.0
    result = sims.half().cpu().numpy()
    del model, vectors, sims, classes
    torch.cuda.empty_cache()
    return result


def encode_v7(
    directory: pathlib.Path,
    rows: dict[str, np.ndarray],
    own_class: np.ndarray,
) -> np.ndarray:
    """Full v7 class-cosine matrix with the source player's own class masked."""
    rich_full.use_rich()
    model, scaling, _, _ = apply.load_model("style_v7")
    classes = F.normalize(model.class_weights.detach().float(), dim=-1)  # pylint: disable=not-callable
    data = token_data(directory, rows)
    vectors = torch.from_numpy(
        train.encode(model, data, scaling, np.arange(len(rows["game_id"])))
    ).float().to(train.DEVICE)
    vectors = F.normalize(vectors, dim=-1)
    sims = vectors @ classes.T
    own = torch.from_numpy(own_class).to(train.DEVICE)
    keep = own >= 0
    sims[torch.arange(len(sims), device=train.DEVICE)[keep], own[keep]] = -5.0
    result = sims.half().cpu().numpy()
    del model, vectors, sims, classes
    torch.cuda.empty_cache()
    return result


def posterior(sims: np.ndarray, temperature: float) -> np.ndarray:
    """Float32 class posterior; full soft assignment avoids hard top-1 noise."""
    tensor = torch.from_numpy(sims.astype(np.float32)).to(train.DEVICE)
    with torch.no_grad():
        probability = torch.softmax(tensor / temperature, dim=1)
    result = probability.cpu().numpy().astype(np.float32)
    del tensor, probability
    torch.cuda.empty_cache()
    return result


def rank_posterior(probability: np.ndarray, player_rank: np.ndarray) -> np.ndarray:
    """Collapses class posterior to ten opponent registered-rank bins."""
    one_hot = np.eye(RANKS, dtype=np.float32)[player_rank]
    return probability @ one_hot


def fit_rank_baseline(
    probability: np.ndarray,
    own_rank: np.ndarray,
    kind: np.ndarray,
    player_rank: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """E[own rank | opponent registered rank, game kind] on one half."""
    rank_prob = rank_posterior(probability[mask], player_rank)
    y = own_rank[mask].astype(np.float64)
    k = kind[mask]
    baseline = np.zeros((3, RANKS), dtype=np.float64)
    for game_kind in range(3):
        rows = k == game_kind
        for opponent_rank in range(RANKS):
            weights = rank_prob[rows, opponent_rank].astype(np.float64)
            if weights.sum() < 1e-6:
                baseline[game_kind, opponent_rank] = float(opponent_rank)
            else:
                baseline[game_kind, opponent_rank] = float(
                    np.dot(weights, y[rows]) / weights.sum()
                )
    return baseline


def predict_baseline(
    probability: np.ndarray,
    kind: np.ndarray,
    player_rank: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Continuous own-rank prediction from rank-only opponent evidence."""
    rank_prob = rank_posterior(probability, player_rank).astype(np.float64)
    return np.asarray(
        [np.dot(rank_prob[index], baseline[int(kind[index])]) for index in range(len(kind))],
        dtype=np.float64,
    )


def estimate_delta(
    probability: np.ndarray,
    own_rank: np.ndarray,
    kind: np.ndarray,
    player_rank: np.ndarray,
    mask: np.ndarray,
    shrinkage: float,
) -> HalfEstimate:
    """Posterior-weighted class residual beyond registered-rank transition."""
    baseline = fit_rank_baseline(probability, own_rank, kind, player_rank, mask)
    base_prediction = predict_baseline(
        probability[mask],
        kind[mask],
        player_rank,
        baseline,
    )
    residual = own_rank[mask].astype(np.float64) - base_prediction
    responsibilities = probability[mask].astype(np.float64)
    mass = responsibilities.sum(0)
    numerator = responsibilities.T @ residual
    delta = numerator / (mass + shrinkage)
    return HalfEstimate(delta, mass, baseline, residual)


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Pearson and Spearman with finite/variance guards."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def permutation_null(
    left: np.ndarray,
    right: np.ndarray,
    classes: np.ndarray,
    player_rank: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Within-rank identity permutation null for observed class alignment."""
    observed = correlation(left, right)
    if observed["pearson"] is None:
        return {"observed": observed, "permutations": 0}
    rng = np.random.default_rng(seed)
    null_p = np.empty(permutations, dtype=np.float64)
    null_s = np.empty(permutations, dtype=np.float64)
    ranks = player_rank[classes]
    for trial in range(permutations):
        shuffled = right.copy()
        for rank in range(RANKS):
            positions = np.flatnonzero(ranks == rank)
            if len(positions) > 1:
                shuffled[positions] = shuffled[rng.permutation(positions)]
        null_p[trial] = pearsonr(left, shuffled).statistic
        null_s[trial] = spearmanr(left, shuffled).statistic
    return {
        "observed": observed,
        "permutations": permutations,
        "pearson_null_q95": float(np.quantile(null_p, 0.95)),
        "pearson_null_q99": float(np.quantile(null_p, 0.99)),
        "pearson_one_sided_p": float((1 + (null_p >= observed["pearson"]).sum()) / (permutations + 1)),
        "spearman_null_q95": float(np.quantile(null_s, 0.95)),
        "spearman_null_q99": float(np.quantile(null_s, 0.99)),
        "spearman_one_sided_p": float((1 + (null_s >= observed["spearman"]).sum()) / (permutations + 1)),
    }


def rounded_utility(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Contest-like utility from rounded continuous rank predictions."""
    chosen = np.clip(np.rint(prediction), 0, 9).astype(np.int64)
    gap = np.abs(chosen - truth.astype(np.int64))
    return float(np.mean(np.where(gap == 0, 1.0, np.where(gap == 1, math.exp(-1.0), 0.0))))


def cross_prediction(
    estimator: HalfEstimate,
    validator_probability: np.ndarray,
    validator_baseline: np.ndarray,
    own_rank: np.ndarray,
    kind: np.ndarray,
    player_rank: np.ndarray,
    eval_mask: np.ndarray,
    mass_threshold: float,
) -> dict:
    """Apply estimator-encoder deltas through the validator encoder posterior."""
    reliable = estimator.mass >= mass_threshold
    delta = estimator.delta * reliable
    base = predict_baseline(
        validator_probability[eval_mask],
        kind[eval_mask],
        player_rank,
        validator_baseline,
    )
    correction = validator_probability[eval_mask].astype(np.float64) @ delta
    corrected = base + correction
    truth = own_rank[eval_mask].astype(np.float64)
    residual = truth - base
    corr = correlation(correction, residual)
    return {
        "eval_games": int(eval_mask.sum()),
        "reliable_classes": int(reliable.sum()),
        "posterior_mass_on_reliable_classes": float(
            validator_probability[eval_mask][:, reliable].sum(1).mean()
        ) if reliable.any() else 0.0,
        "correction_vs_baseline_residual": corr,
        "mae_baseline": float(np.mean(np.abs(residual))),
        "mae_corrected": float(np.mean(np.abs(truth - corrected))),
        "mae_delta": float(
            np.mean(np.abs(truth - corrected)) - np.mean(np.abs(residual))
        ),
        "utility_baseline": rounded_utility(base, truth),
        "utility_corrected": rounded_utility(corrected, truth),
        "utility_delta": rounded_utility(corrected, truth) - rounded_utility(base, truth),
    }


def cross_direction(
    name: str,
    estimator_probability: np.ndarray,
    validator_probability: np.ndarray,
    sample: dict[str, np.ndarray],
    player_rank: np.ndarray,
    shrinkage: float,
    mass_thresholds: tuple[float, ...],
    permutations: int,
) -> dict:
    """Two source-player folds for one estimator -> validator direction."""
    official_like = sample["kind"] < 3
    folds = []
    for target_half in (0, 1):
        train_mask = official_like & (sample["half"] != target_half)
        eval_mask = official_like & (sample["half"] == target_half)

        estimate_train = estimate_delta(
            estimator_probability,
            sample["own_rank"],
            sample["kind"],
            player_rank,
            train_mask,
            shrinkage,
        )
        # Independent class residual on the held-out half from the validator.
        estimate_eval = estimate_delta(
            validator_probability,
            sample["own_rank"],
            sample["kind"],
            player_rank,
            eval_mask,
            shrinkage,
        )
        validator_baseline = fit_rank_baseline(
            validator_probability,
            sample["own_rank"],
            sample["kind"],
            player_rank,
            train_mask,
        )

        by_mass = {}
        for threshold in mass_thresholds:
            reliable = (
                (estimate_train.mass >= threshold)
                & (estimate_eval.mass >= threshold)
            )
            classes = np.flatnonzero(reliable)
            null = permutation_null(
                estimate_train.delta[classes],
                estimate_eval.delta[classes],
                classes,
                player_rank,
                permutations,
                SEED + target_half + int(threshold * 100),
            )
            prediction = cross_prediction(
                estimate_train,
                validator_probability,
                validator_baseline,
                sample["own_rank"],
                sample["kind"],
                player_rank,
                eval_mask,
                threshold,
            )
            by_mass[str(threshold)] = {
                "cross_encoder_class_stability": {
                    "classes": int(len(classes)),
                    **null,
                },
                "heldout_prediction": prediction,
            }
        folds.append(
            {
                "target_half": target_half,
                "train_games": int(train_mask.sum()),
                "eval_games": int(eval_mask.sum()),
                "by_mass_threshold": by_mass,
            }
        )
    return {"direction": name, "folds": folds}


def load_official_v4_sims(official_scores: pathlib.Path) -> np.ndarray | None:
    """Official v4 class cosines; call before switching the global layout to v7."""
    if not official_scores.exists():
        return None
    with np.load(official_scores) as saved:
        vectors = torch.from_numpy(saved["vectors"].astype(np.float32)).to(train.DEVICE)
    model4, _ = load_v4()
    classes4 = F.normalize(model4.class_weights.detach().float(), dim=-1)  # pylint: disable=not-callable
    sims4 = (vectors @ classes4.T).half().cpu().numpy()
    del model4, vectors, classes4
    torch.cuda.empty_cache()
    return sims4


def official_coverage(
    training_probability4: np.ndarray,
    training_probability7: np.ndarray,
    official_sims4: np.ndarray | None,
    official_v7: pathlib.Path,
    mass_thresholds: tuple[float, ...],
    temperature: float,
) -> dict | None:
    """Official posterior mass on classes with enough training evidence."""
    if official_sims4 is None or not official_v7.exists():
        return None
    # Full-sample mass is only a coverage statistic; no labels from official data.
    mass4 = training_probability4.sum(0)
    mass7 = training_probability7.sum(0)

    sims7 = np.load(official_v7)
    combined = (official_sims4.astype(np.float32) + sims7.astype(np.float32)) / 2.0
    probability = posterior(combined.astype(np.float16), temperature)

    out = {}
    for threshold in mass_thresholds:
        reliable = (mass4 >= threshold) & (mass7 >= threshold)
        per_game = probability[:, reliable].sum(1) if reliable.any() else np.zeros(len(probability))
        out[str(threshold)] = {
            "reliable_classes": int(reliable.sum()),
            "mean_v47_posterior_mass": float(per_game.mean()),
            "games_mass_ge_0_25": float((per_game >= 0.25).mean()),
            "games_mass_ge_0_50": float((per_game >= 0.50).mean()),
        }
    return out


def summarize_gate(report: dict, primary_threshold: float) -> dict:
    """Predeclared conservative go/no-go summary."""
    key = str(primary_threshold)
    correlations = []
    utility_deltas = []
    positive_all = True
    class_counts = []
    significant_checks = 0
    total_checks = 0
    for direction in ("v4_to_v7", "v7_to_v4"):
        for fold in report[direction]["folds"]:
            item = fold["by_mass_threshold"][key]
            stability = item["cross_encoder_class_stability"]
            observed = stability["observed"]
            spearman = observed["spearman"]
            pearson = observed["pearson"]
            class_counts.append(stability["classes"])
            total_checks += 1
            if spearman is not None and pearson is not None:
                correlations.append(0.5 * (float(spearman) + float(pearson)))
                positive_all &= float(spearman) > 0 and float(pearson) > 0
                significant_checks += int(
                    stability.get("pearson_one_sided_p", 1.0) < 0.05
                    and stability.get("spearman_one_sided_p", 1.0) < 0.05
                )
            else:
                positive_all = False
            utility_deltas.append(item["heldout_prediction"]["utility_delta"])
            positive_all &= item["heldout_prediction"]["utility_delta"] >= 0

    mean_corr = float(np.mean(correlations)) if correlations else float("nan")
    mean_utility = float(np.mean(utility_deltas)) if utility_deltas else float("nan")
    official = report.get("official_coverage")
    coverage = None
    if official is not None:
        coverage = official[key]["mean_v47_posterior_mass"]

    pass_gate = bool(
        correlations
        and min(class_counts, default=0) >= 50
        and positive_all
        and mean_corr >= 0.25
        and significant_checks >= 3
        and mean_utility >= 0.003
        and (coverage is None or coverage >= 0.25)
    )
    return {
        "primary_mass_threshold": primary_threshold,
        "minimum_reliable_classes_across_checks": min(class_counts, default=0),
        "mean_cross_encoder_correlation": mean_corr,
        "permutation_significant_checks": significant_checks,
        "total_cross_encoder_checks": total_checks,
        "mean_rounded_utility_delta": mean_utility,
        "official_mean_posterior_mass_on_reliable_classes": coverage,
        "all_four_checks_nonnegative": positive_all,
        "passes_gate": pass_gate,
        "rule": (
            ">=50 reliable classes, mean of Pearson/Spearman across the four "
            "cross-encoder checks >=0.25, >=3/4 checks significant at p<0.05 "
            "against the within-rank permutation null for BOTH correlations, "
            "mean rounded-utility delta >=+0.003, no fold/direction negative, "
            "and official reliable posterior mass >=0.25 when available."
        ),
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Runs the cross-encoder residual audit and writes one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=pathlib.Path,
        default=pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/opp_sample.csv"),
    )
    parser.add_argument("--v4-tokens", type=pathlib.Path, required=True)
    parser.add_argument("--v7-tokens", type=pathlib.Path, required=True)
    parser.add_argument(
        "--official-scores",
        type=pathlib.Path,
        default=pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/test_opp_scores.npz"),
    )
    parser.add_argument(
        "--official-v7",
        type=pathlib.Path,
        default=pathlib.Path("/tmp2/richard/claude_scratch/opp_probe/test_opp_sims_v7.npy"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "identity_matchmaking_graph/cross_encoder_report.json"
        ),
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--shrinkage", type=float, default=2.0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument(
        "--mass-thresholds",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0, 5.0),
    )
    parser.add_argument("--primary-mass-threshold", type=float, default=2.0)
    args = parser.parse_args()

    sample = load_sample(args.sample)
    player_rank, own_class = class_tables(sample)
    if len(player_rank) != 6807:
        print(f"warning: expected 6807 classes, found {len(player_rank)}", flush=True)

    print("encoding v4 class similarities", flush=True)
    sims4 = encode_v4(args.v4_tokens, sample, own_class)
    # Must happen before rich_full.use_rich() in encode_v7 changes the global layout.
    official_sims4 = load_official_v4_sims(args.official_scores)
    print("encoding v7 class similarities", flush=True)
    sims7 = encode_v7(args.v7_tokens, sample, own_class)

    top_agreement = float((sims4.argmax(1) == sims7.argmax(1)).mean())
    print(f"v4/v7 top-class agreement {top_agreement:.4f}", flush=True)
    probability4 = posterior(sims4, args.temperature)
    probability7 = posterior(sims7, args.temperature)

    thresholds = tuple(float(value) for value in args.mass_thresholds)
    report: dict[str, object] = {
        "sample_games": int(len(sample["game_id"])),
        "official_like_games_hd0_hd1": int((sample["kind"] < 3).sum()),
        "source_players": int(len(np.unique(sample["player_id"]))),
        "source_player_half_counts": {
            str(half): int(len(np.unique(sample["player_id"][sample["half"] == half])))
            for half in (0, 1)
        },
        "v4_v7_top_class_agreement": top_agreement,
        "temperature": args.temperature,
        "shrinkage": args.shrinkage,
        "mass_thresholds": list(thresholds),
    }
    report["v4_to_v7"] = cross_direction(
        "v4_to_v7",
        probability4,
        probability7,
        sample,
        player_rank,
        args.shrinkage,
        thresholds,
        args.permutations,
    )
    report["v7_to_v4"] = cross_direction(
        "v7_to_v4",
        probability7,
        probability4,
        sample,
        player_rank,
        args.shrinkage,
        thresholds,
        args.permutations,
    )
    report["official_coverage"] = official_coverage(
        probability4,
        probability7,
        official_sims4,
        args.official_v7,
        thresholds,
        args.temperature,
    )
    report["gate"] = summarize_gate(report, args.primary_mass_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
