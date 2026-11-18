"""Audit: can shared UNKNOWN opponents be found across official rank questions?

The opponent-identity record lists "a joint model over unknown shared opponents"
as estimated at +0.002 to +0.005 and never built. That estimate rests on a
premise this script tests directly against the official inputs: that repeated
opponents can be located among the games whose opponent is *not* identifiable.

Read-only. Trains nothing and reads only cached artifacts:
`test_opp_scores.npz` (v4 style embeddings, questions), `test_opp_sims_v7.npy`
(v7 class cosines), `controls.npz` (absent/present control tops) and the
style_v4 checkpoint.

Run from the repository root:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition2_rank_prediction.experiments.opponent_identity.\
audit_shared_opponents
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch
import torch.nn.functional as F

PROBE = "/tmp2/richard/claude_scratch/opp_probe"
V4_CHECKPOINT = "production/trunk_style_20261113/style_v4/checkpoint.pt"
QUANTILE_LIMIT = 8_000_000
SEED = 20261118


def quantile(values: torch.Tensor, level: float) -> float:
    """torch.quantile, falling back to a fixed subsample past its size limit."""
    if values.numel() <= QUANTILE_LIMIT:
        return float(values.quantile(level))
    generator = torch.Generator(device=values.device).manual_seed(SEED)
    pick = torch.randint(
        values.numel(), (QUANTILE_LIMIT,), generator=generator, device=values.device
    )
    return float(values[pick].quantile(level))


def describe(values: torch.Tensor) -> dict:
    """Quantile summary of a 1-D tensor."""
    if values.numel() == 0:
        return {"n": 0}
    return {
        "n": int(values.numel()),
        "mean": round(float(values.mean()), 4),
        "q50": round(quantile(values, 0.5), 4),
        "q99": round(quantile(values, 0.99), 4),
        "q999": round(quantile(values, 0.999), 4),
        "max": round(float(values.max()), 4),
    }


def load(device: torch.device) -> dict:
    """Official opponent representations, class cosines, questions and controls."""
    with np.load(f"{PROBE}/test_opp_scores.npz") as official:
        vectors = torch.from_numpy(
            official["vectors"].astype(np.float32)  # pylint: disable=no-member
        ).to(device)
        question = official["question"].astype(str)  # pylint: disable=no-member
        top_cached = official["top"]
    state = torch.load(V4_CHECKPOINT, map_location=device, weights_only=False)["state"]
    embedding4 = F.normalize(vectors, dim=-1)
    cosines4 = embedding4 @ F.normalize(
        state["class_weights"].float().to(device), dim=-1
    ).T
    cosines7 = torch.from_numpy(
        np.load(f"{PROBE}/test_opp_sims_v7.npy")
    ).float().to(device)
    # Per-game representations: the v4 embedding, and the v7 class-cosine
    # profile centred so the shared "similar to everyone" component drops out.
    embedding7 = F.normalize(cosines7 - cosines7.mean(1, keepdim=True), dim=-1)
    result = {
        "embedding4": embedding4, "embedding7": embedding7,
        "combined": (cosines4 + cosines7) / 2.0,
        "question": question, "top_cached": top_cached,
    }
    del cosines4, cosines7
    torch.cuda.empty_cache()
    with np.load(f"{PROBE}/controls.npz") as controls:
        result["absent_top"] = controls["a_top"]
        result["present_top"] = controls["p_top"]
    return result


def known_share(data: dict) -> dict:
    """Population mixture estimate of the known-opponent share, and separability."""
    absent, present, official = (
        data["absent_top"], data["present_top"], data["top_cached"]
    )
    rows = {}
    for percentile in (95.0, 99.0, 99.9):
        tau = float(np.percentile(absent, percentile))
        false_rate = float((absent > tau).mean())
        true_rate = float((present > tau).mean())
        observed = float((official > tau).mean())
        rows[str(percentile)] = {
            "threshold": round(tau, 4),
            "present_detection_rate": round(true_rate, 4),
            "official_above": round(observed, 4),
            "mixture_known_share": round(
                (observed - false_rate) / max(true_rate - false_rate, 1e-9), 4
            ),
        }
    return {
        "note": (
            "The known share is a population mixture, not a per-game label. At "
            "the 99th-percentile cut only about 8% of genuinely present games "
            "are flagged, so single games cannot be sorted into known/unknown."
        ),
        "medians": {
            "absent": round(float(np.median(absent)), 4),
            "present": round(float(np.median(present)), 4),
            "official": round(float(np.median(official)), 4),
        },
        "by_absent_percentile": rows,
    }


def masks(question: np.ndarray, confident: torch.Tensor, device: torch.device) -> dict:
    """Pair masks: strictly upper triangular, cross-question, confidence strata."""
    games = len(question)
    codes = torch.from_numpy(np.unique(question, return_inverse=True)[1]).to(device)
    cross = ~(codes[:, None] == codes[None, :])
    upper = torch.triu(torch.ones(games, games, dtype=torch.bool, device=device), 1)
    return {
        "cross_upper": upper & cross,
        "same_question_upper": upper & ~cross,
        "both": confident[:, None] & confident[None, :],
        "one": confident[:, None] ^ confident[None, :],
        "neither": (~confident)[:, None] & (~confident)[None, :],
    }


def components(pairs: list[tuple[int, int]], size: int) -> list[list[int]]:
    """Connected components over an edge list, union-find."""
    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for node in range(size):
        groups[find(node)].append(node)
    return [members for members in groups.values() if len(members) > 1]


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Runs the audit and prints one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    data = load(device)
    question = data["question"]
    games = len(question)
    report = {"step1_known_unknown": known_share(data)}

    similarity = (
        0.5 * (data["embedding4"] @ data["embedding4"].T)
        + 0.5 * (data["embedding7"] @ data["embedding7"].T)
    )
    similarity.fill_diagonal_(-1.0)

    top_value, top_class = data["combined"].max(1)
    tau = float(np.percentile(data["absent_top"], args.percentile))
    confident = top_value > tau
    pair = masks(question, confident, device)
    same_class = top_class[:, None] == top_class[None, :]

    # Positives: cross-question pairs whose top class agrees and both games are
    # confidently identified. Null: the same, where the top classes differ.
    positives = similarity[pair["cross_upper"] & pair["both"] & same_class]
    null = similarity[pair["cross_upper"] & pair["both"] & ~same_class]
    report["step2_similarity"] = {
        "confident_games": int(confident.sum()),
        "same_player_cross_question": describe(positives),
        "different_player_cross_question": describe(null),
        "same_question_contamination": describe(
            similarity[pair["same_question_upper"]]
        ),
        "contamination_note": (
            "Same-question pairs are opponents of one asked player and are "
            "nearly always different people, yet they are far more similar than "
            "cross-question pairs. The opponent-side embedding therefore carries "
            "question-level context, not opponent identity alone, so "
            "same-question pairs are NOT a valid different-player null."
        ),
    }

    report["step3_excess_by_stratum"] = {}
    strata = {
        "both_confident": pair["cross_upper"] & pair["both"],
        "neither_confident": pair["cross_upper"] & pair["neither"],
        "all_cross_question": pair["cross_upper"],
    }
    for false_rate in (0.001, 0.0001):
        cut = quantile(null, 1.0 - false_rate)
        detection = float((positives > cut).float().mean())
        by_stratum: dict[str, dict] = {}
        entry = {
            "threshold": round(cut, 4),
            "same_player_detection_rate": round(detection, 4),
            "by_stratum": by_stratum,
        }
        for name, mask in strata.items():
            values = similarity[mask]
            observed = int((values > cut).sum())
            expected = int(values.numel()) * false_rate
            by_stratum[name] = {
                "pairs": int(values.numel()),
                "observed_above": observed,
                "expected_if_all_different": round(expected, 1),
                "excess": round(observed - expected, 1),
                "implied_repeat_pairs": max(
                    0, round((observed - expected) / max(detection, 1e-9))
                ),
            }
            del values
            torch.cuda.empty_cache()
        report["step3_excess_by_stratum"][str(false_rate)] = entry

    # Detection power degrades on exactly the games a joint model would need.
    reference = quantile(null, 0.999)
    power = {}
    for name in ("both", "one", "neither"):
        values = similarity[pair["cross_upper"] & pair[name] & same_class]
        power[f"{name}_confident"] = {
            "same_class_pairs": int(values.numel()),
            "mean": round(float(values.mean()), 4) if values.numel() else None,
            "above_reference": round(
                float((values > reference).float().mean()), 4
            ) if values.numel() else None,
        }
        del values
        torch.cuda.empty_cache()
    report["step4_detection_power"] = {
        "reference_threshold": round(reference, 4),
        "note": (
            "Top-class agreement is itself unreliable for unidentifiable games, "
            "so these are an upper bound on degradation rather than a clean "
            "power curve. The direction is what matters."
        ),
        "strata": power,
    }

    # What a joint model would actually have to work with.
    strict = quantile(null, 0.9999)
    mask = pair["cross_upper"] & pair["neither"] & (similarity > strict)
    rows, cols = torch.nonzero(mask, as_tuple=True)
    clusters = components(list(zip(rows.tolist(), cols.tolist())), games)
    spans = [len({question[member] for member in cluster}) for cluster in clusters]
    report["step5_joint_model_reach"] = {
        "threshold": round(strict, 4),
        "unidentifiable_games": int((~confident).sum()),
        "linked_cross_question_pairs": int(mask.sum()),
        "clusters": len(clusters),
        "max_questions_in_a_cluster": int(max(spans)) if spans else 0,
        "note": (
            "A cluster spanning hundreds of questions is transitive-closure "
            "noise, not a shared opponent."
        ),
    }

    by_class: dict[int, set[str]] = collections.defaultdict(set)
    for position in torch.nonzero(confident).flatten().tolist():
        by_class[int(top_class[position])].add(question[position])
    spans_known = [len(value) for value in by_class.values()]
    report["step6_known_recurrence_reference"] = {
        "note": (
            "These opponents are identified, so v47 already uses their "
            "registered rank; recurrence here is not what the joint model needs."
        ),
        "distinct_opponents": len(by_class),
        "opponents_in_2plus_questions": sum(1 for s in spans_known if s >= 2),
        "share_in_2plus": round(
            sum(1 for s in spans_known if s >= 2) / max(len(spans_known), 1), 4
        ),
    }

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
