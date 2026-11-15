"""v4 + v7 score ensemble: cohort check (official-like protocol) and new-pool submission.

Reproduces Competition 1 style submission v9 (`submission_style_v9_ens`),
step 5 of 5 (after original_match.py wrote v6 and rich_pipeline.py wrote
the `rerank_v7/` scores). New pools: CSLS over the sum of the v4
(`rerank/`, style_v5_match) and v7 (`rerank_v7/`, style_v7_match) scores;
original pools keep the v6 answers.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.ensemble_v9
"""
import csv
import json

import numpy as np
import torch

from competition1_player_identification.experiments.trunk_style import (
    matching, matching_apply, rerank_apply, train,
)

D = matching.DEVICE
ROOT = train.ROOT
MODELS = (("rerank", "style_v5_match"), ("rerank_v7", "style_v7_match"))
W = (1.0, 0.5, 0.5, 0.0)


def sources(split):
    """Cohort score sources of both models for one split."""
    out = []
    for folder, head in MODELS:
        matching_apply.RERANK = ROOT / folder
        out.append(matching_apply.sources(head, split))
    return out


def cohort_check() -> None:
    """Prints the official-like cohort scores of v4, v7 and v4+v7."""
    for split in ("validation", "final"):
        v4, v7 = sources(split)  # pylint: disable=unbalanced-tuple-unpacking  # one per MODELS
        both = {count: v4[count] + v7[count] for count in v4}
        for label, source, weights in (("v4", v4, W), ("v7", v7, W), ("v4+v7", both, W + W)):
            print(
                split, label, "official-like",
                round(matching_apply.official_like(source, weights, False), 4),
                "with CSLS", round(matching_apply.official_like(source, weights, True), 4),
                "| all 600 asked", round(matching_apply.diag_utility(source, weights)[0], 4),
                flush=True,
            )


def new_pool_answers() -> tuple[dict, dict]:
    """Top-5 answers of the new pools from the summed v4 + v7 scores."""
    answers, report = {}, {}
    for pool in ("pub2", "priv2"):
        names = np.load(ROOT / "rerank" / f"pool_{pool}.npz")
        parts = []
        for folder, _ in MODELS:
            saved = np.load(ROOT / folder / f"pool_scores_{pool}.npz")
            parts.append([
                matching.standardize(torch.from_numpy(saved[key]).to(D))
                for key in ("cosine", "untrained", "trained")
            ])
        single = matching.csls(matching_apply.fused(parts[0], W[:3]))
        final = matching.csls(matching_apply.fused(parts[0] + parts[1], W[:3] + W[:3]))
        order = final.argsort(dim=1, descending=True)[:, :5].cpu().numpy()
        old = single.argsort(dim=1, descending=True)[:, :5].cpu().numpy()
        for question, chosen in zip(names["questions"], order):
            answers[str(question)] = [str(names["players"][i]) for i in chosen]
        report[pool] = {
            "top1_changed_vs_v4_recomputed": float((order[:, 0] != old[:, 0]).mean()),
            "new_top1_in_old_top5": float((old == order[:, :1]).any(1).mean()),
            "top1_repeat_share": 1 - len(np.unique(order[:, 0])) / len(order),
            "old_top1_repeat_share": 1 - len(np.unique(old[:, 0])) / len(old),
            "v4_recomputed_first": [str(names["players"][i]) for i in old[:3, 0]],
        }
    return answers, report


def main() -> None:
    """Cohort check, then writes submission_style_v9_ens on top of v6."""
    cohort_check()
    answers, report = new_pool_answers()
    source = ROOT / "submission_style_v6_orig"
    with open(source / "submission_player.csv", encoding="utf-8", newline="") as file:
        previous = {row["question_id"]: row["top_1"] for row in csv.DictReader(file)}
    for pool in ("pub2", "priv2"):
        names = np.load(ROOT / "rerank" / f"pool_{pool}.npz")
        report[pool]["v6_first"] = [previous[str(q)] for q in names["questions"][:3]]
    written = rerank_apply.write_submission(
        source, ROOT / "submission_style_v9_ens", answers, previous
    )
    report.update({
        "rows": written["rows"], "replaced": written["replaced"],
        "top1_changed_vs_v6": written["top1_changed"], "sha256": written["sha256"],
    })
    (ROOT / "submission_style_v9_ens" / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
