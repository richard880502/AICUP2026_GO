"""Exports completed, verified experiment findings and aggregate figures.

Run after final_evaluate. Only aggregate reports and a figure are copied into
Git; licensed records, game predictions and model weights remain in production.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from competition1_player_identification.experiments.open_set_v2 import plots

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    final_evaluate,
)
from competition1_player_identification.experiments.open_set_v2 import fusion
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)

DESTINATION = pathlib.Path(__file__).parent


def figure(root: pathlib.Path, reports: dict, final: dict) -> None:
    """Plots validation ablations separately from the locked final comparison."""
    canvas, axes = plt.subplots(1, 3, figsize=(17, 5))
    names = list(reports)
    values = [reports[name]["score"] for name in names]
    axes[0].barh(names, values)
    axes[0].set_title("Validation: selection only")
    axes[0].set_xlabel("Exponential Top-5 score")
    axes[0].tick_params(axis="y", labelsize=8)
    for name in ("baseline", "proposed"):
        by_count = final[name]["by_query_count"]
        plots.query_curve(axes[1], by_count, name)
    axes[1].set_title("Final: 600 previously unseen players")
    axes[1].legend()
    points = np.asarray(
        [final[name]["score"] for name in ("baseline", "proposed")]
    )
    intervals = np.asarray(
        [
            final[name]["player_bootstrap_95ci"]
            for name in ("baseline", "proposed")
        ]
    )
    errors = np.stack(
        (points - intervals[:, 0], intervals[:, 1] - points)
    ).clip(min=0)
    axes[2].bar(("baseline", "proposed"), points, yerr=errors, capsize=5)
    axes[2].set_title("Final mean + player-bootstrap 95% interval")
    axes[2].set_ylabel("Exponential Top-5 score")
    for index, point in enumerate(points):
        axes[2].text(index, point + 0.02, f"{point:.4f}", ha="center")
    canvas.suptitle(
        "Open-set v2: 600 candidates, 10 references, mixed colors; "
        "configuration locked before final"
    )
    canvas.tight_layout()
    destination = DESTINATION / "figures"
    destination.mkdir(exist_ok=True)
    canvas.savefig(destination / "completed_summary.svg")
    canvas.savefig(root / "completed_summary.png", dpi=160)
    plt.close(canvas)


def final_query_rows(results: dict) -> list[str]:
    """Formats the fixed final query-count comparison."""
    lines = []
    for count in evaluation.COUNTS:
        row = results["proposed"]["by_query_count"][str(count)]
        base = results["baseline"]["by_query_count"][str(count)]
        lines.append(
            f"| {count} | {base['score']:.6f} | {row['score']:.6f} "
            f"| {row['top1']:.2%} | {row['top5']:.2%} |"
        )
    return lines


def markdown(context: tuple) -> str:
    """Builds the complete record without extrapolating to a leaderboard score."""
    root, configuration, record, reports = context
    results = record["results"]
    delta = results["paired_delta"]
    interval = delta["player_bootstrap_95ci"]
    matching = json.loads((root / "matching/completed.json").read_text())
    rank = json.loads((root / "rank_engine/completed.json").read_text())
    lines = [
        "# Completed mixed-color open-set experiment",
        "",
        "All five planned directions were evaluated. These are training-only "
        "proxy results, **not a new Public score**. The production Public "
        "submission remains **0.5236602**, and Private is unknown.",
        "",
        "![Validation and locked final comparison](figures/completed_summary.svg)",
        "",
        "## Fixed protocol and final result",
        "",
        "Training: 6,807 players after reserving separate 600-player "
        "validation and final cohorts. Player and full move-content hash "
        "overlaps are zero. Each evaluation pool has 600 candidates, ten "
        "references each, and 2,400 nested queries balanced across 1/2/3/5 "
        "games. Confidence intervals bootstrap players, not questions.",
        "",
        "The 120,561-game engine cache contains up to sixteen games per "
        "training player plus both evaluation cohorts. Extracting final "
        "features did not fit any model or access final retrieval outcomes. "
        "All preprocessing, class supervision and pair-model fitting used "
        "training players. Validation selected models and fusion. Serialized "
        "inference replay passed before the configuration was sealed.",
        "",
        "| Locked final method | Exponential Top-5 | Player-bootstrap 95% CI |",
        "| --- | ---: | --- |",
    ]
    for name in ("baseline", "proposed"):
        report = results[name]
        bounds = report["player_bootstrap_95ci"]
        lines.append(
            f"| {name} | {report['score']:.6f} | [{bounds[0]:.6f}, {bounds[1]:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Paired difference: **{delta['delta']:+.6f}**, 95% interval "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}].",
            "",
            "The baseline is the predeclared training-normalized, Fisher-weighted "
            "rich-profile centroid method. The proposed recipe is selected "
            "entirely before final. Neither result is the historical black-only "
            "400-way / 100-reference proxy or the mixed-pool official leaderboard.",
            "",
            "| Final query games | Baseline score | Proposed score "
            "| Proposed Top1 | Proposed Top5 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(final_query_rows(results))
    lines.extend(
        [
            "",
            "## Validation ablations",
            "",
            "These results were used for selection, so their gains and intervals "
            "are exploratory; the fixed final comparison above is the confirmation.",
            "",
            "| Source / selected aggregation | Validation score |",
            "| --- | ---: |",
        ]
    )
    for name, report in reports.items():
        lines.append(f"| {name} | {report['score']:.6f} |")
    lines.extend(
        [
            "",
            "The two style encoders used the established 9,198-dimensional "
            "KataGo/HumanSL/ownership profiles, 512D and 1024D normalized outputs, "
            "32 epochs of 256 player episodes, ArcFace + SupCon + episodic "
            "matching. Tiny training libraries use distinct available games; "
            "singletons contribute ArcFace only. Final references always number ten.",
            "",
            "Pair fitting freezes both encoders, mines natural Top-20 from "
            "synthetic 600-way training pools, and compares first-branch, "
            "second-branch and both-branch local matching features. Stage-one "
            "misses are recorded and never forced into the Top-20. Selected "
            f"family: `{matching['selected_branch']}`, blend "
            f"{matching['selection']['blend']:.2f}.",
            "",
            "| Pair feature family | Dimensions | Validation score | Blend |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("a", "b", "both"):
        report = json.loads(
            (root / f"matching/ablation_{name}.json").read_text()
        )
        lines.append(
            f"| {name} | {report['feature_dimensions']} "
            f"| {report['validation']['score']:.6f} "
            f"| {report['selection']['blend']:.2f} |"
        )
    lines.extend(
        [
            "",
            "A fresh C2 raw-move ordinal rank control and a stronger three-seed "
            "C2 decision-Transformer rank prior were tested. The latter adapts "
            "the 128D / three-layer ordinal architecture to 64x41 cached "
            "decisions. It is not the full C1+C2 rank ensemble. Both exclude "
            "style-validation/final identities; 300 inner training players "
            "select and calibrate rank models. No hard candidate pruning is used.",
            f"Engine-rank inner NLL: {rank['inner_rank_nll']:.6f}; "
            f"temperature: {rank['temperature']:.2f}.",
            "",
            "## Locked fusion",
            "",
            "```json",
            json.dumps(configuration["weights"], indent=2),
            "```",
            "",
            "## Findings and next improvements",
            "",
            "- Per-game engine features and cross-game matching have separate "
            "ablation records. Do not attribute the complete fused gain to the "
            "reranker alone. A zero-strength reranker remained a valid selection.",
            "- Exact symmetry-invariant opening-prefix matching complements "
            "coarse statistics. The first raw sequence control was weak at its "
            "eight-epoch budget; this does not reject stronger or board-conditioned "
            "sequence models in general.",
            "- Query-count and query-color subgroup results are stored in the "
            "final JSON. Any remaining one-game weakness should first be "
            "addressed through per-game representation and Stage1 Recall@20; "
            "a Top-20 reranker cannot recover a player outside its pool.",
            "- Future priorities: rank-matched hard negatives, explicitly "
            "cross-color positives, and stronger calibrated rank models. Evaluate "
            "changes on a new player-disjoint cohort, with encoders retrained to "
            "exclude it. Do not tune further on the now-reported final players.",
            "",
            "## Generalization limits and reproducibility",
            "",
            "The held-out population requires at least fifteen games and five "
            "in each color; this activity/color restriction and nested-query "
            "sampling differ from the real leaderboard. Strict game-content "
            "deduplication also removes shared-opponent records. This result "
            "confirms only the fixed sampled proxy, not an official improvement.",
            "",
            "207,390 source games have nonzero HD. The inherited engine "
            "representation is a moves-only Japanese 6.5-komi proxy; initial "
            "stones and actual komi are not guessed. Representation artifacts "
            "and source distribution shift remain limitations.",
            "",
            "Artifacts: `production/open_set_v2_20261111/` contains audits, "
            "source/model hashes, normalization, complete histories, pair "
            "ablations, validation replay, locked_configuration.json, final "
            "scores and replayable final_results.json. Source, aggregate "
            "documentation and this figure are tracked; data and weights are not.",
            "",
            "No new competition upload is claimed, and the 0.5236602 submission "
            "was not replaced. Python checks and invariant tests must pass before "
            "commit. Existing local Git hooks are used; the original Redmine "
            "infra document was unavailable in this session.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Exports only fully verified completed numerical results."""
    root = dataset.ROOT
    configuration = json.loads((root / "locked_configuration.json").read_text())
    record = final_evaluate.verify_completed(root, configuration)
    sources, _ = fusion.load_sources(root)
    reports = {
        name: evaluation.summarize(scores) for name, scores in sources.items()
    }
    reports["locked_fusion"] = configuration["validation"]
    figure(root, reports, record["results"])
    (DESTINATION / "RESULTS.md").write_text(
        markdown((root, configuration, record, reports)), encoding="utf-8"
    )
    artifacts.write_report(
        root / "experiment_complete.json",
        {
            "status": "complete",
            "configuration_sha256": record["configuration_sha256"],
            "final_prediction_sha256": record["prediction_sha256"],
            "final_results": record["results"],
            "record": str(DESTINATION / "RESULTS.md"),
            "official_public_unchanged": 0.5236602,
        },
        emit=True,
    )


if __name__ == "__main__":
    main()
