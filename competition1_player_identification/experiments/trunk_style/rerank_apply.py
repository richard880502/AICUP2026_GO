"""Checks the trained reranker under repeated targets and applies it to pools.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.rerank_apply \
      listwise_s0 listwise_s1 listwise_s2

Logits of the listed checkpoints (seeds) are averaged. The new-pool rows of
the source submission are replaced; original-pool rows are kept as they are.
"""

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np
import torch

from competition1_player_identification.experiments.trunk_style import rerank
from competition1_player_identification.experiments.trunk_style import train

DEVICE = rerank.DEVICE
SOURCE = train.ROOT / "submission_style_v4"
OUTPUT = train.ROOT / "submission_style_v4_rerank"


def load_models(names: list) -> list:
    """Restores the named reranker checkpoints in evaluation mode."""
    models = []
    for name in names:
        model = rerank.Reranker().to(DEVICE)
        model.load_state_dict(
            torch.load(rerank.RERANK / f"{name}.pt", map_location=DEVICE)
        )
        models.append(model.eval())
    return models


def ensemble(models: list, batch: dict) -> torch.Tensor:
    """Averages the logits of all checkpoints."""
    with torch.no_grad():
        return torch.stack([model(batch) for model in models]).mean(0)


def repeated(
    models: list, embeddings: np.ndarray, repeats: int, seeds: int = 20
) -> list:
    """Utility of base / CSLS / reranker when some targets repeat."""
    games = torch.from_numpy(embeddings).to(DEVICE)
    counts = torch.tensor(rerank.COUNTS, device=DEVICE).repeat_interleave(100)
    totals = np.zeros(3)
    for seed in range(seeds):
        targets = torch.from_numpy(rerank.repeated_targets(seed, repeats)).to(DEVICE)
        batch = rerank.build(games, targets, counts, False)
        # base: order the same 20 by the standardized cosine (truth outside => 0)
        totals += [
            rerank.utility(batch["scalars"][..., 0], batch).mean().item(),
            rerank.utility(batch["scalars"][..., 1], batch).mean().item(),
            rerank.utility(ensemble(models, batch), batch).mean().item(),
        ]
    return [round(float(x), 4) for x in totals / seeds]


def pool_answers(models: list, name: str, answers: dict) -> dict:
    """Adds reranked Top-5 players of one new pool to answers; returns stats."""
    with np.load(rerank.RERANK / f"pool_{name}.npz", allow_pickle=True) as arrays:
        games = torch.from_numpy(arrays["embeddings"]).to(DEVICE)
        counts = torch.from_numpy(arrays["counts"]).to(DEVICE)
        questions = np.asarray(arrays["questions"])
        players = np.asarray(arrays["players"])
    games = torch.nn.functional.normalize(games, dim=-1)
    rows = torch.arange(len(counts), device=DEVICE)
    batch = rerank.build(games, rows, counts, False)
    order = ensemble(models, batch).argsort(dim=1, descending=True)
    chosen = batch["chosen"].gather(1, order)[:, :5].cpu().numpy()
    before = batch["chosen"][:, :5].cpu().numpy()
    for question, row in zip(questions, chosen):
        answers[str(question)] = [str(players[index]) for index in row]
    return {
        "questions": len(chosen),
        "top1_changed_vs_csls": float((chosen[:, 0] != before[:, 0]).mean()),
        "top1_repeat_share": 1 - len(np.unique(chosen[:, 0])) / len(chosen),
    }


def write_submission(  # pylint: disable=too-many-locals
    source: pathlib.Path,
    output: pathlib.Path,
    answers: dict,
    previous: dict | None = None,
) -> dict:
    """Replaces the answered rows of a submission and writes it to output.

    Top-1 changes are counted against `previous` (question id -> top_1), by
    default the source itself; unanswered rows must agree with it.
    """
    with open(source / "submission_player.csv", encoding="utf-8", newline="") as file:
        table = list(csv.DictReader(file))
    if previous is None:
        previous = {row["question_id"]: row["top_1"] for row in table}
    replaced = changed = 0
    for row in table:
        identifier = row["question_id"]
        if identifier in answers:
            new = answers[identifier]
            changed += previous[identifier] != new[0]
            for place, player in enumerate(new, start=1):
                row[f"top_{place}"] = player
            replaced += 1
        elif previous[identifier] != row["top_1"]:
            raise ValueError("Original-pool rows differ between submissions")
    if replaced != len(answers):
        raise ValueError("Question ids do not match the submission")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "submission_player.csv"
    with open(path, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return {
        "rows": len(table),
        "replaced": replaced,
        "top1_changed": changed / replaced,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Runs the repeated-target check, then writes the reranked submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--source", type=pathlib.Path, default=SOURCE)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    arguments = parser.parse_args()
    models = load_models(arguments.checkpoints)
    cohorts = np.load(rerank.RERANK / "cohorts_style_v4.npz")
    report: dict = {
        "checkpoints": arguments.checkpoints,
        "columns": ["cosine_in_top20", "csls", "reranked"],
    }
    for split in ("validation", "final"):
        report[split] = {
            f"repeats_{count}": repeated(models, cohorts[split], count)
            for count in (0, 20, 40, 80)
        }
    print(json.dumps(report, indent=1), flush=True)
    answers: dict = {}
    for name in ("pub2", "priv2"):
        report[name] = pool_answers(models, name, answers)
    written = write_submission(arguments.source, arguments.output, answers)
    report["rows"], report["replaced"] = written["rows"], written["replaced"]
    report["top1_changed_vs_v4"] = written["top1_changed"]
    report["sha256"] = written["sha256"]
    (arguments.output / "report.json").write_text(json.dumps(report, indent=1))
    shown = ("pub2", "priv2", "rows", "replaced", "top1_changed_vs_v4", "sha256")
    print(json.dumps({key: report[key] for key in shown}, indent=1))


if __name__ == "__main__":
    main()
