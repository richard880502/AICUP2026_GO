"""Probes, validates and applies per-move matching next to the game vector.

Example (repository root), in order:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.matching_apply \
      probe
    ... matching_apply validation
    ... matching_apply pools

probe: training-free matching on the v4 encoder's hidden states (and on raw
trunk activations) for the validation and final cohorts.
validation: fusion of v4 cosine, untrained matching, the fine-tuned matching
head and its tuned cosine; weights are chosen on the validation cohort and
the final cohort is reported for them.
pools: applies weights (1, 0.5, 0.5) and CSLS to the official new pools and
writes the submission.
"""

import argparse
import csv
import itertools
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.trunk_style import apply
from competition1_player_identification.experiments.trunk_style import matching
from competition1_player_identification.experiments.trunk_style import rerank_apply
from competition1_player_identification.experiments.trunk_style import rerank_inputs
from competition1_player_identification.experiments.trunk_style import train

WEIGHTS = (1.0, 0.5, 0.5)
PROBE_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf"))
RERANK = matching.RERANK
SOURCE = train.ROOT / "submission_style_v4"
PREVIOUS = train.ROOT / "submission_style_v4_rerank"
OUTPUT = train.ROOT / "submission_style_v5_match"
DEVICE = matching.DEVICE
COUNTS = matching.COUNTS


def fused(parts: tuple | list, weights: tuple) -> torch.Tensor:
    """Weighted sum of standardized score matrices."""
    terms = [weight * part for weight, part in zip(weights, parts)]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def probe(  # pylint: disable=too-many-locals
    kinds: list, output: pathlib.Path
) -> None:
    """Saves training-free MaxSim scores of both cohorts and prints fusions."""
    model, scaling, _, _ = apply.load_model("style_v4")
    corpus = train.load_corpus(0)
    _, final = infer.positions(train.OPEN, "final_holdout")
    splits = {"validation": np.asarray(corpus["validation"]), "final": np.asarray(final)}
    cosine = np.load(RERANK / "cohorts_style_v4.npz")
    for kind in kinds:
        for split, rows in splits.items():
            tokens, valid, _ = matching.move_tokens(
                model, corpus, scaling, rows.ravel(), kind
            )
            per_game = matching.maxsim(
                tokens.reshape(600, 15, 160, -1), valid.reshape(600, 15, 160), step=4
            )
            torch.save(per_game.cpu(), output / f"maxsim_{kind}_{split}.pt")
            table = matching.fusion_table(
                torch.from_numpy(cosine[split]).to(DEVICE), per_game, PROBE_WEIGHTS
            )
            shown = {
                key: [round(values[0], 4)] + [round(value, 3) for value in values[1:]]
                for key, values in table.items()
            }
            print(kind, split, json.dumps(shown), flush=True)
            del tokens, valid
            torch.cuda.empty_cache()


def cache_cohorts(name: str) -> None:
    """Scores both cohorts with the fine-tuned head unless already saved."""
    if (RERANK / f"match_{name}_final.pt").exists():
        return
    encoder, scaling, _, _ = apply.load_model("style_v4")
    model = matching.TokenModel(encoder).to(DEVICE)
    state = torch.load(RERANK / f"{name}.pt", map_location=DEVICE)
    model.load_state_dict(state["state"])
    print("checkpoint epoch", state["epoch"], flush=True)
    data = train.load_corpus(0)
    _, final = infer.positions(train.OPEN, "final_holdout")
    for split, rows in (("validation", data["validation"]), ("final", final)):
        matching.cohort(
            model, data, scaling, rows, save=str(RERANK / f"match_{name}_{split}.pt")
        )


def sources(name: str, split: str) -> dict:
    """Standardized score matrices of one cohort for every query count.

    Order: v4 cosine, untrained matching, trained matching, tuned cosine.
    """
    cosine = np.load(RERANK / "cohorts_style_v4.npz")
    games = torch.from_numpy(cosine[split]).to(DEVICE)
    hidden = torch.load(RERANK / f"maxsim_hidden_{split}.pt").to(DEVICE)
    saved = torch.load(RERANK / f"match_{name}_{split}.pt")
    trained, tuned = saved["per_game"].to(DEVICE), saved["games"].to(DEVICE)
    out = {}
    for count in COUNTS:
        out[count] = (
            matching.cosine_scores(games, count),
            matching.standardize(hidden[:, :count].mean(1)),
            matching.standardize(trained[:, :count].mean(1)),
            matching.cosine_scores(tuned, count),
        )
    return out


def diag_utility(source: dict, weights: tuple) -> tuple[float, list]:
    """Utility with all 600 players asked; mean over counts and each count."""
    values = [matching.utility(fused(source[count], weights)) for count in COUNTS]
    return float(np.mean(values)), values


def official_like(source: dict, weights: tuple, correct: bool) -> float:
    """Utility with 400 distinct targets of 600 and mixed query counts."""
    totals = []
    for seed in range(20):
        picks = np.random.default_rng(seed).permutation(600)[:400].reshape(4, 100)
        blocks = [
            fused(source[count], weights)[torch.from_numpy(rows).to(DEVICE)]
            for count, rows in zip(COUNTS, picks)
        ]
        scores = torch.cat(blocks)
        if correct:
            scores = matching.csls(scores)
        truth = torch.from_numpy(picks.ravel()).to(DEVICE)
        own = scores.gather(1, truth[:, None])
        ranks = (scores > own).sum(1)
        totals.append(
            torch.where(ranks < 5, torch.exp(-ranks.float()), 0.0).mean().item()
        )
    return round(float(np.mean(totals)), 4)


def validation(name: str) -> None:
    """Chooses fusion weights on validation and reports the final cohort."""
    cache_cohorts(name)
    val, fin = sources(name, "validation"), sources(name, "final")
    grid = list(
        itertools.product(
            (1.0,), (0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 1.0), (0.0, 0.5, 1.0)
        )
    )
    table = {weights: diag_utility(val, weights)[0] for weights in grid}
    ranked = sorted(table, key=lambda weights: -table[weights])
    print("columns: v4 cosine, untrained matching, trained matching, tuned cosine")
    shown = [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.0, 0.0),
        (1.0, 0.0, 0.5, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ] + ranked[:6]
    for weights in shown:
        f_mean, f_counts = diag_utility(fin, weights)
        row = {
            "weights": weights,
            "validation": round(diag_utility(val, weights)[0], 4),
            "final": round(f_mean, 4),
            "final_by_count": [round(x, 3) for x in f_counts],
            "official_like_val": official_like(val, weights, False),
            "official_like_val_csls": official_like(val, weights, True),
            "official_like_final": official_like(fin, weights, False),
            "official_like_final_csls": official_like(fin, weights, True),
        }
        print(json.dumps(row), flush=True)


def by_count(per_game: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """per_game [600,5,600] -> question scores [Q,600] by each question's count."""
    keep = (torch.arange(5, device=DEVICE)[None] < counts[:, None]).float()
    rows = per_game[: len(counts)]
    return (rows * keep[..., None]).sum(1) / counts[:, None].float()


def pool_state(  # pylint: disable=too-many-locals
    name: str, encoder: train.StyleEncoder, scaling: tuple
) -> dict:
    """Game-vector cosine and untrained matching of one official new pool."""
    data, positions, names = rerank_inputs.pool_tokens(name)
    rows = positions.ravel()
    counts = torch.from_numpy(np.asarray(names["counts"]).astype(np.int64)).to(DEVICE)
    tokens, valid, games = matching.move_tokens(encoder, data, scaling, rows)
    untrained = matching.maxsim(
        tokens.reshape(600, 15, 160, -1), valid.reshape(600, 15, 160)
    )
    del tokens
    games = games.reshape(600, 15, -1)
    centroids = F.normalize(games[:, :10].mean(1), dim=-1)
    keep = (torch.arange(5, device=DEVICE)[None] < counts[:, None]).float()
    queries = F.normalize((games[: len(counts), 10:] * keep[..., None]).sum(1), dim=-1)
    state = {
        "data": data,
        "rows": rows,
        "counts": counts,
        "names": names,
        "cosine": queries @ centroids.T,
        "untrained": by_count(untrained, counts),
    }
    del untrained, valid, games
    torch.cuda.empty_cache()
    return state


def pool_answers(
    state: dict, trained: torch.Tensor, scores: pathlib.Path, answers: dict
) -> dict:
    """Fuses the three scores of one pool, adds Top-5 answers, returns stats."""
    cosine, names = state["cosine"], state["names"]
    parts = [
        matching.standardize(part) for part in (cosine, state["untrained"], trained)
    ]
    final = matching.csls(fused(parts, WEIGHTS))
    order = final.argsort(dim=1, descending=True)[:, :5].cpu().numpy()
    plain = parts[0].argsort(dim=1, descending=True)[:, :5].cpu().numpy()
    np.savez_compressed(
        scores,
        cosine=cosine.cpu().numpy(),
        untrained=state["untrained"].cpu().numpy(),
        trained=trained.cpu().numpy(),
    )
    for question, chosen in zip(names["questions"], order):
        answers[str(question)] = [str(names["players"][index]) for index in chosen]

    def correlation(index: int) -> float:
        pair = torch.stack((parts[0].flatten(), parts[index].flatten()))
        return float(torch.corrcoef(pair)[0, 1])

    return {
        "questions": len(order),
        "top1_changed_vs_cosine": float((order[:, 0] != plain[:, 0]).mean()),
        "new_top1_in_cosine_top5": float((plain == order[:, :1]).any(1).mean()),
        "top1_repeat_share": 1 - len(np.unique(order[:, 0])) / len(order),
        "correlation_cosine_untrained": correlation(1),
        "correlation_cosine_trained": correlation(2),
    }


def pools(  # pylint: disable=too-many-locals
    name: str, output: pathlib.Path, scores: pathlib.Path
) -> None:
    """Writes the matching submission for the official new pools."""
    report: dict = {"weights": WEIGHTS, "correction": "csls_k5"}
    answers: dict = {}
    encoder, scaling, _, _ = apply.load_model("style_v4")
    states = {pool: pool_state(pool, encoder, scaling) for pool in ("pub2", "priv2")}
    del encoder
    encoder, scaling, _, _ = apply.load_model("style_v4")
    tuned = matching.TokenModel(encoder).to(DEVICE)
    tuned.load_state_dict(
        torch.load(RERANK / f"{name}.pt", map_location=DEVICE)["state"]
    )
    for pool, state in states.items():
        _, moves, valid = matching.embed(tuned, state["data"], scaling, state["rows"])
        trained = by_count(
            matching.maxsim(
                moves.reshape(600, 15, 160, -1), valid.reshape(600, 15, 160)
            ),
            state["counts"],
        )
        report[pool] = pool_answers(
            state, trained, scores / f"pool_scores_{pool}.npz", answers
        )
    with open(
        PREVIOUS / "submission_player.csv", encoding="utf-8", newline=""
    ) as source:
        previous = {row["question_id"]: row["top_1"] for row in csv.DictReader(source)}
    written = rerank_apply.write_submission(SOURCE, output, answers, previous)
    report["rows"], report["replaced"] = written["rows"], written["replaced"]
    report["top1_changed_vs_rerank_submission"] = written["top1_changed"]
    report["sha256"] = written["sha256"]
    (output / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    """Runs the selected step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("probe", "validation", "pools"))
    parser.add_argument("--name", default="style_v5_match", help="Matching head.")
    parser.add_argument(
        "--kinds", nargs="+", default=["hidden", "raw"], help="probe: token kinds."
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=OUTPUT, help="pools: submission."
    )
    parser.add_argument(
        "--scores",
        type=pathlib.Path,
        default=RERANK,
        help="Directory of the probe dumps and of the pool score dumps.",
    )
    arguments = parser.parse_args()
    arguments.scores.mkdir(parents=True, exist_ok=True)
    if arguments.step == "probe":
        probe(arguments.kinds, arguments.scores)
    elif arguments.step == "validation":
        validation(arguments.name)
    else:
        pools(arguments.name, arguments.output, arguments.scores)


if __name__ == "__main__":
    main()
