"""Seed robustness and a dimension control for the ten-vs-three gate.

`audit.py` reports one question sampling and one fold split. This repeats the
``three`` and ``ten`` arms under many seeds, and adds a control: the 21 extra
columns that ``ten`` has over ``three`` are kept, but shuffled across
questions. That control matches ``ten``'s dimension count and marginals while
destroying the information in the extra profiles, so it separates "more
profiles carry information" from "more columns help the learner".

Seeds change question sampling and folds only; the 400 players are fixed, so
this tests robustness to sampling, not to the cohort.

CPU only; reads cached features and trains no network.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from competition2_rank_prediction.experiments.rankmle_audit import audit

BASE_SEED = 20261118


def one_seed(data: dict, seed: int) -> dict:
    """ten-minus-three and shuffled-minus-three for one seed."""
    audit.SEED = seed
    questions = audit.make_questions(data, 20)
    labels, players = questions["labels"], questions["players"]
    fold = audit.player_folds(questions, 5)
    extra = tuple(s for s in ("normal",) + audit.TEN if s not in audit.THREE)
    three = audit.question_features(data, questions, audit.THREE, (0, 1, 2))
    ten = audit.question_features(data, questions, ("normal",) + audit.TEN, (0, 1, 2))
    extra_columns = audit.question_features(data, questions, extra, (0, 1, 2))
    order = np.random.default_rng(seed + 7).permutation(len(labels))
    shuffled = np.concatenate([three, extra_columns[order]], axis=1)
    score = {
        name: audit.utility(audit.decode(audit.learned(x, labels, fold)), labels)
        for name, x in (("three", three), ("ten", ten), ("shuffled", shuffled))
    }
    return {
        "seed": seed,
        "three": round(float(score["three"].mean()), 6),
        "ten_minus_three": audit.paired(score["ten"], score["three"], players),
        "shuffled_minus_three": audit.paired(score["shuffled"], score["three"], players),
    }


def main() -> None:
    """Runs every seed and prints one JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    data = audit.load(args.features)
    rows = [one_seed(data, BASE_SEED + 1000 * k) for k in range(args.seeds)]
    delta = np.array([row["ten_minus_three"]["delta"] for row in rows])
    lower = np.array([row["ten_minus_three"]["ci95"][0] for row in rows])
    control = np.array([row["shuffled_minus_three"]["delta"] for row in rows])
    summary = {
        "seeds": len(rows),
        "ten_minus_three_mean": round(float(delta.mean()), 6),
        "ten_minus_three_min_max": [round(float(delta.min()), 6), round(float(delta.max()), 6)],
        "seeds_positive": int((delta > 0).sum()),
        "seeds_passing_gate": int(((delta >= 0.003) & (lower > 0)).sum()),
        "shuffled_minus_three_mean": round(float(control.mean()), 6),
        "shuffled_minus_three_min_max": [
            round(float(control.min()), 6), round(float(control.max()), 6)
        ],
        "per_seed": rows,
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
