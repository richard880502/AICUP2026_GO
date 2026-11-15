"""How many opponents are labelled training players? (closed-set test with v4)

Reproduces Competition 2 rank submission v47, step 5 of 8 (after
link_inputs.py and opp_sample.py and the trunk-token extraction of their
lists). Writes `opp_probe/test_opp_scores.npz` (v4 vectors, question,
own colour and handicap of every official opponent-side game), read by
h3_final_v47.py, plus test_own_scores.npz and controls.npz.

v4 was trained on up to 48 games of each of 6,807 labelled players, so its
ArcFace class directions recognise those players in unseen games. Controls:
  present : unseen own-side games of those players (true class known)
  absent  : games of the 1,200 held-out cohort players (never seen)
Queries: opponents in training games, opponents and own side of the official
rank-test games. The share of recognised players is estimated from how often
the top class cosine exceeds the absent controls' 99th percentile.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 python -m \
      competition2_rank_prediction.experiments.opponent_identity.link_test
"""
import csv
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.open_set_v2 import dataset, infer
from competition1_player_identification.experiments.trunk_style import apply, train
from competition2_rank_prediction.experiments.opponent_identity import link_inputs

csv.field_size_limit(10**9)
DEVICE = train.DEVICE
# The research machine's location for the opponent-identity intermediate files.
PROBE = train.pathlib.Path("/tmp2/richard/claude_scratch/opp_probe")
RANKS = dataset.cache.RANKS


class Probe:
    """The v4 encoder, its class directions and the present-control thresholds."""

    def __init__(self):
        self.model, self.scaling, _, _ = apply.load_model("style_v4")
        # torch Parameter method; pylint cannot see that it is callable.
        weights = self.model.class_weights.detach()  # pylint: disable=not-callable
        self.classes = F.normalize(weights, dim=-1)
        self.thresholds = {}
        self.p_top = self.p_margin = None
        self.report = {}

    def score(self, data, exclude=None):
        """Top class cosine, top-1 margin, top class and game vectors."""
        vectors = torch.from_numpy(
            train.encode(self.model, data, self.scaling, np.arange(len(data["masks"])))
        ).to(DEVICE)
        sims = vectors @ self.classes.T
        if exclude is not None:
            keep = torch.from_numpy(exclude >= 0).to(DEVICE)
            sims[
                torch.arange(len(sims), device=DEVICE)[keep],
                torch.from_numpy(exclude).to(DEVICE)[keep],
            ] = -1
        top = sims.topk(2, dim=1)
        return (
            top.values[:, 0].cpu().numpy(),
            (top.values[:, 0] - top.values[:, 1]).cpu().numpy(),
            top.indices[:, 0].cpu().numpy(),
            vectors,
        )

    def estimate(self, name, top, margin):
        """Estimated share of known players among the queries, against the controls."""
        out = {}
        for q, (cut, margin_cut) in self.thresholds.items():
            for label, values, control, limit in (
                ("cosine", top, self.p_top, cut), ("margin", margin, self.p_margin, margin_cut)
            ):
                share, base = float((values > limit).mean()), float((control > limit).mean())
                out[f"{label}>{q}"] = {
                    "share": round(share, 4), "present_share": round(base, 4),
                    "estimated_known_fraction": round((share - (1 - q)) / (base - (1 - q)), 3),
                }
        self.report[name] = out
        print(name, json.dumps(out), flush=True)


def gallery():
    """Corpus rows of v4's training games, rank per class and class per corpus row."""
    owners, rows, selected = link_inputs.v4_gallery()
    # owners is an NpzFile array; pylint infers it as the np.array function.
    player_rank = np.zeros(int(owners.max()) + 1, np.int64)  # pylint: disable=no-member
    player_rank[owners] = selected[rows] // 100000
    owner_of_row = np.full(len(selected), -1)
    owner_of_row[rows] = owners
    return selected, player_rank, owner_of_row


def table(name):
    """Rows of one game list in the probe directory."""
    with open(PROBE / f"{name}.csv", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def from_shards(name, rows_csv):
    """train.encode input from trunk-only shards (no engine scalars)."""
    directory = PROBE / (name if name != "opp_sample" else "tokens")
    masks = train.load_shards(directory, "masks", 0)
    return {
        "trunk": train.load_shards(directory, "tokens", 0),
        "engine": np.zeros((len(masks), 160, 42), np.float16),
        "masks": masks,
        "colors": np.asarray([int(row["color"] == "W") for row in rows_csv], np.int64),
        "has_engine": np.zeros(len(masks), bool),
    }


def line(name, top, margin):
    """One summary line of top-cosine quantiles and the median margin."""
    q = np.quantile(top, [0.1, 0.5, 0.9]).round(3).tolist()
    return (
        f"{name:26s} n {len(top):5d} top cosine q10/50/90 {q} "
        f"margin median {np.median(margin):.3f}"
    )


def handicaps(rows_csv):
    """HD[...] value of every game."""
    return np.asarray(
        [int(re.search(r"HD\[(\d+)\]", row["sgf_content"]).group(1)) for row in rows_csv]
    )


def absent_controls(probe):
    """Scores of the 1,200 held-out cohort players' games (never seen by v4)."""
    corpus = train.load_corpus(0)
    _, final = infer.positions(train.OPEN, "final_holdout")
    absent_rows = np.concatenate(
        (np.asarray(corpus["validation"]).ravel(), np.asarray(final).ravel())
    )
    absent = {key: corpus[key][absent_rows] for key in ("trunk", "engine", "masks", "colors")}
    absent["has_engine"] = np.zeros(len(absent_rows), bool)
    a_top, a_margin, _, _ = probe.score(absent)
    del corpus
    print(line("absent (cohort players)", a_top, a_margin), flush=True)
    return a_top, a_margin


def training_opponents(probe, selected, player_rank, owner_of_row):  # pylint: disable=too-many-locals
    """Opponents in sampled training games: known share and rank agreement."""
    sample = table("opp_sample")
    corpus_rows = np.asarray([int(row["corpus_row"]) for row in sample])
    own = owner_of_row[corpus_rows]
    o_top, o_margin, o_index, _ = probe.score(from_shards("opp_sample", sample), own)
    print(line("training-game opponents", o_top, o_margin), flush=True)
    probe.estimate("training_opponents", o_top, o_margin)
    labelled = selected[corpus_rows] // 100000
    handicap = handicaps(sample)
    opp_white = np.asarray([row["color"] == "W" for row in sample])
    cut = probe.thresholds[0.999][1]
    for name, mask in (("confident (margin)", o_margin > cut), ("rest", o_margin <= cut)):
        for hd in (0, 1):
            m = mask & (handicap == hd)
            diff = player_rank[o_index[m]] - labelled[m]
            row = {
                "subset": name, "HD": hd, "n": int(m.sum()),
                "same_rank": round(float((diff == 0).mean()), 3),
                "within_1": round(float((np.abs(diff) <= 1).mean()), 3),
            }
            if hd == 1:
                for label, colour in (("opp_white", True), ("opp_black", False)):
                    c = m & (opp_white == colour)
                    d = player_rank[o_index[c]] - labelled[c]
                    row[label] = {
                        "n": int(c.sum()),
                        "mean_diff": round(float(d.mean()), 3) if c.sum() else None,
                        "share +1/0/-1": (
                            [round(float((d == v).mean()), 3) for v in (1, 0, -1)]
                            if c.sum() else None
                        ),
                    }
            print(json.dumps(row), flush=True)
    return {
        "o_top": o_top, "o_margin": o_margin, "o_index": o_index, "labelled": labelled,
        "handicap": handicap, "opp_white": opp_white,
    }


def official(probe):
    """Scores both sides of every official rank-test game and saves them."""
    for name in ("test_opp", "test_own"):
        rows_csv = table(name)
        top, margin, index, vectors = probe.score(from_shards(name, rows_csv))
        print(line(f"official {name}", top, margin), flush=True)
        probe.estimate(name, top, margin)
        np.savez(PROBE / f"{name}_scores.npz", top=top, margin=margin, index=index,
                 vectors=vectors.cpu().numpy().astype(np.float16),
                 question=np.asarray([row["player_id"] for row in rows_csv]),
                 own_color=np.asarray([row["note"] for row in rows_csv]),
                 handicap=handicaps(rows_csv))


def main() -> None:
    """Controls, training-game opponents, then the official games."""
    probe = Probe()
    selected, player_rank, owner_of_row = gallery()
    a_top, a_margin = absent_controls(probe)

    present_rows = table("present")
    truth = np.asarray([int(row["note"].split(":")[0]) for row in present_rows])
    p_top, p_margin, p_index, _ = probe.score(from_shards("present", present_rows))
    print(
        line("present (unseen games)", p_top, p_margin),
        "top-1 correct", round(float((p_index == truth).mean()), 4), flush=True,
    )
    probe.thresholds = {
        q: (float(np.quantile(a_top, q)), float(np.quantile(a_margin, q))) for q in (0.99, 0.999)
    }
    probe.p_top, probe.p_margin = p_top, p_margin
    probe.estimate("present", p_top, p_margin)
    opponents = training_opponents(probe, selected, player_rank, owner_of_row)
    official(probe)
    np.savez(PROBE / "controls.npz", a_top=a_top, a_margin=a_margin, p_top=p_top,
             p_margin=p_margin, p_correct=p_index == truth, **opponents, player_rank=player_rank)


if __name__ == "__main__":
    main()
