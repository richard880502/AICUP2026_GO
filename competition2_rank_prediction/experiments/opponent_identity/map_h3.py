"""Map holdout3 question games to global training rows by SGF move text + player.

Also reports corpus coverage. Reproduces Competition 2 rank submission v47,
step 1 of 8: writes `rank_probe/h3_map.npz` (game key `question#index` ->
training row), read by h3_dump.py and h3_final_v47.py.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition2_rank_prediction.experiments.opponent_identity.map_h3
"""
import collections
import csv
import hashlib
import re

import numpy as np

csv.field_size_limit(10**9)
R = "runtime/competition2_research"
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
# The research machine's location for the rank-probe intermediate files.
OUT = "/tmp2/richard/claude_scratch/rank_probe"


def key(sgf):
    """MD5 of the move text (from the first move on)."""
    m = re.search(r";[BW]\[", sgf)
    return hashlib.md5(sgf[m.start():].encode()).digest() if m else b""


def wanted():
    """Question labels and (move key, player) -> [(question, game index, colour)]."""
    lab = {}
    with open(f"{R}/features/holdout3/question_folds.csv", encoding="utf-8") as source:
        for r in csv.DictReader(source):
            lab[r["question_id"]] = r
    want = collections.defaultdict(list)  # (key,player)->[(qid,i)]
    nq = 0
    for split in ("public", "private"):
        path = f"{R}/features/holdout3/tests/rank_prediction_test_{split}.csv"
        with open(path, encoding="utf-8") as source:
            for r in csv.DictReader(source):
                q = r["question_id"]
                nq += 1
                for i in range(1, int(r["num_games"]) + 1):
                    want[(key(r[f"sgf_{i}"]), lab[q]["player"])].append((q, i, r[f"color_{i}"]))
    print("questions", nq, "games", sum(len(v) for v in want.values()), "unique keys", len(want))
    return lab, want


def locate(want):
    """(question, game index) -> running row number over the ten training files."""
    found = {}
    row = 0
    amb = 0
    for rk in RANKS:
        with open(f"data/training/train_{rk}.csv", encoding="utf-8") as source:
            for r in csv.DictReader(source):
                k = (key(r["sgf_content"]), r["player_id"])
                if k in want:
                    for (q, i, _) in want[k]:
                        if (q, i) in found:
                            amb += 1
                        found[(q, i)] = row
                row += 1
    print("rows", row, "found", len(found), "ambiguous", amb)
    return found


def coverage(lab, found, gr):
    """Prints how much of holdout3 the style corpus covers, then checks the folds."""
    gi = np.load("production/open_set_v2_20261111/engine_inputs/global_indices.npy")
    g2 = np.load("production/trunk_style_20261113/inputs/corpus2_global_indices.npy")
    sel = np.concatenate([gi, g2])
    cov = np.isin(gr, sel)
    print("covered games", cov.sum(), "of", len(gr), cov.mean())
    perq = collections.defaultdict(lambda: [0, 0])
    for (q, _), c in zip(found, cov):
        perq[q][0] += c
        perq[q][1] += 1
    a = np.array(list(perq.values()))
    fr = a[:, 0] / a[:, 1]
    print(
        "questions fully covered", (fr == 1).sum(), "zero covered", (a[:, 0] == 0).sum(),
        "mean frac", fr.mean(),
        "covered games/q pct", np.percentile(a[:, 0], [5, 25, 50, 75, 95]),
        "games/q pct", np.percentile(a[:, 1], [5, 50, 95]),
    )
    check_folds(lab)


def check_folds(lab):
    """Prints whether each fold's players match its exclude list, and the rank counts."""
    ex = []
    for j in (0, 1):
        with open(f"{R}/features/holdout3/exclude_c{j}.txt", encoding="utf-8") as source:
            ex.append(set(source.read().split()))
    for j in (0, 1):
        pj = {r["player"] for r in lab.values() if int(r["fold"]) == j}
        print("fold", j, len(pj), "== exclude:", pj == ex[j])
    print(collections.Counter(r["rank"] for r in lab.values()))


def main() -> None:
    """Writes h3_map.npz and prints the coverage report."""
    lab, want = wanted()
    found = locate(want)
    gk = np.array([f"{q}#{i}" for (q, i) in found])
    gr = np.array(list(found.values()))
    np.savez(f"{OUT}/h3_map.npz", game_key=gk, row=gr)
    coverage(lab, found, gr)


if __name__ == "__main__":
    main()
