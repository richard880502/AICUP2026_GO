"""Rank from recognised opponents, on the 2,000 holdout3 questions: v4 similarity dump.

Reproduces Competition 2 rank submission v47, step 6 of 8. Writes
`opp_probe/h3_sims.pt`: for every holdout3 game the cosine of the
opponent's v4 game vector to each of the 6,807 v4 classes (the question's
own player masked), game kind, question index, truth, fold, the C1+C2 base
probabilities and the v4 vectors. h3_v7.py adds the v7 similarities to it.

For every game of a question the opponent's moves are compared with the
labelled training players (v4 class directions). The similarity-weighted
ranks of those players give a posterior over the opponent's registered rank.

This is the scratch script h3_link.py stopped right after it saves
h3_sims.pt (the scratch h3_dump.py raised SystemExit there); the cross-fitted
logistic experiment that followed was never reached and is not kept here.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 python -m \
      competition2_rank_prediction.experiments.opponent_identity.h3_dump
"""
import csv
import re

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.trunk_style import apply, train
from competition2_rank_prediction.experiments.opponent_identity import link_inputs

csv.field_size_limit(10**9)
DEVICE = train.DEVICE
# The research machine's locations for the opponent-identity and rank-probe files.
PROBE = train.pathlib.Path("/tmp2/richard/claude_scratch/opp_probe")
RANK = "/tmp2/richard/claude_scratch/rank_probe"
RANKS = list(dataset.cache.RANKS)
GAIN = torch.tensor(
    [[1.0 if a == b else (np.exp(-1) if abs(a - b) == 1 else 0.0) for b in range(10)]
     for a in range(10)],
    device=DEVICE, dtype=torch.float32,
)


def player_tables():
    """Registered rank per v4 class, class per player hash and handicap per training row."""
    owners, rows, selected = link_inputs.v4_gallery()
    # owners is an NpzFile array; pylint infers it as the np.array function.
    players = int(owners.max()) + 1  # pylint: disable=no-member
    player_rank = np.zeros(players, np.int64)
    player_rank[owners] = selected[rows] // 100000

    # player hash -> class index, and handicap of every training row
    first_row = {}
    for owner, row in zip(owners.tolist(), selected[rows].tolist()):  # pylint: disable=no-member
        first_row.setdefault(row, owner)
    class_of_player, handicap_of_row = {}, np.zeros(1000000, np.int8)
    for index, rank in enumerate(RANKS):
        with open(f"data/training/train_{rank}.csv", encoding="utf-8", newline="") as source:
            for position, row in enumerate(csv.DictReader(source)):
                key = index * 100000 + position
                handicap_of_row[key] = int(re.search(r"HD\[(\d+)\]", row["sgf_content"]).group(1))
                if key in first_row:
                    class_of_player[row["player_id"]] = first_row[key]
    return player_rank, class_of_player, handicap_of_row


def opponent_sims(  # pylint: disable=too-many-locals
    classes, encoder, questions, class_of_player, handicap_of_row
):
    """v4 vectors of the holdout3 opponents and their class cosines, own class masked."""
    model, scaling = encoder
    with np.load(f"{RANK}/h3_map.npz") as saved:
        # pylint: disable-next=no-member  # NpzFile arrays; pylint infers the np.array function.
        row_of_game = dict(zip(saved["game_key"].tolist(), saved["row"].tolist()))
    with open(PROBE / "h3_opp.csv", encoding="utf-8", newline="") as source:
        games = list(csv.DictReader(source))
    directory = PROBE / "h3_opp"
    masks = train.load_shards(directory, "masks", 0)
    data = {
        "trunk": train.load_shards(directory, "tokens", 0),
        "engine": np.zeros((len(masks), 160, 42), np.float16),
        "masks": masks,
        "colors": np.asarray([int(row["color"] == "W") for row in games], np.int64),
        "has_engine": np.zeros(len(masks), bool),
    }
    vectors = torch.from_numpy(train.encode(model, data, scaling, np.arange(len(masks)))).to(DEVICE)
    sims = vectors @ classes.T
    qid = np.asarray([row["player_id"] for row in games])
    own_class = np.asarray([class_of_player.get(questions[q]["player"], -1) for q in qid])
    keep = torch.from_numpy(own_class >= 0).to(DEVICE)
    sims[
        torch.arange(len(sims), device=DEVICE)[keep], torch.from_numpy(own_class).to(DEVICE)[keep]
    ] = -1
    handicap = np.asarray([handicap_of_row[row_of_game[row["game_id"]]] for row in games])
    opp_white = data["colors"] == 1
    # 0 even, 1 HD1 opp white, 2 HD1 opp black, 3 HD>=2
    kind = np.where(handicap == 0, 0, np.where(handicap == 1, np.where(opp_white, 1, 2), 3))
    print("games", len(games), "kinds", np.bincount(kind).tolist(), flush=True)
    return {"sims": sims, "vectors": vectors, "qid": qid, "own_class": own_class, "kind": kind}


def question_targets(questions, qid):
    """Question order, game -> question index, truth, fold and C1+C2 base probabilities."""
    order = sorted(questions)
    index_of = {q: i for i, q in enumerate(order)}
    game_question = torch.from_numpy(np.asarray([index_of[q] for q in qid])).to(DEVICE)
    truth = torch.tensor([RANKS.index(questions[q]["rank"]) for q in order], device=DEVICE)
    fold = np.asarray([int(questions[q]["fold"]) for q in order])
    with np.load(f"{RANK}/full_qprob.npz", allow_pickle=True) as saved:
        # pylint: disable-next=no-member  # NpzFile arrays; pylint infers the np.array function.
        position = {q: i for i, q in enumerate(saved["qid"].tolist())}
        base = torch.from_numpy(
            # pylint: disable-next=unsubscriptable-object  # same NpzFile inference
            np.stack([saved["qfull"][position[q]] for q in order])
        ).float().to(DEVICE)
    base = base / base.sum(1, keepdim=True)
    return order, game_question, truth, fold, base


def holdout3_questions():
    """question id -> holdout3 question_folds.csv row (player, rank, fold)."""
    questions = {}
    path = "runtime/competition2_research/features/holdout3/question_folds.csv"
    with open(path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            questions[row["question_id"]] = row
    return questions


def decode(probabilities):
    """Expected-score decoding."""
    return (probabilities @ GAIN).argmax(1)


def score(probabilities, truth, subset=None):
    """Mean official gain of the decoded ranks."""
    gain = GAIN[decode(probabilities), truth]
    return gain.mean().item() if subset is None else gain[subset].mean().item()


def main() -> None:
    """Writes h3_sims.pt."""
    model, scaling, _, _ = apply.load_model("style_v4")
    # torch Parameter method; pylint cannot see that it is callable.
    classes = F.normalize(model.class_weights.detach(), dim=-1)  # pylint: disable=not-callable
    player_rank, class_of_player, handicap_of_row = player_tables()

    questions = holdout3_questions()
    holdout_classes = sorted(
        {class_of_player[q["player"]] for q in questions.values() if q["player"] in class_of_player}
    )
    print("holdout3 players that are v4 classes", len(holdout_classes), "of", len(questions),
          flush=True)
    games = opponent_sims(
        classes, (model, scaling), questions, class_of_player, handicap_of_row
    )
    order, game_question, truth, fold, base = question_targets(questions, games["qid"])

    print("baseline C1+C2", round(score(base, truth), 4), flush=True)
    torch.save(
        {"sims": games["sims"].half().cpu(), "kind": games["kind"],
         "game_question": game_question.cpu(), "truth": truth.cpu(), "fold": fold,
         "base": base.cpu(), "order": order, "player_rank": player_rank,
         "holdout_classes": holdout_classes,
         "vectors": games["vectors"].half().cpu(), "own_class": games["own_class"]},
        PROBE / "h3_sims.pt",
    )


if __name__ == "__main__":
    main()
