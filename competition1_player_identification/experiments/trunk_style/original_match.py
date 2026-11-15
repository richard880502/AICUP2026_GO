"""All-moves encoder + move matching + CSLS on the original pools.

Reproduces Competition 1 style submission v6 (`submission_style_v6_orig`),
step 1 of 5 of the v9 chain: v9 keeps these original-pool answers and
rich_pipeline.py / ensemble_v9.py start from this submission. Reads the
`style_v4` encoder, the `*_all` token shards of the original pools and the
v5 submission (`submission_style_v5_match`).

Original pools: 400 candidates x 100 black games, questions of 5-20 games.
Validation holds out the last 5 games of every candidate and asks 1/2/3/5
of them against the other 95 (a harder protocol than the official one, so
that methods can be told apart). The official answers use all 100 games.
Writes production/trunk_style_20261113/submission_style_v6_orig/.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.original_match \
      [pub priv]
"""
import csv
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.trunk_style import (
    apply,
    matching,
    train,
)
from competition1_player_identification.production_pipeline import (
    open_set_official as official,
)

ROOT = train.ROOT
INPUT = apply.INPUT
SOURCE = ROOT / "submission_style_v5_match"
OUTPUT = ROOT / "submission_style_v6_orig"
DEVICE = train.DEVICE
COUNTS = (1, 2, 3, 5)
WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def load_games(name, group):
    """All-moves trunk tokens with the engine rows scattered to their moves."""
    features = official.load_shards(
        INPUT / name / "features" / group, ("tokens", "masks", "record_ids", "group_indices")
    )
    directory = ROOT / f"{name}_{group}_all"
    if not np.array_equal(train.load_shards(directory, "record_ids", 0), features["record_ids"]):
        raise ValueError("Trunk and engine caches list different games")
    masks = train.load_shards(directory, "masks", 0)
    engine = train.scatter_engine(features["tokens"], features["masks"], masks)
    games = {
        "trunk": train.load_shards(directory, "tokens", 0),
        "engine": engine,
        "masks": masks,
        "colors": np.zeros(len(masks), np.int64),
    }
    return games, features["group_indices"], float(engine[:, :, 41].any(1).mean())


def encode(games, encoder, scaling):
    """Unit hidden-state move vectors (half), validity and game vectors."""
    rows = np.arange(len(games["masks"]))
    outputs, masks, vectors = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), 512):
            tokens, valid, colors = train.batch_tensors(games, scaling, rows[start : start + 512])
            hidden = encoder.project(tokens) + encoder.step + encoder.color(colors)[:, None]
            hidden = encoder.norm(encoder.encoder(hidden, src_key_padding_mask=~valid))
            outputs.append(F.normalize(hidden.float(), dim=-1).half())
            masks.append(valid)
            vectors.append(encoder(tokens, valid, colors).float())
    return torch.cat(outputs), torch.cat(masks), torch.cat(vectors)


def maxsim(queries, query_valid, refs, ref_valid):
    """queries [G,160,D]; refs [C,T,D] -> [G,C] mean best cosine per query move."""
    players, slots, width = refs.shape
    flat = refs.reshape(-1, width)
    penalty = torch.where(ref_valid, 0.0, -4.0).half()[None, None]
    out = torch.zeros(len(queries), players, device=DEVICE)
    for start in range(0, len(queries), 2):
        block = queries[start : start + 2]
        sims = (block.reshape(-1, width) @ flat.T).reshape(len(block), 160, players, slots)
        best = sims.add_(penalty).amax(-1).float()
        weight = query_valid[start : start + 2].float()
        out[start : start + 2] = (
            (best * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp(min=1)
        )
        del sims
    return out


def utility(scores, truth):
    """Official-style score: exp(-rank) when the true player is in the top 5."""
    own = scores.gather(1, truth[:, None])
    ranks = (scores > own).sum(1)
    return torch.where(ranks < 5, torch.exp(-ranks.float()), 0.0).mean().item()


def top1(scores, truth):
    """Share of questions whose best-scored player is the true one."""
    return (scores.argmax(1) == truth).float().mean().item()


def variants(cosine, match):
    """All fusion weights, each with and without CSLS."""
    out = {}
    for weight in WEIGHTS:
        fused = matching.standardize(cosine) + weight * matching.standardize(match)
        out[f"w{weight}"] = fused
        out[f"w{weight}_csls"] = matching.csls(fused)
    return out


def validate(tokens, valid, vectors):
    """Held-out candidate games (last 5) against the other 95."""
    refs = tokens[:, :95].reshape(400, 95 * 160, -1).contiguous()
    ref_valid = valid[:, :95].reshape(400, 95 * 160)
    held = tokens[:, 95:].reshape(2000, 160, -1)
    per_game = maxsim(held, valid[:, 95:].reshape(2000, 160), refs, ref_valid).reshape(400, 5, 400)
    centroids = F.normalize(vectors[:, :95].mean(1), dim=-1)
    truth = torch.arange(400, device=DEVICE)
    table = {}
    for count in COUNTS:
        cosine = F.normalize(vectors[:, 95 : 95 + count].mean(1), dim=-1) @ centroids.T
        for key, scores in variants(cosine, per_game[:, :count].mean(1)).items():
            table.setdefault(key, []).append((utility(scores, truth), top1(scores, truth)))
    del refs
    torch.cuda.empty_cache()
    return {
        key: {"score": [round(v[0], 4) for v in values], "top1": [round(v[1], 4) for v in values]}
        for key, values in table.items()
    }


def score_pool(  # pylint: disable=too-many-locals
    name, encoder, scaling, report, tables
):
    """Validates on one original pool and scores its official questions.

    Fills report[name] and tables[name]; returns the answer state
    (cosine, match, question ids, player ids).
    """
    metadata = json.loads((INPUT / name / "features/metadata.json").read_text())
    games, owners, agreement = load_games(name, "candidates")
    if not np.array_equal(owners, np.repeat(np.arange(400), 100)):
        raise ValueError("Candidate games are not grouped 100 per player in order")
    tokens, valid, vectors = encode(games, encoder, scaling)
    del games
    tokens = tokens.reshape(400, 100, 160, -1)
    valid = valid.reshape(400, 100, 160)
    vectors = vectors.reshape(400, 100, -1)
    tables[name] = validate(tokens, valid, vectors)
    print(name, "validation", json.dumps(tables[name]), flush=True)
    query_games, groups, query_agreement = load_games(name, "queries")
    query_tokens, query_valid, query_vectors = encode(query_games, encoder, scaling)
    refs = tokens.reshape(400, 16000, -1)
    per_game = maxsim(query_tokens, query_valid, refs, valid.reshape(400, 16000))
    questions = len(metadata["question_ids"])
    if not np.array_equal(np.bincount(groups), np.asarray(metadata["query_game_counts"])):
        raise ValueError("Query game counts disagree with extraction metadata")
    index = torch.from_numpy(np.asarray(groups)).to(DEVICE)
    sizes = torch.bincount(index, minlength=questions).float()[:, None]
    match = torch.zeros(questions, 400, device=DEVICE).index_add_(0, index, per_game) / sizes
    pooled = torch.zeros(questions, query_vectors.shape[1], device=DEVICE).index_add_(
        0, index, query_vectors
    )
    cosine = F.normalize(pooled, dim=-1) @ F.normalize(vectors.mean(1), dim=-1).T
    np.savez_compressed(
        ROOT / "rerank" / f"orig_scores_{name}.npz",
        cosine=cosine.cpu().numpy(), match=match.cpu().numpy(),
    )
    report[name] = {
        "engine_rows_placed": [agreement, query_agreement],
        "questions": questions,
        "validation": tables[name],
    }
    states = (cosine, match, metadata["question_ids"], metadata["player_ids"])
    del tokens, valid, vectors, refs, per_game, query_tokens
    torch.cuda.empty_cache()
    return states


def write_answers(report, answers, tables):  # pylint: disable=too-many-locals
    """Selects one fusion weight and CSLS decision, then writes the v6 submission."""
    # One fusion weight and one CSLS decision for both pools: best mean held-out
    # score over all counts and pools.
    keys = list(next(iter(tables.values())))
    mean = {
        key: float(np.mean([np.mean(tables[name][key]["score"]) for name in tables]))
        for key in keys
    }
    choice = max(mean, key=lambda key: mean[key])
    report["held_out_mean"] = {key: round(value, 4) for key, value in mean.items()}
    report["selected"] = choice
    weight = float(choice.split("_")[0][1:])
    with open(SOURCE / "submission_player.csv", encoding="utf-8", newline="") as source:
        table = list(csv.DictReader(source))
    previous = {row["question_id"]: row for row in table}
    new_rows = {}
    for name, (cosine, match, identifiers, players) in answers.items():
        fused = matching.standardize(cosine) + weight * matching.standardize(match)
        if choice.endswith("csls"):
            fused = matching.csls(fused)
        order = fused.argsort(dim=1, descending=True)[:, :5].cpu().numpy()
        plain = cosine.argmax(1).cpu().numpy()
        same = moved = 0
        for position, (identifier, chosen) in enumerate(zip(identifiers, order)):
            picks = [str(players[index]) for index in chosen]
            new_rows[str(identifier)] = picks
            same += previous[str(identifier)]["top_1"] == picks[0]
            moved += chosen[0] != plain[position]
        report[name]["top1_same_as_current_submission"] = f"{same}/{len(order)}"
        report[name]["top1_changed_vs_all_moves_cosine"] = int(moved)
        report[name]["top1_repeat_share"] = 1 - len(np.unique(order[:, 0])) / len(order)
    replaced = 0
    for row in table:
        if row["question_id"] in new_rows:
            for place, player in enumerate(new_rows[row["question_id"]], start=1):
                row[f"top_{place}"] = player
            replaced += 1
    if replaced != len(new_rows):
        raise ValueError("Question ids do not match the submission")
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "submission_player.csv"
    with open(path, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report["rows"], report["replaced"] = len(table), replaced
    report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "counts"}, indent=1))


def main() -> None:
    """Scores the requested original pools (default pub, priv) and writes v6."""
    pools = sys.argv[1:] or ["pub", "priv"]
    encoder, scaling, _, _ = apply.load_model("style_v4")
    encoder.eval()
    report: dict = {"counts": COUNTS}
    answers = {}
    tables: dict = {}
    for name in pools:
        answers[name] = score_pool(name, encoder, scaling, report, tables)
    write_answers(report, answers, tables)


if __name__ == "__main__":
    main()
