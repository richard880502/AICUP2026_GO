"""One-shot test inference: selected models -> log-probability fusion -> rank.

Prerequisites (run once the test CSVs exist; see README):
  scripts/test_to_games.py, scripts/run_extraction96.sh-style dense extraction on features/shards/test
  (96 decisions, rank_5k/1d/5d), scripts/prep_test_moves.py.
  Models: results/d96_seed*.pt, results/move.pt, results/resnet5_epoch{3,4,5}.pt, results/resnet8c_epoch{5..8}.pt.
Fusion weights are re-fitted on the whole validation cohort (results/val_probs_*.npz).
The default ``best2`` model set uses the dense Transformer and cosine ResNet;
``--model-set full4`` preserves the historical four-model pipeline.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import blend_eval
import board_sim  # noqa: F401  (numba compile cache)
import move_transformer as mv
import rank_lib as lib
import resnet_rank as rn
import transformer_rank as tr

ROOT = lib.ROOT


def transformer_probs(feats, device, suffix=""):
    """Dense Transformer: up to 96 decisions per game, KataGo normal + HumanSL rank_5k/1d/5d, 3 seeds averaged."""
    x, valid = tr.decision_tokens(feats["tokens"]), feats["masks"]
    probs = []
    for path in sorted((ROOT / "results").glob(f"d96{suffix}_seed*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = tr.RankTransformer(x.shape[2], x.shape[1], a["d_model"], a["heads"], a["ffn"], a["layers"], a["dropout"]).to(device)
        model.load_state_dict(ckpt["state"])
        xs = np.where(valid[..., None], (x - ckpt["mean"]) / ckpt["std"], 0).astype(np.float32)
        probs.append(tr.predict(model, xs, valid, device))
    assert probs, "no d96 checkpoints found"
    return np.mean(probs, axis=0)


def aligned_moves(path, game_keys):
    z = np.load(path)
    index = {k: i for i, k in enumerate(z["game_key"])}
    rows = np.array([index[k] for k in game_keys])
    return z["tokens"][rows], z["length"][rows], z["colour"][rows]


def move_probs(tokens, lengths, colour, device, suffix=""):
    model = mv.MoveTransformer().to(device)
    model.load_state_dict(torch.load(ROOT / f"results/move{suffix}.pt", map_location="cpu")["state"])
    sym = torch.from_numpy(mv.symmetry_tables()).to(device)
    return mv.predict(model, tokens[:, :256], np.minimum(lengths, 256).astype(np.int16), colour, device, sym)


# Snapshot ensembles: the checkpoint with the lowest new-player NLL is a poor pick for the question-level score, and the
# constant-LR run is unstable from epoch to epoch, so each ResNet is the mean over the later epochs of its training.
SNAPSHOT_EPOCHS = {"resnet5": (3, 4, 5), "resnet8c": (5, 6, 7, 8)}


def resnet_probs(tokens, lengths, colour, device, tag):
    base = tag.split("_h")[0]
    data = rn.Games(tokens, lengths, np.zeros(len(tokens), np.int64), np.arange(len(tokens)), colour)
    probs = []
    for epoch in SNAPSHOT_EPOCHS[base]:
        model = rn.RankNet(in_channels=500, num_classes=10).to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(ROOT / f"results/{tag}_epoch{epoch}.pt", map_location="cpu")["state"])
        probs.append(rn.predict(model, DataLoader(data, batch_size=256, num_workers=8), device))
    return np.mean(probs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
    ap.add_argument("--feat", type=pathlib.Path, default=ROOT / "features/feat96/test", help="dense 96-decision features")
    ap.add_argument("--moves", type=pathlib.Path, default=ROOT / "features/moves/test_moves.npz")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "submission_rank.csv")
    ap.add_argument("--expect", type=int, default=800, help="required number of questions")
    ap.add_argument("--labels", type=pathlib.Path, default=None, help="rehearsal only: question_id,rank to score against")
    ap.add_argument("--pool", default="logmean", choices=("mean", "logmean"))
    ap.add_argument(
        "--model-set",
        default="best2",
        choices=("best2", "full4"),
        help="best2 is the selected dense+cosine-ResNet pipeline; full4 is the historical ensemble",
    )
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced decode (each rank predicted equally often). Only valid if the test set is balanced; OFF by default")
    ap.add_argument("--use-aggregator", action="store_true", help="use the learned question aggregator (off by default: no gain on the holdout)")
    ap.add_argument("--no-offsets", action="store_true", help="skip the validation-learned per-rank bias correction")
    ap.add_argument("--suffix", default="", help="model tag suffix, e.g. _h for the holdout-excluded retrains")
    ap.add_argument("--dump", type=pathlib.Path, default=None, help="save per-game probabilities of every model (for eval_holdout.py)")
    ap.add_argument("--aggregator", type=pathlib.Path, default=None,
                    help="learned question aggregator; falls back to weighted fusion + pooling if the file is missing")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    feats = lib.load_features(args.feat)
    keys = feats["game_key"]
    tokens, lengths, colour = aligned_moves(args.moves, keys)
    sfx = args.suffix
    per_model = {f"d96{sfx}": transformer_probs(feats, device, sfx)}
    if args.model_set == "full4":
        per_model[f"move{sfx}"] = move_probs(
            tokens,
            lengths,
            colour,
            device,
            sfx,
        )
        per_model[f"resnet5{sfx}"] = resnet_probs(
            tokens,
            lengths,
            colour,
            device,
            f"resnet5{sfx}",
        )
    per_model[f"resnet8c{sfx}"] = resnet_probs(
        tokens,
        lengths,
        colour,
        device,
        f"resnet8c{sfx}",
    )
    if args.dump is not None:
        np.savez(args.dump, game_key=feats["game_key"], player_id=feats["player_id"], **per_model)
        print("dumped per-game probabilities ->", args.dump)
    models = list(per_model)
    val_logs = []
    for m in models:
        z = np.load(ROOT / f"results/val_probs_{m}.npz", allow_pickle=True)
        val_logs.append(np.log(np.clip(z["transformer" if m.startswith(("transformer_v", "d96")) else m], 1e-6, 1.0)))
        y, val_player = z["y"], z["player"]
    weights = blend_eval.fit_weights(np.stack(val_logs), y)
    print("fusion weights:", dict(zip(models, weights.round(3))))
    game_prob = blend_eval.fuse(np.stack([np.log(np.clip(per_model[m], 1e-6, 1.0)) for m in models]), weights)

    qids = pd.concat([pd.read_csv(args.tests_dir / f"rank_prediction_test_{s}.csv", usecols=["question_id"])
                      for s in ("public", "private")])["question_id"].tolist()
    by_question = {}
    for i, pid in enumerate(feats["player_id"]):
        by_question.setdefault(pid, []).append(i)
    aggregator = None
    agg_path = args.aggregator or ROOT / f"results/aggregator{sfx}.pkl"
    if args.use_aggregator and agg_path.exists():
        import pickle
        aggregator = pickle.load(open(agg_path, "rb"))
        assert aggregator["models"] == models, f"aggregator expects {aggregator['models']}, have {models}"
        game_logs = np.stack([np.log(np.clip(per_model[m], 1e-6, 1.0)) for m in models])
        print("using learned aggregator", agg_path)
    qprob = []
    for qid in qids:
        idx = by_question.get(qid)
        if not idx:
            p = np.full(10, 0.1)  # no parsable game: uninformative
        elif aggregator is not None:
            f = np.concatenate([game_logs[:, idx].mean(1).reshape(-1), [1 / np.sqrt(len(idx)), np.log(len(idx))]])[None]
            p = aggregator["clf"].predict_proba(aggregator["scaler"].transform(f))[0]
        else:
            p = lib.pool(game_prob[idx], args.pool)
        qprob.append(p)
    qprob = np.stack(qprob)
    if aggregator is None and not args.no_offsets:
        # per-rank bias correction learned on validation questions only (10 random 5-20 game questions per player)
        vfused = blend_eval.fuse(np.stack(val_logs), weights)
        rng = np.random.default_rng(0)
        vq = []
        for _ in range(10):
            for pid in np.unique(val_player):
                rows_ = np.flatnonzero(val_player == pid)
                pick = rng.choice(rows_, size=min(int(rng.integers(5, 21)), len(rows_)), replace=False)
                vq.append(lib.pool(vfused[pick], args.pool))
        offsets = lib.fit_class_offsets(np.stack(vq))
        print("per-rank offsets:", dict(zip(lib.LEVELS, offsets.round(2))))
        qprob = lib.apply_class_offsets(qprob, offsets)
    value = qprob @ (np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1)))  # expected score per class
    if args.balanced:
        from scipy.optimize import linear_sum_assignment
        per = -(-len(qids) // 10)
        r_, c_ = linear_sum_assignment(-np.repeat(value, per, axis=1))
        pred_all = np.empty(len(qids), int); pred_all[r_] = np.minimum(c_ // per, 9)
        print("class-balanced decode ON")
    else:
        pred_all = value.argmax(1)
    rows = [(qid, lib.LEVELS[int(pr)]) for qid, pr in zip(qids, pred_all)]
    sub = pd.DataFrame(rows, columns=["question_id", "pred_rank"])
    assert len(sub) == args.expect and sub.question_id.is_unique, f"expected {args.expect} unique questions, got {len(sub)}"
    sub.to_csv(args.out, index=False)
    print("wrote", args.out, sub.pred_rank.value_counts().to_dict())
    if args.labels is not None:
        truth = pd.read_csv(args.labels).set_index("question_id")["rank"]
        pred_i = np.array([lib.LEVELS.index(r) for r in sub["pred_rank"]])
        true_i = np.array([lib.LEVELS.index(truth[q]) for q in sub["question_id"]])
        print(f"rehearsal score {lib.competition_score(pred_i, true_i):.4f} on {len(sub)} questions")


if __name__ == "__main__":
    main()
