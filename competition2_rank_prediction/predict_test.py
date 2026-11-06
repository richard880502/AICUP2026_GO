"""One-shot test inference: selected models -> log-probability fusion -> rank.

Prerequisites (run once the test CSVs exist; see INDEX.md):
  scripts/test_to_games.py, scripts/run_test_extraction.sh (dense 96-decision extraction of the
  player's AND the opponent's decisions on features/shards/test -> features/feat96/test,
  features/feat96opp/test, plus final-position features -> features/final_result/test),
  scripts/prep_test_moves.py.
  Models: results/d96_seed*.pt, results/resnet8c_epoch{5..8}.pt, results/opp_scratch_seed*.pt
  (full4 also needs results/move.pt and results/resnet5_epoch{3,4,5}.pt).
Fusion weights are re-fitted on the whole validation cohort (results/val_probs_*.npz).
The default ``best2`` model set uses the dense Transformer and cosine ResNet, plus an opponent
member (since 2026-10-21 the opponent-specific Transformer opp_scratch; ``--opp-model d96`` =
2026-10-20 default, ``--no-opp`` = old two-model pipeline), and the question-level shift residual
(residual_shift.py; ``--no-residual`` = per-rank offsets instead); ``--model-set full4`` preserves
the historical four-model pipeline.
"""
import argparse
import dataclasses
import math
import pathlib
import pickle

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

import blend_eval
import board_sim  # pylint: disable=unused-import  # noqa: F401  (numba compile cache)
import move_transformer as mv
import rank_lib as lib
import residual_shift as rs
import resnet_rank as rn
import transformer_rank as tr

ROOT = lib.ROOT
SCORE_MATRIX = np.eye(10) + lib.ADJACENT * (np.eye(10, k=1) + np.eye(10, k=-1))
# Snapshot ensembles: the checkpoint with the lowest new-player NLL is a poor pick for the
# question-level score, and the constant-LR run is unstable from epoch to epoch, so each ResNet is
# the mean over the later epochs of its training.
SNAPSHOT_EPOCHS = {"resnet5": (3, 4, 5), "resnet8c": (5, 6, 7, 8)}
# A question without any usable game: under balanced labels every rank from C to 5D has the same
# expected score, so take the middle one.
PRIOR_RANK = lib.LEVELS.index("1D")


def transformer_probs(feats: dict, device: torch.device, suffix: str = "",
                      tag: str = "d96") -> np.ndarray:
    """Dense Transformer (up to 96 decisions per game, KataGo normal + HumanSL rank_5k/1d/5d):
    mean probabilities of results/<tag><suffix>_seed*.pt."""
    x, valid = tr.decision_tokens(feats["tokens"]), feats["masks"]
    probs = []
    for path in sorted((ROOT / "results").glob(f"{tag}{suffix}_seed*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hyper = ckpt["args"]
        model = tr.RankTransformer(x.shape[2], x.shape[1], hyper["d_model"], hyper["heads"],
                                   hyper["ffn"], hyper["layers"], hyper["dropout"]).to(device)
        model.load_state_dict(ckpt["state"])
        scaled = np.where(valid[..., None], (x - ckpt["mean"]) / ckpt["std"], 0)
        probs.append(tr.predict(model, scaled.astype(np.float32), valid, device))
    assert probs, f"no {tag}{suffix} checkpoints found"
    return np.mean(probs, axis=0)


def opponent_probs(opp_feats: dict, game_keys: np.ndarray, device: torch.device,
                   suffix: str = "", tag: str = "d96") -> np.ndarray:
    """A dense Transformer on the opponent's decisions of each game, rows in the order of game_keys.

    Opponents are matched at about the player's rank, so this is extra evidence about the
    labelled player. Games without any opponent decision get a uniform distribution.
    """
    index = {k: i for i, k in enumerate(opp_feats["game_key"])}
    missing = [k for k in game_keys if k not in index]
    assert not missing, f"{len(missing)} games lack opponent features, e.g. {missing[:3]}"
    rows = np.array([index[k] for k in game_keys])
    masks = opp_feats["masks"][rows].copy()
    empty = ~masks.any(1)
    masks[empty, 0] = True  # avoid an all-masked softmax; those games are set to uniform below
    probs = transformer_probs({"tokens": opp_feats["tokens"][rows], "masks": masks}, device,
                              suffix, tag)
    probs[empty] = 0.1
    return probs


def aligned_moves(path: pathlib.Path,
                  game_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move tokens, lengths and the player's colour of game_keys from a prep_*moves.py archive."""
    data = np.load(path)
    index = {k: i for i, k in enumerate(data["game_key"])}
    rows = np.array([index[k] for k in game_keys])
    return data["tokens"][rows], data["length"][rows], data["colour"][rows]


def move_probs(moves: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device,
               suffix: str = "") -> np.ndarray:
    """Historical move-sequence Transformer (full4 only)."""
    tokens, lengths, colour = moves
    model = mv.MoveTransformer().to(device)
    state = torch.load(ROOT / f"results/move{suffix}.pt", map_location="cpu")["state"]
    model.load_state_dict(state)
    sym = torch.from_numpy(mv.symmetry_tables()).to(device)
    return mv.predict(model, tokens[:, :256], np.minimum(lengths, 256).astype(np.int16), colour,
                      device, sym)


def resnet_probs(tokens: np.ndarray, lengths: np.ndarray, colour: np.ndarray,
                 device: torch.device, tag: str) -> np.ndarray:
    """Snapshot-mean probabilities of a ResNet (tag e.g. resnet8c, resnet8c_h2, resnet8c_c0)."""
    base = tag.split("_")[0]
    data = rn.Games(tokens, lengths, np.zeros(len(tokens), np.int64), np.arange(len(tokens)),
                    colour)
    probs = []
    for epoch in SNAPSHOT_EPOCHS[base]:
        model = rn.RankNet(in_channels=500, num_classes=10).to(device)
        model = model.to(memory_format=torch.channels_last)
        state = torch.load(ROOT / f"results/{tag}_epoch{epoch}.pt", map_location="cpu")["state"]
        model.load_state_dict(state)
        probs.append(rn.predict(model, DataLoader(data, batch_size=256, num_workers=8), device))
    return np.mean(probs, axis=0)


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
    ap.add_argument("--feat", type=pathlib.Path, default=ROOT / "features/feat96/test",
                    help="dense 96-decision features")
    ap.add_argument("--moves", type=pathlib.Path,
                    default=ROOT / "features/moves/test_moves.npz")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "submission_rank.csv")
    ap.add_argument("--expect", type=int, default=800, help="required number of questions")
    ap.add_argument("--labels", type=pathlib.Path, default=None,
                    help="rehearsal only: question_id,rank to score against")
    ap.add_argument("--pool", default="logmean", choices=("mean", "logmean"))
    ap.add_argument("--model-set", default="best2", choices=("best2", "full4"),
                    help="best2 is the selected dense+cosine-ResNet pipeline; "
                         "full4 is the historical ensemble")
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced decode (each rank predicted equally often). "
                         "Only valid if the test set is balanced; OFF by default")
    ap.add_argument("--use-aggregator", action="store_true",
                    help="use the learned question aggregator (off by default: no holdout gain)")
    ap.add_argument("--no-offsets", action="store_true",
                    help="skip the validation-learned per-rank bias correction")
    ap.add_argument("--suffix", default="",
                    help="model tag suffix, e.g. _h for the holdout-excluded retrains")
    ap.add_argument("--dump", type=pathlib.Path, default=None,
                    help="save per-game probabilities of every model (for evaluations)")
    ap.add_argument("--aggregator", type=pathlib.Path, default=None,
                    help="learned question aggregator; falls back to weighted fusion + pooling "
                         "if the file is missing")
    ap.add_argument("--opp-feat", type=pathlib.Path, default=ROOT / "features/feat96opp/test",
                    help="dense features of the OPPONENT's decisions "
                         "(extract_rank_features.py --opponent)")
    ap.add_argument("--no-opp", action="store_true",
                    help="no opponent member (the 2026-10-12 best2 = 0.6609 on the holdout)")
    ap.add_argument("--opp-model", default="opp_scratch",
                    help="opponent member: opp_scratch = Transformer trained to predict the "
                         "player's rank from the opponent's moves (default since 2026-10-21, C2); "
                         "d96 = the player model applied to the opponent's moves")
    ap.add_argument("--final-result", type=pathlib.Path,
                    default=ROOT / "features/final_result/test",
                    help="KataGo final-position features of the test games "
                         "(scripts/extract_final_result.py), for the shift residual")
    ap.add_argument("--val-final-result", type=pathlib.Path,
                    default=ROOT / "features/final_result/val")
    ap.add_argument("--no-residual", action="store_true",
                    help="skip the question-level shift residual (C1, default since 2026-10-21) "
                         "and use the per-rank offsets instead")
    ap.add_argument("--opp-val-feat", type=pathlib.Path,
                    default=ROOT / "features/feat96opp/val",
                    help="opponent features of the validation cohort (for --opp-model d96)")
    ap.add_argument("--report", type=pathlib.Path, default=None,
                    help="per-question method / fallback report (default: <out>_report.csv)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero (after writing everything) if any question needed a "
                         "fallback")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def member_probs(args: argparse.Namespace, feats: dict,
                 device: torch.device) -> dict[str, np.ndarray]:
    """Per-game class probabilities of every fusion member, in the order of feats['game_key']
    (legacy path: --model-set full4 / --use-aggregator; no fallbacks)."""
    keys = feats["game_key"]
    tokens, lengths, colour = aligned_moves(args.moves, keys)
    sfx = args.suffix
    per_model = {f"d96{sfx}": transformer_probs(feats, device, sfx)}
    if args.model_set == "full4":
        per_model[f"move{sfx}"] = move_probs((tokens, lengths, colour), device, sfx)
        per_model[f"resnet5{sfx}"] = resnet_probs(tokens, lengths, colour, device,
                                                  f"resnet5{sfx}")
    per_model[f"resnet8c{sfx}"] = resnet_probs(tokens, lengths, colour, device, f"resnet8c{sfx}")
    if not args.no_opp:
        if not list(args.opp_feat.glob("feat_*.npz")):
            raise SystemExit(f"no opponent features in {args.opp_feat}: run the --opponent "
                             "extraction (scripts/run_test_extraction.sh) or pass --no-opp")
        member = f"d96{sfx}_opp" if args.opp_model == "d96" else f"{args.opp_model}{sfx}"
        per_model[member] = opponent_probs(lib.load_features(args.opp_feat), keys, device, sfx,
                                           args.opp_model)
        val_file = ROOT / f"results/val_probs_{member}.npz"
        if not val_file.exists():  # the same models on the validation opponents, cached
            assert args.opp_model == "d96", \
                f"missing {val_file} (written by transformer_rank.py when {member} is trained)"
            ref = np.load(ROOT / f"results/val_probs_d96{sfx}.npz", allow_pickle=True)
            np.savez(val_file, y=ref["y"], player=ref["player"], game_key=ref["game_key"],
                     transformer=opponent_probs(lib.load_features(args.opp_val_feat),
                                                ref["game_key"], device, sfx))
    return per_model


def validation_logs(models: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Members' validation log-probabilities aligned by game key to the first member's order.
    Returns (logs [M, G, 10], game keys, labels, players)."""
    logs, keys, labels, players = [], np.zeros(0), np.zeros(0), np.zeros(0)
    for i, member in enumerate(models):
        data = np.load(ROOT / f"results/val_probs_{member}.npz", allow_pickle=True)
        transformer = member.startswith(("transformer_v", "d96", "opp_"))
        prob = data["transformer" if transformer else member]
        if i == 0:
            keys, labels, players = data["game_key"], data["y"], data["player"]
        else:  # members can be saved in different game orders (e.g. the opponent-feature order)
            index = {k: n for n, k in enumerate(data["game_key"])}
            prob = prob[[index[k] for k in keys]]
        logs.append(np.log(np.clip(prob, 1e-6, 1.0)))
    return np.stack(logs), keys, labels, players


def fit_residual(args: argparse.Namespace, weights: np.ndarray,
                 val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> rs.ShiftResidual:
    """C1: the question-level shift residual fitted on the validation cohort (player member =
    first, opponent member = last of val's members)."""
    val_fin = rs.load_final(args.val_final_result)
    return rs.fit_on_validation(val[0], weights, (val[2], val[3]),
                                (np.exp(val[0][0]), np.exp(val[0][-1])),
                                np.stack([val_fin[k] for k in val[1]]))


def residual_probs(args: argparse.Namespace, fit: tuple[np.ndarray, np.ndarray, np.ndarray],
                   val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                   questions: list[np.ndarray], game_keys: np.ndarray) -> np.ndarray:
    """C1 on the legacy path: fit = (fusion weights, per-game fused test probabilities,
    [test player-member probs, test opponent-member probs] stacked)."""
    residual = fit_residual(args, fit[0], val)
    test_fin = rs.load_final(args.final_result)
    missing = [k for k in game_keys if k not in test_fin]
    assert not missing, f"{len(missing)} test games lack final-result features: {missing[:3]}"
    z, x = rs.question_inputs(questions, fit[1], fit[2][0], fit[2][1],
                              np.stack([test_fin[k] for k in game_keys]))
    return residual.predict_proba(z, x)


def offset_probs(qprob: np.ndarray, val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 weights: np.ndarray, pool: str) -> np.ndarray:
    """Per-rank bias correction learned on validation questions (10 random 5-20 game questions
    per player)."""
    vfused = blend_eval.fuse(val[0], weights)
    val_player = val[3]
    rng = np.random.default_rng(0)
    vq = []
    for _ in range(10):
        for pid in np.unique(val_player):
            rows = np.flatnonzero(val_player == pid)
            pick = rng.choice(rows, size=min(int(rng.integers(5, 21)), len(rows)), replace=False)
            vq.append(lib.pool(vfused[pick], pool))
    offsets = lib.fit_class_offsets(np.stack(vq))
    print("per-rank offsets:", dict(zip(lib.LEVELS, offsets.round(2))))
    return lib.apply_class_offsets(qprob, offsets)


def decode(qprob: np.ndarray, balanced: bool) -> np.ndarray:
    """Expected-competition-score decision (optionally class-balanced)."""
    value = qprob @ SCORE_MATRIX
    if not balanced:
        return value.argmax(1)
    per = -(-len(qprob) // 10)
    rows, cols = linear_sum_assignment(-np.repeat(value, per, axis=1))
    pred = np.empty(len(qprob), int)
    pred[rows] = np.minimum(cols // per, 9)
    print("class-balanced decode ON")
    return pred


def load_aggregator(args: argparse.Namespace, models: list[str]) -> dict | None:
    """The optional learned question aggregator (off by default)."""
    agg_path = args.aggregator or ROOT / f"results/aggregator{args.suffix}.pkl"
    if not (args.use_aggregator and agg_path.exists()):
        return None
    with open(agg_path, "rb") as handle:
        aggregator = pickle.load(handle)
    assert aggregator["models"] == models, f"aggregator expects {aggregator['models']}"
    print("using learned aggregator", agg_path)
    return aggregator


def pooled_probs(qids: list[str], by_question: dict[str, list[int]],
                 games: tuple[np.ndarray, np.ndarray], aggregator: dict | None,
                 pool: str) -> np.ndarray:
    """Question probabilities from the per-game member logs / fused probabilities."""
    game_logs, game_prob = games
    qprob = []
    for qid in qids:
        idx = by_question.get(qid)
        if not idx:
            qprob.append(np.full(10, 0.1))  # no parsable game: uninformative
        elif aggregator is not None:
            f = np.concatenate([game_logs[:, idx].mean(1).reshape(-1),
                                [1 / np.sqrt(len(idx)), np.log(len(idx))]])[None]
            qprob.append(aggregator["clf"].predict_proba(aggregator["scaler"].transform(f))[0])
        else:
            qprob.append(lib.pool(game_prob[idx], pool))
    return np.stack(qprob)


def read_tests(tests_dir: pathlib.Path) -> tuple[list[str], dict[str, int], list[str]]:
    """Question ids in file order (duplicates dropped), the number of non-empty SGF columns of
    each question, and run-level notes."""
    tests = pd.concat([pd.read_csv(tests_dir / f"rank_prediction_test_{s}.csv",
                                   dtype={"question_id": str})
                       for s in ("public", "private")], ignore_index=True)
    sgf_cols = [c for c in tests.columns if c.startswith("sgf_")]
    listed = tests[sgf_cols].apply(lambda col: col.astype(str).str.strip().ne("")
                                   & col.notna()).sum(1)
    qids: list[str] = []
    n_listed: dict[str, int] = {}
    for qid, n in zip(tests["question_id"].astype(str), listed):
        if qid not in n_listed:
            qids.append(qid)
            n_listed[qid] = int(n)
    notes = [f"{len(tests) - len(qids)} duplicate question rows ignored"] \
        if len(qids) < len(tests) else []
    return qids, n_listed, notes


def _optional(load, what: str, notes: list[str]):
    """load() or None (with a note) when the input is missing or unreadable."""
    try:
        return load()
    except (OSError, KeyError, ValueError) as err:  # FileNotFoundError is an OSError
        notes.append(f"{what} unavailable ({type(err).__name__}: {err})")
        return None


def _scatter(n: int, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    """values placed at rows of an [n, ...] array of NaN (values itself when rows = 0..n-1)."""
    if len(rows) == n and (rows == np.arange(n)).all():
        return values
    out = np.full((n,) + values.shape[1:], np.nan, values.dtype)
    out[rows] = values
    return out


@dataclasses.dataclass
class GameTable:
    """Every test game found in any input, with each member's probabilities (NaN rows where the
    member has no input) and per-game usability."""

    keys: np.ndarray
    question: np.ndarray
    probs: dict[str, np.ndarray]
    usable: dict[str, np.ndarray]
    fin: np.ndarray
    fin_ok: np.ndarray


def _universe(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Game keys (player-feature order first, so normal runs keep the exact historical order)
    and their question ids (the part of the key before the last '#')."""
    own = inputs["own"]
    keys = list(own["game_key"]) if own is not None else []
    seen = set(keys)
    for extra in (inputs["moves"], inputs["opp"]):
        for k in (extra["game_key"] if extra is not None else []):
            if str(k) not in seen:
                seen.add(str(k))
                keys.append(str(k))
    arr = own["game_key"] if own is not None and len(keys) == len(own["game_key"]) \
        else np.array(keys, dtype=str)
    return arr, np.array([k.rsplit("#", 1)[0] for k in arr], dtype=str)


def _player_member(own: dict | None, keys: np.ndarray, index: dict[str, int],
                   device: torch.device, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """d96 on the player's decisions; games without any decision or with non-finite output are
    unusable."""
    n = len(keys)
    if own is None or len(own["game_key"]) == 0:
        return np.full((n, 10), np.nan, np.float32), np.zeros(n, bool)
    rows = np.array([index[k] for k in own["game_key"]])
    masks = own["masks"]
    empty = ~masks.any(1)
    if empty.any():  # avoid an all-masked softmax (NaN); those games are marked unusable
        masks = masks.copy()
        masks[empty, 0] = True
    probs = transformer_probs({"tokens": own["tokens"], "masks": masks}, device, suffix)
    usable = np.zeros(n, bool)
    usable[rows] = ~empty & np.isfinite(probs).all(1)
    return _scatter(n, rows, probs), usable


def _resnet_member(moves: dict | None, keys: np.ndarray, device: torch.device,
                   tag: str) -> tuple[np.ndarray, np.ndarray]:
    """The cosine ResNet on the move tokens; games without tokens / moves are unusable."""
    n = len(keys)
    lookup = {str(k): i for i, k in enumerate(moves["game_key"])} if moves is not None else {}
    rows = np.array([i for i, k in enumerate(keys) if k in lookup], dtype=int)
    if len(rows) == 0:
        return np.full((n, 10), np.nan, np.float32), np.zeros(n, bool)
    assert moves is not None
    src = np.array([lookup[keys[i]] for i in rows])
    probs = resnet_probs(moves["tokens"][src], moves["length"][src], moves["colour"][src],
                         device, tag)
    usable = np.zeros(n, bool)
    usable[rows] = (moves["length"][src] > 0) & np.isfinite(probs).all(1)
    return _scatter(n, rows, probs), usable


def _opponent_member(opp: dict | None, keys: np.ndarray, device: torch.device,
                     args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """The opponent member; games without opponent features / decisions are unusable."""
    n = len(keys)
    lookup = {str(k): i for i, k in enumerate(opp["game_key"])} if opp is not None else {}
    rows = np.array([i for i, k in enumerate(keys) if k in lookup], dtype=int)
    if len(rows) == 0:
        return np.full((n, 10), np.nan, np.float32), np.zeros(n, bool)
    assert opp is not None
    probs = opponent_probs(opp, keys[rows], device, args.suffix, args.opp_model)
    decided = opp["masks"][[lookup[keys[i]] for i in rows]].any(1)
    usable = np.zeros(n, bool)
    usable[rows] = decided & np.isfinite(probs).all(1)
    return _scatter(n, rows, probs), usable


def _final_rows(test_fin: dict | None, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[G, 6] final-position features (NaN rows where missing) and which games have them."""
    n = len(keys)
    if test_fin is None:
        return np.full((n, len(rs.FINAL_NAMES)), np.nan, np.float32), np.zeros(n, bool)
    # a game KataGo rejected (or with no move) has NaN values: treat it as missing
    have = np.array([k in test_fin and bool(np.isfinite(test_fin[k]).all()) for k in keys], bool)
    if have.all() and n:
        return np.stack([test_fin[k] for k in keys]), have
    fin = np.full((n, len(rs.FINAL_NAMES)), np.nan, np.float32)
    for i in np.flatnonzero(have):
        fin[i] = test_fin[keys[i]]
    return fin, have


def build_table(args: argparse.Namespace, names: list[str], device: torch.device,
                notes: list[str]) -> GameTable:
    """Loads whatever inputs exist and computes each member on the games it has input for."""
    inputs = {"own": _optional(lambda: lib.load_features(args.feat), "player features", notes),
              "moves": _optional(lambda: dict(np.load(args.moves)), "move tokens", notes),
              "opp": None if len(names) < 3 else _optional(
                  lambda: lib.load_features(args.opp_feat), "opponent features", notes)}
    keys, question = _universe(inputs)
    index = {k: i for i, k in enumerate(keys)}
    members = [_player_member(inputs["own"], keys, index, device, args.suffix),
               _resnet_member(inputs["moves"], keys, device, names[1])]
    if len(names) > 2:
        members.append(_opponent_member(inputs["opp"], keys, device, args))
    test_fin = None if args.no_residual or len(names) < 3 else _optional(
        lambda: rs.load_final(args.final_result), "final-result features", notes)
    fin, fin_ok = _final_rows(test_fin, keys)
    return GameTable(keys, question, {m: p for m, (p, _) in zip(names, members)},
                     {m: u for m, (_, u) in zip(names, members)}, fin, fin_ok)


@dataclasses.dataclass(frozen=True)
class Method:
    """A way to score a question: fusion members (in the validated member order), then the shift
    residual or the validation per-rank offsets."""

    label: str
    members: tuple[str, ...]
    residual: bool = False


def method_chain(names: list[str], residual: bool) -> list[Method]:
    """Methods from the confirmed default down to single members (validated strength order)."""
    player, resnet = names[0], names[1]
    chain = []
    if len(names) > 2:
        opp = names[2]
        if residual:
            chain.append(Method("default (C1+C2)", (player, resnet, opp), True))
        chain += [Method("3-member fusion + offsets", (player, resnet, opp)),
                  Method("player + opponent", (player, opp)),
                  Method("resnet + opponent", (resnet, opp))]
    chain.append(Method("best2", (player, resnet)))
    if len(names) > 2:
        chain.append(Method("opponent only", (names[2],)))
    chain += [Method("player only", (player,)), Method("resnet only", (resnet,))]
    return chain


class Scorer:  # pylint: disable=too-few-public-methods
    """Question probabilities of one method; weights, offsets and the residual are fitted on the
    validation cohort exactly as in the confirmed pipeline (and only when a method is used)."""

    def __init__(self, args: argparse.Namespace, names: list[str], table: GameTable):
        self.args, self.names, self.table = args, names, table
        self.val = validation_logs(names)
        self.weights: dict[tuple[str, ...], np.ndarray] = {}

    def _fit(self, members: tuple[str, ...]) -> tuple[np.ndarray, tuple]:
        idx = [self.names.index(m) for m in members]
        val = self.val if len(idx) == len(self.names) else \
            (self.val[0][idx], self.val[1], self.val[2], self.val[3])
        if members not in self.weights:
            self.weights[members] = blend_eval.fit_weights(val[0], val[2])
            print(f"fusion weights ({', '.join(members)}):",
                  dict(zip(members, self.weights[members].round(3))))
        return self.weights[members], val

    def __call__(self, method: Method, questions: list[np.ndarray]) -> np.ndarray:
        weights, val = self._fit(method.members)
        logs = np.stack([np.log(np.clip(self.table.probs[m], 1e-6, 1.0))
                         for m in method.members])
        game_prob = blend_eval.fuse(logs, weights)
        if method.residual:
            residual = fit_residual(self.args, weights, val)
            z, x = rs.question_inputs(questions, game_prob, self.table.probs[method.members[0]],
                                      self.table.probs[method.members[-1]], self.table.fin)
            print(f"shift residual on {len(questions)} questions "
                  f"(members: player {method.members[0]}, opponent {method.members[-1]})")
            return residual.predict_proba(z, x)
        qprob = np.stack([lib.pool(game_prob[g], self.args.pool) for g in questions])
        if self.args.no_offsets:
            return qprob
        return offset_probs(qprob, val, weights, self.args.pool)


def _groups(table: GameTable) -> dict[str, np.ndarray]:
    """Question id -> its game rows in the table (table order)."""
    by_question: dict[str, list[int]] = {}
    for i, q in enumerate(table.question):
        by_question.setdefault(q, []).append(i)
    return {q: np.array(rows, dtype=int) for q, rows in by_question.items()}


def _eligible(method: Method, table: GameTable, open_questions: list[tuple[int, np.ndarray]],
              any_ok: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """(question index, games the method can use) for every open question where that is at least
    half of the question's usable games."""
    need = np.ones(len(table.keys), bool)
    for member in method.members:
        need &= table.usable[member]
    if method.residual:
        need &= table.fin_ok
    picked = []
    for qi, games in open_questions:
        n_any = int(any_ok[games].sum())
        if n_any and need[games].sum() >= max(1, math.ceil(n_any / 2)):
            picked.append((qi, games[need[games]]))
    return picked


def choose_methods(qids: list[str], table: GameTable, chain: list[Method],
                   scorer: Scorer) -> tuple[np.ndarray, list[str], list[int]]:
    """For every question, the first method of the chain that can use at least half of the
    question's usable games (and gives finite probabilities); games a method cannot use are left
    out. Questions without any usable game get no method ('prior')."""
    groups = _groups(table)
    any_ok = np.zeros(len(table.keys), bool)
    for usable in table.usable.values():
        any_ok |= usable
    qprob = np.full((len(qids), 10), 0.1)
    method_of, used = ["prior"] * len(qids), [0] * len(qids)
    for method in chain:
        picked = _eligible(method, table, [(qi, groups.get(q, np.zeros(0, dtype=int)))
                                           for qi, q in enumerate(qids)
                                           if method_of[qi] == "prior"], any_ok)
        if picked:
            for (qi, use), p in zip(picked, scorer(method, [use for _, use in picked])):
                if np.isfinite(p).all() and p.sum() > 0:
                    qprob[qi], method_of[qi], used[qi] = p, method.label, len(use)
    return qprob, method_of, used


def question_report(qids: list[str], n_listed: dict[str, int], table: GameTable,
                    chosen: tuple[list[str], list[int]], top: Method) -> pd.DataFrame:
    """One row per question: method, games listed / found / used, and what was missing."""
    groups = _groups(table)
    rows = []
    for qid, method, n_used in zip(qids, *chosen):
        games = groups.get(qid, np.zeros(0, dtype=int))
        missing = [f"{int((~table.usable[m][games]).sum())} without {m}"
                   for m in top.members if len(games) and (~table.usable[m][games]).any()]
        if top.residual and len(games) and (~table.fin_ok[games]).any():
            missing.append(f"{int((~table.fin_ok[games]).sum())} without final result")
        if n_listed.get(qid, 0) > len(games):
            missing.insert(0, f"{n_listed[qid] - len(games)} listed games not in any input")
        rows.append({"question_id": qid, "method": method, "fallback": method != top.label,
                     "games_listed": n_listed.get(qid, 0), "games_found": len(games),
                     "games_used": n_used, "missing": "; ".join(missing)})
    return pd.DataFrame(rows)


def legacy_question_probs(args: argparse.Namespace, device: torch.device,  # pylint: disable=too-many-locals
                          qids: list[str]) -> np.ndarray:
    """--model-set full4 / --use-aggregator: the historical pipeline, without fallbacks."""
    feats = lib.load_features(args.feat)
    per_model = member_probs(args, feats, device)
    if args.dump is not None:
        np.savez(args.dump, game_key=feats["game_key"], player_id=feats["player_id"],
                 **per_model)  # type: ignore[arg-type]
    models = list(per_model)
    val = validation_logs(models)
    weights = blend_eval.fit_weights(val[0], val[2])
    print("fusion weights:", dict(zip(models, weights.round(3))))
    game_logs = np.stack([np.log(np.clip(per_model[m], 1e-6, 1.0)) for m in models])
    game_prob = blend_eval.fuse(game_logs, weights)
    by_question: dict[str, list[int]] = {}
    for i, pid in enumerate(feats["player_id"]):
        by_question.setdefault(pid, []).append(i)
    aggregator = load_aggregator(args, models)
    qprob_arr = pooled_probs(qids, by_question, (game_logs, game_prob), aggregator, args.pool)
    use_residual = aggregator is None and not args.no_residual and not args.no_opp
    if use_residual:
        have = [i for i, q in enumerate(qids) if by_question.get(q)]
        questions = [np.array(by_question[qids[i]]) for i in have]
        members = (per_model[models[0]], per_model[models[-1]])
        qprob_arr[have] = residual_probs(args, (weights, game_prob, np.stack(members)), val,
                                         questions, feats["game_key"])
    if aggregator is None and not args.no_offsets and not use_residual:
        qprob_arr = offset_probs(qprob_arr, val, weights, args.pool)
    return qprob_arr


def member_names(args: argparse.Namespace) -> list[str]:
    """Fusion members of the best2 pipeline, in the validated order."""
    sfx = args.suffix
    names = [f"d96{sfx}", f"resnet8c{sfx}"]
    if not args.no_opp:
        names.append(f"d96{sfx}_opp" if args.opp_model == "d96" else f"{args.opp_model}{sfx}")
        val_file = ROOT / f"results/val_probs_{names[2]}.npz"
        if not val_file.exists() and args.opp_model == "d96":  # cached once, as before
            ref = np.load(ROOT / f"results/val_probs_d96{sfx}.npz", allow_pickle=True)
            np.savez(val_file, y=ref["y"], player=ref["player"], game_key=ref["game_key"],
                     transformer=opponent_probs(lib.load_features(args.opp_val_feat),
                                                ref["game_key"], torch.device(args.device), sfx))
    return names


def robust_question_probs(args: argparse.Namespace, device: torch.device, qids: list[str],
                          n_listed: dict[str, int], notes: list[str]) -> tuple:
    """best2 (+ opponent + residual) with the per-question fallback chain."""
    names = member_names(args)
    table = build_table(args, names, device, notes)
    if args.dump is not None:
        np.savez(args.dump, game_key=table.keys, player_id=table.question,
                 **table.probs)  # type: ignore[arg-type]
        print("dumped per-game probabilities ->", args.dump)
    residual_ok = not args.no_residual and len(names) > 2
    if residual_ok and not args.val_final_result.exists():
        notes.append(f"validation final results missing ({args.val_final_result}): no residual")
        residual_ok = False
    chain = method_chain(names, residual_ok)
    qprob, method_of, used = choose_methods(qids, table, chain, Scorer(args, names, table))
    report = question_report(qids, n_listed, table, (method_of, used), chain[0])
    return qprob, report


def main() -> None:
    """Writes the submission and the per-question report (and scores it when --labels is
    given)."""
    args = parse_args()
    device = torch.device(args.device)
    qids, n_listed, notes = read_tests(args.tests_dir)
    if args.model_set == "full4" or args.use_aggregator:
        qprob_arr = legacy_question_probs(args, device, qids)
        report = pd.DataFrame({"question_id": qids, "method": "legacy", "fallback": False})
    else:
        qprob_arr, report = robust_question_probs(args, device, qids, n_listed, notes)
    pred_all = decode(qprob_arr, args.balanced)
    pred_all[(report["method"] == "prior").to_numpy()] = PRIOR_RANK
    sub = pd.DataFrame([(qid, lib.LEVELS[int(pr)]) for qid, pr in zip(qids, pred_all)],
                       columns=["question_id", "pred_rank"])
    sub.to_csv(args.out, index=False)
    report_path = args.report or args.out.with_name(args.out.stem + "_report.csv")
    report.assign(pred_rank=sub["pred_rank"]).to_csv(report_path, index=False)
    print("wrote", args.out, sub.pred_rank.value_counts().to_dict())
    print("methods:", report["method"].value_counts().to_dict(), "->", report_path)
    for note in notes:
        print("WARNING:", note)
    if args.labels is not None:
        truth = pd.read_csv(args.labels, dtype={"question_id": str}).set_index("question_id")
        pred_i = np.array([lib.LEVELS.index(r) for r in sub["pred_rank"]])
        true_i = np.array([lib.LEVELS.index(truth["rank"][q]) for q in sub["question_id"]])
        score = lib.competition_score(pred_i, true_i)
        print(f"rehearsal score {score:.4f} on {len(sub)} questions")
    if len(sub) != args.expect:
        raise SystemExit(f"expected {args.expect} questions, wrote {len(sub)} (check --tests-dir)")
    if args.strict and report["fallback"].any():
        raise SystemExit(f"--strict: {int(report['fallback'].sum())} questions used a fallback")


if __name__ == "__main__":
    main()
