"""Transformer encoder over per-decision KataGo/HumanSL tokens with an ordinal head.

    decisions x F -> Linear(d) -> + position embedding -> Transformer x L -> attention pooling
    -> game vector (d) -> cumulative-link ordinal head -> 10 rank probabilities

Example:
    c1/.venv/bin/python transformer_rank.py --train features/feat96/train \
        --val features/feat96/val --device cuda:1
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from torch import nn

import rank_lib as lib
from train_eval import simulate_questions

N_CLASSES = len(lib.LEVELS)
# 1 for the true rank, e^-1 for its neighbours (the competition's scoring rule)
SCORE_MATRIX = np.eye(N_CLASSES) + lib.ADJACENT * (np.eye(N_CLASSES, k=1) + np.eye(N_CLASSES, k=-1))


def decision_tokens(tokens: np.ndarray) -> np.ndarray:
    """[G, S, 11, 3] -> [G, S, F]: the 33 raw channels plus 10 profile-centred surprisals."""
    games, samples = tokens.shape[:2]
    human = tokens[:, :, 1:, 0]
    centred = human - human.mean(2, keepdims=True)
    flat = tokens.reshape(games, samples, -1)
    return np.concatenate([flat, centred], axis=2).astype(np.float32)


class RankTransformer(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Decision-sequence encoder with a cumulative-link (ordinal) head.

    The attribute names are the keys of every saved checkpoint, hence kept as they are.
    """

    def __init__(self, in_dim: int, n_tokens: int, d_model: int = 128,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 heads: int = 4, ffn: int = 384, layers: int = 3, dropout: float = 0.1,
                 head: str = "clm"):
        super().__init__()
        self.head = head
        self.cls = nn.Linear(d_model, N_CLASSES) if head == "softmax" else None
        self.project = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.position = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(d_model, heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, 1)
        self.score = nn.Linear(d_model, 1)
        self.gaps = nn.Parameter(torch.zeros(N_CLASSES - 2))  # softplus gaps between thresholds
        self.first_threshold = nn.Parameter(torch.tensor(-2.0))

    def thresholds(self) -> torch.Tensor:
        """The 9 increasing cut points of the ordinal head."""
        gaps = nn.functional.softplus(self.gaps + 0.5)  # pylint: disable=not-callable
        steps = torch.cat([torch.zeros(1, device=gaps.device), gaps.cumsum(0)])
        return self.first_threshold + steps

    def encode(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Returns the attention-pooled representation of each game."""
        h = self.project(x) + self.position[:, :x.shape[1]]
        h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
        weights = self.pool(h).squeeze(-1).masked_fill(~valid, float("-inf")).softmax(1)
        return (weights.unsqueeze(-1) * h).sum(1)

    def classify(self, game: torch.Tensor) -> torch.Tensor:
        """Returns cumulative-link class log-probabilities for embeddings."""
        cumulative = torch.sigmoid(self.score(game) - self.thresholds())  # P(y > k), k = 0..8
        ones, zeros = torch.ones_like(cumulative[:, :1]), torch.zeros_like(cumulative[:, :1])
        upper = torch.cat([ones, cumulative], 1)
        lower = torch.cat([cumulative, zeros], 1)
        return torch.log((upper - lower).clamp_min(1e-7))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Returns log class probabilities [B, 10] from the ordinal head."""
        return self.classify(self.encode(x, valid))


def add_value_features(x: np.ndarray, valid: np.ndarray, games: dict, value_dir: pathlib.Path,
                       _fraction: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates value/ownership tokens (extract_value_features.py) after checking exact
    alignment (the last argument is unused and kept for call compatibility)."""
    value = lib.load_features(value_dir)
    lookup = {k: i for i, k in enumerate(value["game_key"])}
    rows = np.array([lookup[k] for k in games["game_key"]])  # same games, any order/subset
    tokens, masks = value["tokens"][rows], value["masks"][rows]
    assert (masks == valid).mean() > 0.999, "value and policy decision masks disagree"
    return np.concatenate([x, tokens], axis=2), valid & masks


def add_game_features(x: np.ndarray, games: dict) -> np.ndarray:
    """Appends per-game constants to every decision token: log total moves, own pass count,
    whether the game ends in passes."""
    data = np.load(lib.ROOT / "features/moves/all_moves512.npz")
    idx = {k: i for i, k in enumerate(data["game_key"])}
    rows = np.array([idx[k] for k in games["game_key"]])
    tok = data["tokens"][rows].astype(np.int64)
    length = data["length"][rows].astype(np.int64)
    colour = data["colour"][rows].astype(np.int64)
    valid = np.arange(tok.shape[1])[None] < length[:, None]
    is_pass = ((tok & 0x1FF) == 361) & valid
    mover = (tok >> 9) & 1
    own_pass = (is_pass & (mover == colour[:, None])).sum(1)
    last = np.take_along_axis(is_pass, np.maximum(length - 1, 0)[:, None], 1)[:, 0]
    feats = np.stack([np.log(np.maximum(length, 1)), own_pass, last.astype(np.float64)], 1)
    feats = feats.astype(np.float32)
    shape = (x.shape[0], x.shape[1], feats.shape[1])
    return np.concatenate([x, np.broadcast_to(feats[:, None, :], shape)], 2)


def standardise(train: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of the valid decision tokens."""
    valid = train[mask]
    return valid.mean(0), valid.std(0) + 1e-6


def predict(model: nn.Module, x: np.ndarray, valid: np.ndarray, device: torch.device,
            batch: int = 1024) -> np.ndarray:
    """Class probabilities [G, 10]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            vb = torch.from_numpy(valid[i:i + batch]).to(device)
            out.append(model(xb, vb).exp().cpu().numpy())
    return np.concatenate(out)


def batch_loss(logp: torch.Tensor, yb: torch.Tensor, loss_name: str) -> torch.Tensor:
    """nll, 'soft' (targets shaped like the scoring rule: 1 for the true rank, e^-1 for its
    neighbours) or 'escore' (expected competition score plus a small NLL term for stability)."""
    score_w = torch.tensor(SCORE_MATRIX, dtype=torch.float32, device=logp.device)
    if loss_name == "soft":
        target = score_w[yb] / score_w[yb].sum(1, keepdim=True)
        return -(target * logp).sum(1).mean()
    if loss_name == "escore":
        expected = (logp.exp() * score_w[yb]).sum(1).mean()
        return -expected + 0.1 * nn.functional.nll_loss(logp, yb)
    return nn.functional.nll_loss(logp, yb)


def build_model(n_in: int, n_tokens: int, args: argparse.Namespace, device: torch.device,
                init_state: dict | None) -> nn.Module:
    """A fresh (or initialised) model; args.train_only == 'head' freezes all but the head."""
    model = RankTransformer(n_in, n_tokens, args.d_model, args.heads, args.ffn, args.layers,
                            args.dropout, args.head).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    if getattr(args, "train_only", "all") == "head":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("score.", "gaps", "first_threshold"))
    return model


def inner_metric(model: nn.Module, data: tuple[np.ndarray, np.ndarray, np.ndarray],
                 select: str, device: torch.device) -> float:
    """Early-stopping metric on the inner player-disjoint slice (lower is better)."""
    x, valid, y = data
    prob = predict(model, x, valid, device)
    if select == "score":  # negative expected-decode competition score
        return -lib.competition_score(lib.decode(prob), y)
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-7, 1)).mean())


def inner_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Training rows and a player-disjoint 10% inner slice for early stopping."""
    rng = np.random.default_rng(seed)
    players = np.unique(groups)
    held = set(rng.choice(players, size=max(1, int(0.1 * len(players))), replace=False))
    inner = np.array([g in held for g in groups])
    return np.flatnonzero(~inner), np.flatnonzero(inner)


def train_epoch(model: nn.Module, opt: tuple, tensors: tuple, args: argparse.Namespace) -> None:
    """One pass over the training rows with token dropout and input noise.
    opt = (optimiser, one-cycle schedule); tensors = (x, valid, y) on the device."""
    xt, vt, yt = tensors
    model.train()
    order = torch.randperm(len(xt), device=xt.device)
    for i in range(0, len(xt), args.batch):
        idx = order[i:i + args.batch]
        xb, vb, yb = xt[idx], vt[idx].clone(), yt[idx]
        vb = vb & (torch.rand_like(vb, dtype=torch.float32) > args.token_dropout)  # token dropout
        vb[:, 0] |= ~vb.any(1)  # never mask a whole game
        xb = xb + args.noise * torch.randn_like(xb)
        loss = batch_loss(model(xb, vb), yb, args.loss)
        opt[0].zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt[0].step()
        opt[1].step()


def make_optimiser(model: nn.Module, args: argparse.Namespace, n_rows: int) -> tuple:
    """AdamW over the trainable parameters with a one-cycle schedule."""
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (-(-n_rows // args.batch))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr, total_steps=steps,
                                                   pct_start=0.1)
    return optimiser, schedule


def train_one(data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              args: argparse.Namespace, seed: int, device: torch.device,
              init_state: dict | None = None) -> tuple[nn.Module, dict]:
    """Trains one model; early-stops on a player-disjoint slice of the training data.

    data = (x, valid, y, groups). init_state: optional state dict to start from (fine-tuning);
    args.train_only == "head" then trains only the ordinal head.
    """
    torch.manual_seed(seed)
    tr_rows, va_rows = inner_split(data[3], seed)
    model = build_model(data[0].shape[2], data[0].shape[1], args, device, init_state)
    opt = make_optimiser(model, args, len(tr_rows))
    tensors = tuple(torch.from_numpy(a[tr_rows]).to(device) for a in data[:3])
    inner_data = (data[0][va_rows], data[1][va_rows], data[2][va_rows])
    best_state, history = {}, [1e9]
    for epoch in range(args.epochs):
        train_epoch(model, opt, tensors, args)
        metric = inner_metric(model, inner_data, args.select, device)
        if metric < min(history):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(metric)
        if (epoch + 1) % 5 == 0:
            print(f"  seed {seed} epoch {epoch + 1}/{args.epochs} inner-val nll {metric:.4f} "
                  f"(best {min(history):.4f})", flush=True)
    model.load_state_dict(best_state)
    return model, {"best_inner_nll": min(history), "best_epoch": int(np.argmin(history[1:])) + 1}


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--val", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token-dropout", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--questions-per-player", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "results")
    ap.add_argument("--tag", default="transformer")
    ap.add_argument("--value-train", type=pathlib.Path, default=None,
                    help="value-feature shards aligned with --train")
    ap.add_argument("--value-val", type=pathlib.Path, default=None,
                    help="value-feature shards aligned with --val")
    ap.add_argument("--exclude-players", type=pathlib.Path, default=None,
                    help="file of player ids never to train on (holdout test)")
    ap.add_argument("--head", choices=("clm", "softmax"), default="clm")
    ap.add_argument("--loss", choices=("nll", "soft", "escore"), default="nll")
    ap.add_argument("--select", choices=("nll", "score"), default="nll")
    ap.add_argument("--game-feats", action="store_true",
                    help="append game-level features (log length, own passes, ends-with-pass)")
    ap.add_argument("--keep-profiles", default=None,
                    help="comma list of model indices to keep (0 = normal, 1..10 = HumanSL)")
    ap.add_argument("--token-stride", type=int, default=1,
                    help="use every k-th decision (sensitivity to decisions per game)")
    ap.add_argument("--init-from", default=None,
                    help="start seed s from results/<TAG>_seed<s>.pt, reuse its standardisation")
    ap.add_argument("--train-only", choices=("all", "head"), default="all",
                    help="head = fine-tune only the ordinal head")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="learning-curve: keep this share of training players")
    return ap.parse_args()


def load_splits(args: argparse.Namespace) -> tuple[dict, dict]:
    """Training / validation features after the --train-fraction, --keep-profiles and
    --exclude-players filters."""
    train, val = lib.load_features(args.train), lib.load_features(args.val)
    assert not set(train["player_id"]) & set(val["player_id"]), "train/val players overlap"
    if args.train_fraction < 1.0:
        rng = np.random.default_rng(123)
        players = np.unique(train["player_id"])
        kept = set(rng.choice(players, size=int(args.train_fraction * len(players)),
                              replace=False))
        rows = np.array([p in kept for p in train["player_id"]])
        train = {k: v[rows] for k, v in train.items()}
    if args.keep_profiles:
        profiles = [int(i) for i in args.keep_profiles.split(",")]
        train["tokens"] = train["tokens"][:, :, profiles]
        val["tokens"] = val["tokens"][:, :, profiles]
        print("keeping profile indices", profiles, flush=True)
    if args.exclude_players is not None:
        banned = set(args.exclude_players.read_text().split())
        rows = np.array([p not in banned for p in train["player_id"]])
        print(f"excluding {int((~rows).sum())} training games of {len(banned)} holdout players",
              flush=True)
        train = {k: v[rows] for k, v in train.items()}
    return train, val


def model_inputs(args: argparse.Namespace, train: dict,
                 val: dict) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Token tensors (x, valid) for training and validation, before standardisation."""
    xt, vt = decision_tokens(train["tokens"]), train["masks"]
    xv, vv = decision_tokens(val["tokens"]), val["masks"]
    if args.value_train is not None:
        xt, vt = add_value_features(xt, vt, train, args.value_train, args.train_fraction)
        xv, vv = add_value_features(xv, vv, val, args.value_val, 1.0)
    if args.game_feats:
        xt, xv = add_game_features(xt, train), add_game_features(xv, val)
    if args.token_stride > 1:
        stride = args.token_stride
        xt, vt, xv, vv = xt[:, ::stride], vt[:, ::stride], xv[:, ::stride], vv[:, ::stride]
    return (xt, vt), (xv, vv)


def write_report(args: argparse.Namespace, prob: np.ndarray, val: dict, info: list) -> None:
    """Per-game and simulated-question validation scores -> val_report_<tag>.json."""
    yv = lib.rank_ids(val["rank"])
    report: dict = {"per_game": {"top1": float((prob.argmax(1) == yv).mean()),
                                 "score": lib.competition_score(prob.argmax(1), yv)},
                    "seeds": info, "question": {}}
    print(f"per-game ensemble top1 {report['per_game']['top1']:.4f} "
          f"score {report['per_game']['score']:.4f}")
    questions = simulate_questions(prob, val["player_id"], yv, args.questions_per_player, 0)
    for rule, scored in questions.items():
        ks, sc = np.array([r[0] for r in scored]), np.array([r[1] for r in scored])
        report["question"][rule] = {"all": float(sc.mean()), "k5-9": float(sc[ks < 10].mean()),
                                    "k15-20": float(sc[ks >= 15].mean())}
        print(f"question {rule:16s} {sc.mean():.4f}  (k5-9 {sc[ks < 10].mean():.3f}, "
              f"k15-20 {sc[ks >= 15].mean():.3f})")
    (args.out / f"val_report_{args.tag}.json").write_text(json.dumps(report, indent=2))


def standardised(args: argparse.Namespace, train_xv: tuple[np.ndarray, np.ndarray],
                 val_xv: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Standardised, zero-padded token arrays; the statistics come from --init-from when given.
    Returns (x_train, x_val, {"mean", "std", "inits"})."""
    (xt, vt), (xv, vv) = train_xv, val_xv
    inits = [torch.load(lib.ROOT / f"results/{args.init_from}_seed{s}.pt", map_location="cpu",
                        weights_only=False)
             for s in range(args.seeds)] if args.init_from else None
    mean, std = (inits[0]["mean"], inits[0]["std"]) if inits else standardise(xt, vt)
    if inits:
        assert all(np.allclose(i["mean"], mean) for i in inits), \
            "init checkpoints use different standardisations"
    xt, xv = (xt - mean) / std, (xv - mean) / std
    xt = np.where(vt[..., None], xt, 0).astype(np.float32)
    xv = np.where(vv[..., None], xv, 0).astype(np.float32)
    return xt, xv, {"mean": mean, "std": std, "inits": inits}


def fit_seeds(args: argparse.Namespace, train_set: tuple, val_set: tuple, norm: dict,
              device: torch.device) -> tuple[list[np.ndarray], list[dict]]:
    """Trains and saves one model per seed. train_set = (x, valid, y, groups) standardised,
    val_set = (x, valid, y). Returns per-seed validation probabilities and training info."""
    probs, info = [], []
    for seed in range(args.seeds):
        init = norm["inits"][seed]["state"] if norm["inits"] else None
        model, meta = train_one(train_set, args, seed, device, init)
        probs.append(predict(model, val_set[0], val_set[1], device))
        info.append(meta)
        torch.save({"state": model.state_dict(), "mean": norm["mean"], "std": norm["std"],
                    "args": vars(args) | {"out": str(args.out)}},
                   args.out / f"{args.tag}_seed{seed}.pt")
        print(f"seed {seed}: {meta}  single-model per-game top1 "
              f"{(probs[-1].argmax(1) == val_set[2]).mean():.4f}", flush=True)
    return probs, info


def main() -> None:
    """Trains args.seeds models and writes checkpoints, validation probabilities and a report."""
    args = parse_args()
    device = torch.device(args.device)
    train, val = load_splits(args)
    (xt, vt), (xv, vv) = model_inputs(args, train, val)
    yt, yv = lib.rank_ids(train["rank"]), lib.rank_ids(val["rank"])
    xt, xv, norm = standardised(args, (xt, vt), (xv, vv))
    print(f"train {xt.shape}  val {xv.shape}  device {device}", flush=True)
    probs, info = fit_seeds(args, (xt, vt, yt, train["player_id"]), (xv, vv, yv), norm, device)
    prob = np.mean(probs, axis=0)
    np.savez(args.out / f"val_probs_{args.tag}.npz", y=yv, player=val["player_id"],
             game_key=val["game_key"], transformer=prob)
    write_report(args, prob, val, info)


if __name__ == "__main__":
    main()
