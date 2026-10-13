"""Transformer encoder over per-decision KataGo/HumanSL tokens with an ordinal head.

    decisions x F -> Linear(d) -> + position embedding -> Transformer x L -> attention pooling
    -> game vector (d) -> cumulative-link ordinal head -> 10 rank probabilities

Example:
    c1/.venv/bin/python transformer_rank.py --train features/feat/train --val features/feat/val --device cuda:1
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


def decision_tokens(tokens: np.ndarray) -> np.ndarray:
    """[G, S, 11, 3] -> [G, S, F]: the 33 raw channels plus 10 profile-centred surprisals."""
    games, samples = tokens.shape[:2]
    human = tokens[:, :, 1:, 0]
    centred = human - human.mean(2, keepdims=True)
    return np.concatenate([tokens.reshape(games, samples, -1), centred], axis=2).astype(np.float32)


class RankTransformer(nn.Module):
    def __init__(self, in_dim: int, n_tokens: int, d_model: int = 128, heads: int = 4, ffn: int = 384,
                 layers: int = 3, dropout: float = 0.1, head: str = "clm"):
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
        gaps = nn.functional.softplus(self.gaps + 0.5)
        return self.first_threshold + torch.cat([torch.zeros(1, device=gaps.device), gaps.cumsum(0)])

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


def add_value_features(x, valid, games, value_dir, fraction):
    """Concatenates value/ownership tokens (extract_value_features.py) after checking exact alignment."""
    value = lib.load_features(value_dir)
    lookup = {k: i for i, k in enumerate(value["game_key"])}
    rows = np.array([lookup[k] for k in games["game_key"]])  # same games, possibly different order/subset
    tokens, masks = value["tokens"][rows], value["masks"][rows]
    assert (masks == valid).mean() > 0.999, "value and policy decision masks disagree"
    return np.concatenate([x, tokens], axis=2), valid & masks


def add_game_features(x, games):
    """Appends per-game constants to every decision token: log total moves, own pass count, whether the game ends in passes."""
    z = np.load(lib.ROOT / "features/moves/all_moves512.npz")
    idx = {k: i for i, k in enumerate(z["game_key"])}
    rows = np.array([idx[k] for k in games["game_key"]])
    tok, length, colour = z["tokens"][rows].astype(np.int64), z["length"][rows].astype(np.int64), z["colour"][rows].astype(np.int64)
    valid = np.arange(tok.shape[1])[None] < length[:, None]
    is_pass = ((tok & 0x1FF) == 361) & valid
    mover = (tok >> 9) & 1
    own_pass = (is_pass & (mover == colour[:, None])).sum(1)
    last = np.take_along_axis(is_pass, np.maximum(length - 1, 0)[:, None], 1)[:, 0]
    feats = np.stack([np.log(np.maximum(length, 1)), own_pass, last.astype(np.float64)], 1).astype(np.float32)
    return np.concatenate([x, np.broadcast_to(feats[:, None, :], (x.shape[0], x.shape[1], feats.shape[1]))], 2)


def standardise(train: np.ndarray, mask: np.ndarray):
    valid = train[mask]
    return valid.mean(0), valid.std(0) + 1e-6


def predict(model, x, valid, device, batch=1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            vb = torch.from_numpy(valid[i:i + batch]).to(device)
            out.append(model(xb, vb).exp().cpu().numpy())
    return np.concatenate(out)


def train_one(x, valid, y, groups, args, seed, device):
    """Trains one model; early-stops on a player-disjoint slice of the training data."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    players = np.unique(groups)
    held = set(rng.choice(players, size=max(1, int(0.1 * len(players))), replace=False))
    inner = np.array([g in held for g in groups])
    tr, va = np.flatnonzero(~inner), np.flatnonzero(inner)
    model = RankTransformer(x.shape[2], x.shape[1], args.d_model, args.heads, args.ffn, args.layers, args.dropout,
                            args.head).to(device)
    score_w = torch.tensor(np.eye(N_CLASSES) + lib.ADJACENT * (np.eye(N_CLASSES, k=1) + np.eye(N_CLASSES, k=-1)),
                           dtype=torch.float32, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * (-(-len(tr) // args.batch))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr, total_steps=steps, pct_start=0.1)
    xt, vt, yt = (torch.from_numpy(a).to(device) for a in (x[tr], valid[tr], y[tr]))
    best, best_state, history = 1e9, None, []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), args.batch):
            idx = order[i:i + args.batch]
            xb, vb, yb = xt[idx], vt[idx].clone(), yt[idx]
            keep = torch.rand_like(vb, dtype=torch.float32) > args.token_dropout  # random token dropout
            vb = vb & keep
            vb[:, 0] |= ~vb.any(1)  # never mask a whole game
            xb = xb + args.noise * torch.randn_like(xb)
            logp = model(xb, vb)
            if args.loss == "soft":  # targets shaped like the scoring rule: 1 for the true rank, e^-1 for its neighbours
                target = score_w[yb] / score_w[yb].sum(1, keepdim=True)
                loss = -(target * logp).sum(1).mean()
            elif args.loss == "escore":  # maximise the expected competition score, small NLL term for stability
                loss = -(logp.exp() * score_w[yb]).sum(1).mean() + 0.1 * nn.functional.nll_loss(logp, yb)
            else:
                loss = nn.functional.nll_loss(logp, yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
        prob = predict(model, x[va], valid[va], device)
        nll = float(-np.log(np.clip(prob[np.arange(len(va)), y[va]], 1e-7, 1)).mean())
        if args.select == "score":  # lower is better: negative expected-decode competition score
            nll = -lib.competition_score(lib.decode(prob), y[va])
        history.append(nll)
        if nll < best:
            best, best_state = nll, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  seed {seed} epoch {epoch + 1}/{args.epochs} inner-val nll {nll:.4f} (best {best:.4f})", flush=True)
    model.load_state_dict(best_state)
    return model, {"best_inner_nll": best, "best_epoch": int(np.argmin(history)) + 1}


def main() -> None:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--value-train", type=pathlib.Path, default=None, help="value-feature shards aligned with --train")
    ap.add_argument("--value-val", type=pathlib.Path, default=None, help="value-feature shards aligned with --val")
    ap.add_argument("--exclude-players", type=pathlib.Path, default=None, help="file of player ids never to train on (holdout test)")
    ap.add_argument("--head", choices=("clm", "softmax"), default="clm")
    ap.add_argument("--loss", choices=("nll", "soft", "escore"), default="nll")
    ap.add_argument("--select", choices=("nll", "score"), default="nll")
    ap.add_argument("--game-feats", action="store_true", help="append game-level features (log length, own passes, ends-with-pass) to every token")
    ap.add_argument("--keep-profiles", default=None, help="comma list of model indices to keep (0 = normal, 1..10 = HumanSL 11k..6d)")
    ap.add_argument("--token-stride", type=int, default=1, help="use every k-th decision (tests sensitivity to decisions per game)")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="learning-curve: keep this share of training players")
    args = ap.parse_args()
    device = torch.device(args.device)

    train, val = lib.load_features(args.train), lib.load_features(args.val)
    assert not set(train["player_id"]) & set(val["player_id"]), "train/val players overlap"
    if args.train_fraction < 1.0:
        rng = np.random.default_rng(123)
        players = np.unique(train["player_id"])
        keep = set(rng.choice(players, size=int(args.train_fraction * len(players)), replace=False))
        rows = np.array([p in keep for p in train["player_id"]])
        train = {k: v[rows] for k, v in train.items()}
    if args.keep_profiles:
        keep = [int(i) for i in args.keep_profiles.split(",")]
        train["tokens"], val["tokens"] = train["tokens"][:, :, keep], val["tokens"][:, :, keep]
        print("keeping profile indices", keep, flush=True)
    if args.exclude_players is not None:
        banned = set(args.exclude_players.read_text().split())
        rows = np.array([p not in banned for p in train["player_id"]])
        print(f"excluding {int((~rows).sum())} training games of {len(banned)} holdout players", flush=True)
        train = {k: v[rows] for k, v in train.items()}
    xt, vt, yt = decision_tokens(train["tokens"]), train["masks"], lib.rank_ids(train["rank"])
    xv, vv, yv = decision_tokens(val["tokens"]), val["masks"], lib.rank_ids(val["rank"])
    if args.value_train is not None:
        xt, vt = add_value_features(xt, vt, train, args.value_train, args.train_fraction)
        xv, vv = add_value_features(xv, vv, val, args.value_val, 1.0)
    if args.game_feats:
        xt, xv = add_game_features(xt, train), add_game_features(xv, val)
    if args.token_stride > 1:
        xt, vt, xv, vv = xt[:, ::args.token_stride], vt[:, ::args.token_stride], xv[:, ::args.token_stride], vv[:, ::args.token_stride]
    mean, std = standardise(xt, vt)
    xt, xv = (xt - mean) / std, (xv - mean) / std
    xt, xv = np.where(vt[..., None], xt, 0).astype(np.float32), np.where(vv[..., None], xv, 0).astype(np.float32)
    print(f"train {xt.shape}  val {xv.shape}  device {device}", flush=True)

    probs, info = [], []
    for seed in range(args.seeds):
        model, meta = train_one(xt, vt, yt, train["player_id"], args, seed, device)
        probs.append(predict(model, xv, vv, device))
        info.append(meta)
        torch.save({"state": model.state_dict(), "mean": mean, "std": std, "args": vars(args) | {"out": str(args.out)}},
                   args.out / f"{args.tag}_seed{seed}.pt")
        print(f"seed {seed}: {meta}  single-model per-game top1 {(probs[-1].argmax(1) == yv).mean():.4f}", flush=True)
    prob = np.mean(probs, axis=0)
    np.savez(args.out / f"val_probs_{args.tag}.npz", y=yv, player=val["player_id"], game_key=val["game_key"], transformer=prob)

    report = {"per_game": {"top1": float((prob.argmax(1) == yv).mean()), "score": lib.competition_score(prob.argmax(1), yv)},
              "seeds": info, "question": {}}
    print(f"per-game ensemble top1 {report['per_game']['top1']:.4f} score {report['per_game']['score']:.4f}")
    for rule, rows in simulate_questions(prob, val["player_id"], yv, args.questions_per_player, 0).items():
        ks, sc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        report["question"][rule] = {"all": float(sc.mean()), "k5-9": float(sc[ks < 10].mean()), "k15-20": float(sc[ks >= 15].mean())}
        print(f"question {rule:16s} {sc.mean():.4f}  (k5-9 {sc[ks < 10].mean():.3f}, k15-20 {sc[ks >= 15].mean():.3f})")
    (args.out / f"val_report_{args.tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
