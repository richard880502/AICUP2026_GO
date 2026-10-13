"""Move-sequence transformer trained on all ~1M games (no engine features).

tokens: (position, colour, move number, is-player's-colour) embeddings -> Transformer ->
attention pooling -> cumulative-link ordinal head. Training augments with the 8 board symmetries.

Example:
    c1/.venv/bin/python move_transformer.py --device cuda:0 --epochs 4
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

import rank_lib as lib
from train_eval import simulate_questions

N_CLASSES = 10
PAD = 0xFFFF


def symmetry_tables() -> np.ndarray:
    """[8, 362] position maps for the dihedral group of the 19x19 board (pass stays pass)."""
    grid = np.arange(361).reshape(19, 19)
    tables = []
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(grid, k)
            g = g[:, ::-1] if flip else g
            table = np.empty(362, np.int64)
            table[:361] = g.reshape(-1).argsort()  # inverse permutation maps old pos -> new pos
            table[361] = 361
            tables.append(table)
    return np.stack(tables)


class MoveTransformer(nn.Module):
    def __init__(self, d_model=192, heads=6, ffn=768, layers=4, dropout=0.1, length=256):
        super().__init__()
        self.pos = nn.Embedding(362, d_model)
        self.colour = nn.Embedding(2, d_model)
        self.own = nn.Embedding(2, d_model)
        self.step = nn.Embedding(length, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, 1)
        self.score = nn.Linear(d_model, 1)
        self.gaps = nn.Parameter(torch.zeros(N_CLASSES - 2))
        self.first = nn.Parameter(torch.tensor(-2.0))

    def thresholds(self):
        gaps = nn.functional.softplus(self.gaps + 0.5)
        return self.first + torch.cat([torch.zeros(1, device=gaps.device), gaps.cumsum(0)])

    def encode(self, pos, colour, valid, player_colour):
        """Returns the attention-pooled representation of each game."""
        steps = torch.arange(pos.shape[1], device=pos.device)
        h = self.pos(pos) + self.colour(colour) + self.own((colour == player_colour[:, None]).long()) + self.step(steps)[None]
        h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
        w = self.pool(h).squeeze(-1).float().masked_fill(~valid, float("-inf")).softmax(1)
        return (w.unsqueeze(-1) * h.float()).sum(1)

    def classify(self, game):
        """Returns ordinal class log-probabilities from game embeddings."""
        cum = torch.sigmoid(self.score(game) - self.thresholds())
        upper = torch.cat([torch.ones_like(cum[:, :1]), cum], 1)
        lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], 1)
        return torch.log((upper - lower).clamp_min(1e-7))

    def forward(self, pos, colour, valid, player_colour):
        return self.classify(self.encode(pos, colour, valid, player_colour))


def unpack(tokens, lengths, device):
    tokens = tokens.to(device).long()
    valid = torch.arange(tokens.shape[1], device=device)[None] < lengths.to(device)[:, None].long()
    pos = (tokens & 0x1FF).clamp(max=361)
    colour = (tokens >> 9) & 1
    return pos.masked_fill(~valid, 0), colour.masked_fill(~valid, 0), valid


def predict(model, tokens, lengths, player_colour, device, sym, tta=8, batch=1024):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(tokens), batch):
            pos, colour, valid = unpack(torch.from_numpy(tokens[i:i + batch].astype(np.int32)), torch.from_numpy(lengths[i:i + batch]), device)
            pc = torch.from_numpy(player_colour[i:i + batch]).to(device).long()
            prob = 0
            for s in range(tta):
                prob = prob + model(sym[s][pos], colour, valid, pc).float().exp()
            out.append((prob / tta).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=pathlib.Path, default=lib.ROOT / "features/moves/all_moves.npz")
    ap.add_argument("--val", type=pathlib.Path, default=lib.ROOT / "features/feat/val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--exclude-players", type=pathlib.Path, default=None, help="file of player ids never to train on")
    ap.add_argument("--max-games", type=int, default=0, help="debug: train on a subset")
    ap.add_argument("--questions-per-player", type=int, default=30)
    ap.add_argument("--tag", default="move")
    ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    z = np.load(args.moves)
    tokens, lengths, colour, rank, player, key = (z[k] for k in ("tokens", "length", "colour", "rank", "player_id", "game_key"))
    val = lib.load_features(args.val)
    val_players = set(val["player_id"])
    in_val_player = np.array([p in val_players for p in player])
    index_of = {k: i for i, k in enumerate(key)}
    val_rows = np.array([index_of[k] for k in val["game_key"]])
    assert set(player[val_rows]) == val_players and in_val_player.sum() >= len(val_rows)
    excluded = np.zeros(len(player), bool)
    if args.exclude_players is not None:
        banned = set(args.exclude_players.read_text().split())
        excluded = np.array([p in banned for p in player])
        print(f"excluding {int(excluded.sum())} games of {len(banned)} holdout players", flush=True)
    train_pool = np.flatnonzero(~in_val_player & ~excluded)  # every game of every validation/holdout player is excluded
    inner_players = set(rng.choice(np.unique(player[train_pool]), size=300, replace=False))
    inner = np.array([p in inner_players for p in player[train_pool]])
    tr, va = train_pool[~inner], train_pool[inner]
    if args.max_games:
        tr = rng.permutation(tr)[:args.max_games]
    print(f"train {len(tr)} games, inner-val {len(va)}, val cohort {len(val_rows)} (players disjoint)", flush=True)

    sym = torch.from_numpy(symmetry_tables()).to(device)
    tok_t = torch.from_numpy(tokens.astype(np.int32)).to(device)
    len_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    col_t = torch.from_numpy(colour.astype(np.int64)).to(device)
    y_t = torch.from_numpy(rank.astype(np.int64)).to(device)
    model = MoveTransformer().to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(tr) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    tr_t = torch.from_numpy(tr).to(device)
    best, best_state = 1e9, None
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), device=device)]
        for step in range(steps_per_epoch):
            idx = order[step * args.batch:(step + 1) * args.batch]
            pos, c, valid = unpack(tok_t[idx], len_t[idx], device)
            pos = sym[torch.randint(0, 8, (len(idx),), device=device)[:, None], pos]  # random board symmetry per game
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.nll_loss(model(pos, c, valid, col_t[idx]), y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 500 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps_per_epoch} loss {loss.item():.4f}  {time.time() - started:.0f}s", flush=True)
        prob = predict(model, tokens[va], lengths[va], colour[va], device, sym, tta=1)
        nll = float(-np.log(np.clip(prob[np.arange(len(va)), rank[va]], 1e-7, 1)).mean())
        print(f"epoch {epoch + 1}/{args.epochs} inner-val nll {nll:.4f} top1 {(prob.argmax(1) == rank[va]).mean():.4f}", flush=True)
        if nll < best:
            best, best_state = nll, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state": best_state}, args.out / f"{args.tag}.pt")

    prob = predict(model, tokens[val_rows], lengths[val_rows], colour[val_rows], device, sym)
    yv = lib.rank_ids(val["rank"])
    np.savez(args.out / f"val_probs_{args.tag}.npz", y=yv, player=val["player_id"], game_key=val["game_key"], **{args.tag: prob})
    report = {"per_game": {"top1": float((prob.argmax(1) == yv).mean()), "score": lib.competition_score(prob.argmax(1), yv)}, "question": {}}
    print(f"per-game top1 {report['per_game']['top1']:.4f} score {report['per_game']['score']:.4f}")
    for rule, rows in simulate_questions(prob, val["player_id"], yv, args.questions_per_player, 0).items():
        ks, sc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        report["question"][rule] = {"all": float(sc.mean()), "k5-9": float(sc[ks < 10].mean()), "k15-20": float(sc[ks >= 15].mean())}
        print(f"question {rule:16s} {sc.mean():.4f}  (k5-9 {sc[ks < 10].mean():.3f}, k15-20 {sc[ks >= 15].mean():.3f})")
    (args.out / f"val_report_{args.tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
