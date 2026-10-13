"""Official-baseline reproduction: GoRankResNet on the last 250 board states (500 planes).

Uses the tutorial's network unchanged (read from ../aicup_tutorial_review/network.py), cross-entropy,
AdamW lr 1e-3, batch 256. Board states are simulated on the fly (board_sim.py). Training games exclude
every player in the validation cohort, so the score is comparable with the other models' proxy scores.

Example:
    c1/.venv/bin/python resnet_rank.py --device cuda:1 --epochs 2
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import board_sim
import rank_lib as lib
from train_eval import simulate_questions

sys.path.insert(0, "/tmp2/richard/aicup_tutorial_review")
from network import GoRankResNet  # noqa: E402  the tutorial's architecture, unmodified


class RankNet(GoRankResNet):
    """The tutorial network; forward is identical except reshape replaces view (channels_last safe)."""

    def forward(self, x):
        out = nn.functional.relu(self.bn_in(self.conv_in(x)))
        out = self.layers(out)
        out = nn.functional.relu(self.bn_in2(self.conv_in2(out)))
        out = out.reshape(out.size(0), -1)
        out = self.dropout(nn.functional.relu(self.fc(out)))
        return self.fc_2(out)


class Games(Dataset):
    def __init__(self, tokens, lengths, labels, rows, colours):
        self.tokens, self.lengths, self.labels, self.rows, self.colours = tokens, lengths, labels, rows, colours

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (torch.from_numpy(board_sim.history_planes(self.tokens[r], int(self.lengths[r]))), int(self.labels[r]),
                int(self.colours[r]))


def to_planes(states: torch.Tensor, colour: torch.Tensor | None = None) -> torch.Tensor:
    """int8 [B, 250, 19, 19] -> float [B, 500(+1), 19, 19]; black/white planes interleaved per state.

    With `colour` (0 = the labelled player is black, 1 = white) one constant plane is appended.
    """
    black, white = (states == 1), (states == 2)
    planes = torch.stack([black, white], 2).flatten(1, 2).float()
    if colour is not None:
        planes = torch.cat([planes, colour.float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)], 1)
    return planes


def symmetry(states: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """One of the 8 board symmetries applied to a whole batch of states."""
    states = torch.rot90(states, k, dims=(2, 3))
    return states.flip(3) if flip else states


def predict(model, loader, device, use_colour=False, tta=1):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for states, _, colour in loader:
            states = states.to(device, non_blocking=True)
            colour = colour.to(device) if use_colour else None
            prob = 0
            for t in range(tta):  # tta = 8 averages the dihedral group
                x = to_planes(symmetry(states, t % 4, t >= 4), colour).contiguous(memory_format=torch.channels_last)
                prob = prob + model(x).float().softmax(1)
            out.append((prob / tta).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=pathlib.Path, default=lib.ROOT / "features/moves/all_moves512.npz")
    ap.add_argument("--val", type=pathlib.Path, default=lib.ROOT / "features/feat/val")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exclude-players", type=pathlib.Path, default=None, help="file of player ids never to train on")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--cosine", action="store_true", help="one-cycle cosine learning-rate schedule instead of a constant rate")
    ap.add_argument("--colour-plane", action="store_true", help="tell the network which side is the labelled player")
    ap.add_argument("--symmetry-aug", action="store_true", help="random board symmetry in training, 8-fold TTA at the end")
    ap.add_argument("--random-holdout", type=int, default=20000,
                    help="games held out at random (same players still in training) = the tutorial's own validation protocol")
    ap.add_argument("--questions-per-player", type=int, default=30)
    ap.add_argument("--tag", default="resnet")
    ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = True

    z = np.load(args.moves)
    tokens, lengths, rank, player, key, colour = (z[k] for k in ("tokens", "length", "rank", "player_id", "game_key", "colour"))
    val = lib.load_features(args.val)
    val_players = set(val["player_id"])
    in_val = np.array([p in val_players for p in player])
    index_of = {k: i for i, k in enumerate(key)}
    val_rows = np.array([index_of[k] for k in val["game_key"]])
    excluded = np.zeros(len(player), bool)
    if args.exclude_players is not None:
        banned = set(args.exclude_players.read_text().split())
        excluded = np.array([p in banned for p in player])
        print(f"excluding {int(excluded.sum())} games of {len(banned)} holdout players", flush=True)
    pool = np.flatnonzero(~in_val & ~excluded)
    inner_players = set(rng.choice(np.unique(player[pool]), size=100, replace=False))
    inner = np.array([p in inner_players for p in player[pool]])
    tr, va = pool[~inner], rng.permutation(pool[inner])[:6000]  # random draw: rank-balanced (the file is sorted by rank)
    tr = rng.permutation(tr)
    rh, tr = tr[:args.random_holdout], tr[args.random_holdout:]
    if args.max_games:
        tr = tr[:args.max_games]
    print(f"train {len(tr)} games, random-holdout {len(rh)} (training players), inner-val {len(va)} (new players), "
          f"val cohort {len(val_rows)}", flush=True)

    def loader(rows, shuffle):
        return DataLoader(Games(tokens, lengths, rank, rows, colour), batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=shuffle, persistent_workers=shuffle, prefetch_factor=4)

    model = RankNet(in_channels=501 if args.colour_plane else 500, num_classes=10).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = loader(tr, True)
    schedule = (torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.05)
                if args.cosine else None)
    best, best_state, started = -1e9, None, time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (states, y, c) in enumerate(train_loader):
            states = states.to(device, non_blocking=True)
            if args.symmetry_aug:
                states = symmetry(states, int(torch.randint(0, 4, (1,))), bool(torch.rand(1) < 0.5))
            x = to_planes(states, c.to(device) if args.colour_plane else None).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if schedule is not None:
                schedule.step()
            if step % 200 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}  {time.time() - started:.0f}s", flush=True)
        def nll_acc(rows):
            pr = predict(model, loader(rows, False), device, args.colour_plane)
            return float(-np.log(np.clip(pr[np.arange(len(rows)), rank[rows]], 1e-7, 1)).mean()), float((pr.argmax(1) == rank[rows]).mean())

        nll_new, acc_new = nll_acc(va)
        nll_rh, acc_rh = nll_acc(rh)
        pc = predict(model, loader(val_rows, False), device, args.colour_plane)
        acc_cohort = float((pc.argmax(1) == lib.rank_ids(val["rank"])).mean())
        print(f"epoch {epoch + 1}/{args.epochs} | random-holdout (seen players) top1 {acc_rh:.4f} nll {nll_rh:.4f} | "
              f"new-player inner-val top1 {acc_new:.4f} nll {nll_new:.4f} | val cohort top1 {acc_cohort:.4f}", flush=True)
        torch.save({"state": {k: v.detach().clone() for k, v in model.state_dict().items()}}, args.out / f"{args.tag}_epoch{epoch + 1}.pt")
        if -nll_new > best:  # keep the epoch with the lowest new-player NLL
            best, best_state = -nll_new, {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state": best_state}, args.out / f"{args.tag}.pt")
    model.load_state_dict(best_state)

    prob = predict(model, loader(val_rows, False), device, args.colour_plane, tta=8 if args.symmetry_aug else 1)
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
