"""Parses every training game into a fixed-length move-token array (no engine needed).

token = position (y*19+x, pass=361) | colour << 9 ; 0xFFFF = padding.
Output features/moves/all_moves.npz: tokens [N, L] uint16, length, colour (player's), rank id,
player_id, game_key. Example:
    c1/.venv/bin/python scripts/prep_moves.py --workers 8
"""
import argparse, pathlib, re
from multiprocessing import Pool
import numpy as np, pandas as pd

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERN = re.compile(r";([BW])\[([a-s]{0,2})\]")
PAD = 0xFFFF

def parse_chunk(args):
    sgfs, length = args
    out = np.full((len(sgfs), length), PAD, np.uint16)
    lens = np.zeros(len(sgfs), np.int16)
    for i, sgf in enumerate(sgfs):
        moves = PATTERN.findall(sgf)[:length]
        lens[i] = len(moves)
        for j, (c, xy) in enumerate(moves):
            pos = 361 if len(xy) < 2 else (ord(xy[1]) - 97) * 19 + (ord(xy[0]) - 97)
            out[i, j] = pos | ((c == "W") << 9)
    return out, lens

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=256)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/moves/all_moves.npz")
a = ap.parse_args()
a.out.parent.mkdir(parents=True, exist_ok=True)
parts = {k: [] for k in ("tokens", "length", "colour", "rank", "player_id", "game_key")}
with Pool(a.workers) as pool:
    for rank in LEVELS:
        d = pd.read_csv(ROOT / f"data/training/train_{rank}.csv")
        sgfs = d["sgf_content"].tolist()
        chunks = [(sgfs[i:i + 2000], a.length) for i in range(0, len(sgfs), 2000)]
        res = pool.map(parse_chunk, chunks)
        parts["tokens"].append(np.concatenate([r[0] for r in res]))
        parts["length"].append(np.concatenate([r[1] for r in res]))
        parts["colour"].append((d["color"] == "W").to_numpy(np.int8))
        parts["rank"].append(np.full(len(d), LEVELS.index(rank), np.int8))
        parts["player_id"].append(d["player_id"].to_numpy(dtype="U16"))
        parts["game_key"].append((rank + "_" + d["game_id"]).to_numpy(dtype="U12"))
        print(rank, len(d), "games; mean moves %.0f" % parts["length"][-1].mean(), flush=True)
np.savez(a.out, **{k: np.concatenate(v) for k, v in parts.items()})
print("saved", a.out)
