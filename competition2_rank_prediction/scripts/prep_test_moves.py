"""Tokenises the test games (features/shards/test/games_*.csv) the same way prep_moves.py does.

Output features/moves/test_moves.npz: tokens [N, 512] uint16, length, colour (0 = labelled player is black), game_key.
"""
import argparse, pathlib, re
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERN = re.compile(r";([BW])\[([a-s]{0,2})\]")
PAD = 0xFFFF
ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=pathlib.Path, default=ROOT / "features/shards/test")
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/moves/test_moves.npz")
ap.add_argument("--length", type=int, default=512)
a = ap.parse_args()

games = pd.concat([pd.read_csv(p) for p in sorted(a.shards.glob("games_*.csv"))], ignore_index=True)
tokens = np.full((len(games), a.length), PAD, np.uint16)
lengths = np.zeros(len(games), np.int16)
for i, sgf in enumerate(games["sgf_content"]):
    moves = PATTERN.findall(sgf)[:a.length]
    lengths[i] = len(moves)
    for j, (c, xy) in enumerate(moves):
        pos = 361 if len(xy) < 2 else (ord(xy[1]) - 97) * 19 + (ord(xy[0]) - 97)
        tokens[i, j] = pos | ((c == "W") << 9)
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, tokens=tokens, length=lengths, colour=(games["color"] == "W").to_numpy(np.int8),
         game_key=games["game_key"].to_numpy(dtype="U32"), player_id=games["player_id"].to_numpy(dtype="U32"))
print(len(games), "games tokenised ->", a.out)
