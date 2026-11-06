"""Tokenises the test games (features/shards/test/games_*.csv) the same way prep_moves.py does.

Output features/moves/test_moves.npz: tokens [N, 512] uint16, length, colour (0 = labelled player is
black), game_key.
A game whose colour is not B / W (test_to_games.py already drops those) is kept with length 0, which
predict_test.py treats as "no ResNet input" instead of silently reading it as Black.
"""
import argparse
import pathlib
import re

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERN = re.compile(r";([BW])\[([a-s]{0,2})\]")
PAD = 0xFFFF


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--shards", type=pathlib.Path, default=ROOT / "features/shards/test")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/moves/test_moves.npz")
    ap.add_argument("--length", type=int, default=512)
    return ap.parse_args()


def main() -> None:
    """Writes the token archive."""
    args = parse_args()
    shards = sorted(args.shards.glob("games_*.csv"))
    if not shards:
        raise SystemExit(f"no games_*.csv in {args.shards}: run scripts/test_to_games.py first")
    games = pd.concat([pd.read_csv(p, dtype={"game_key": str, "player_id": str})
                       for p in shards], ignore_index=True)
    tokens = np.full((len(games), args.length), PAD, np.uint16)
    lengths = np.zeros(len(games), np.int16)
    colour_ok = games["color"].isin(["B", "W"]).to_numpy()
    for i, sgf in enumerate(games["sgf_content"]):
        if not colour_ok[i] or not isinstance(sgf, str):
            continue
        moves = PATTERN.findall(sgf)[:args.length]
        lengths[i] = len(moves)
        for j, (c, xy) in enumerate(moves):
            pos = 361 if len(xy) < 2 else (ord(xy[1]) - 97) * 19 + (ord(xy[0]) - 97)
            tokens[i, j] = pos | ((c == "W") << 9)
    if not colour_ok.all():
        print(f"WARNING: {int((~colour_ok).sum())} games without a B / W colour kept with length 0")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, tokens=tokens, length=lengths,
             colour=(games["color"] == "W").to_numpy(np.int8),
             game_key=games["game_key"].to_numpy(dtype="U32"),
             player_id=games["player_id"].to_numpy(dtype="U32"))
    print(len(games), "games tokenised ->", args.out)


if __name__ == "__main__":
    main()
