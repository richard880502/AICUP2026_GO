"""Samples a small player-diverse pilot set (N games per rank, one game per player where possible)."""
import sys, pandas as pd
n = int(sys.argv[1]); out = sys.argv[2]
parts = []
for r in ["D","C","B","A","1D","2D","3D","4D","5D","6D"]:
    d = pd.read_csv(f"data/training/train_{r}.csv")
    d = d.sample(frac=1, random_state=0).drop_duplicates("player_id").head(n)
    d["game_key"] = r + "_" + d["game_id"]
    parts.append(d)
pd.concat(parts, ignore_index=True)[["game_key","player_id","rank","color","sgf_content"]].to_csv(out, index=False)
