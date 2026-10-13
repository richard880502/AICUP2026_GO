"""Per-game probabilities of several saved epochs of one ResNet, on the validation cohort and the holdout games.

Output results/snap_<tag>.npz with val_<epoch> / hold_<epoch> arrays (+ keys, labels). Used to test snapshot ensembles.
"""
import argparse, pathlib
import numpy as np, torch
from torch.utils.data import DataLoader
import rank_lib as lib, resnet_rank as rn

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--epochs", type=int, nargs="+", required=True)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
dev = torch.device(a.device)
R = lib.ROOT

val = lib.load_features(R / "features/feat/val")
z = np.load(R / "features/moves/all_moves512.npz")
index = {k: i for i, k in enumerate(z["game_key"])}
rows = np.array([index[k] for k in val["game_key"]])
sets = {"val": (z["tokens"][rows], z["length"][rows], z["colour"][rows])}
h = np.load(R / "features/moves/holdout_moves.npz")
hold_keys = h["game_key"]
sets["hold"] = (h["tokens"], h["length"], h["colour"])
out = {"val_key": val["game_key"], "hold_key": hold_keys}
for ep in a.epochs:
    model = rn.RankNet(in_channels=500, num_classes=10).to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(R / f"results/{a.tag}_epoch{ep}.pt", map_location="cpu")["state"])
    for name, (tok, ln, col) in sets.items():
        ds = rn.Games(tok, ln, np.zeros(len(tok), np.int64), np.arange(len(tok)), col)
        out[f"{name}_{ep}"] = rn.predict(model, DataLoader(ds, batch_size=256, num_workers=8), dev)
    print("epoch", ep, "done", flush=True)
np.savez(R / f"results/snap_{a.tag}.npz", **out)
