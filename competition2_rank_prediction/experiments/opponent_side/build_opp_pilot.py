"""Builds the three pilot datasets (player-only / opponent-only / player+opponent) on identical games."""
import pathlib, numpy as np, rank_lib as lib
R = lib.ROOT
for split in ("train", "val"):
    opp = lib.load_features(R / f"experiments/opponent_side/features/feat96opp/{split}")
    me = lib.load_features(R / f"features/feat96/{split}")
    idx = {k: i for i, k in enumerate(me["game_key"])}
    rows = np.array([idx[k] for k in opp["game_key"]])
    base = {k: me[k][rows] for k in ("game_key", "player_id", "rank")}
    assert (base["game_key"] == opp["game_key"]).all()
    variants = {
        "player": (me["tokens"][rows], me["masks"][rows]),
        "opponent": (opp["tokens"], opp["masks"]),
        "both": (np.concatenate([me["tokens"][rows], opp["tokens"]], 1), np.concatenate([me["masks"][rows], opp["masks"]], 1)),
    }
    for name, (t, m) in variants.items():
        out = R / f"experiments/opponent_side/features/pilot_opp/{name}/{split}"; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "feat_0000.npz", tokens=t, masks=m, **base)
        print(split, name, t.shape, "valid/game %.1f" % m.sum(1).mean())
