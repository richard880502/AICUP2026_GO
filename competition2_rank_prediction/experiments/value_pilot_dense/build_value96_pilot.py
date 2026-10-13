"""Pilot sets with identical games: dense 4-model features alone vs the same plus dense value tokens (as extra feature channels)."""
import numpy as np, rank_lib as lib
R = lib.ROOT
for split in ("train", "val"):
    val = lib.load_features(R / f"experiments/value_pilot_dense/features/feat96val/{split}")           # [G, 96, 32] value tokens
    me = lib.load_features(R / f"features/feat96/{split}")
    idx = {k: i for i, k in enumerate(me["game_key"])}
    rows = np.array([idx[k] for k in val["game_key"]])
    t, m = me["tokens"][rows], me["masks"][rows]
    assert (m == val["masks"]).mean() > 0.999
    base = {k: me[k][rows] for k in ("game_key", "player_id", "rank")}
    flat = t.reshape(len(t), t.shape[1], -1)
    # store the combined channels as one extra "model": [G, 96, 4*3 + 32] reshaped to [G, 96, K, 3] is awkward, so pad to a multiple of 3
    both = np.concatenate([flat, val["tokens"]], 2)
    pad = (-both.shape[2]) % 3
    both = np.pad(both, ((0, 0), (0, 0), (0, pad)))
    both = both.reshape(len(t), t.shape[1], -1, 3)
    for name, tok in (("dense", t), ("dense_value", both)):
        out = R / f"experiments/value_pilot_dense/features/pilot_val96/{name}/{split}"; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "feat_0000.npz", tokens=tok, masks=m, **base)
        print(split, name, tok.shape)
