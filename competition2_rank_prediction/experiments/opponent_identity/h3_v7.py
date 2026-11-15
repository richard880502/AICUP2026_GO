"""Opponent identification with v7 (960-D) and the v4+v7 average.

For holdout3 and the official test.

Reproduces Competition 2 rank submission v47, step 7 of 8 (after h3_dump.py
and Competition 1's extract_rich.py on `h3_opp.csv` / `test_opp.csv`).
Writes `opp_probe/h3_sims_v7.pt`, `opp_probe/h3_sims_v47.pt` (h3_sims.pt
with the v7 or the v4+v7 average similarities, without the vectors) and
`opp_probe/test_opp_sims_v7.npy`, read by h3_final_v47.py.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 python -m \
      competition2_rank_prediction.experiments.opponent_identity.h3_v7
"""
import csv

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.trunk_style import apply, train
from competition1_player_identification.experiments.trunk_style import rich_full as full

csv.field_size_limit(10**9)
# The research machine's locations for the opponent-identity files and their 960-D tokens.
PROBE = "/tmp2/richard/claude_scratch/opp_probe"
RICH = train.pathlib.Path("/mnt/share_data_77/richard/AICUP2026_GO/opp_probe")


def vectors(name, model, scaling):
    """Unit v7 game vectors of one opponent-side game list."""
    with open(f"{PROBE}/{name}.csv", encoding="utf-8", newline="") as f:
        colors = np.asarray([int(r["color"] == "W") for r in csv.DictReader(f)], np.int64)
    masks = train.load_shards(RICH / f"{name}_rich", "masks", 0)
    data = {
        "trunk": train.load_shards(RICH / f"{name}_rich", "tokens", 0),
        "engine": np.zeros((len(masks), 160, 42), np.float16),
        "masks": masks, "colors": colors, "has_engine": np.zeros(len(masks), bool),
    }
    return F.normalize(
        torch.from_numpy(train.encode(model, data, scaling, np.arange(len(masks))))
        .float().to(train.DEVICE),
        dim=-1,
    )


def main() -> None:
    """Writes h3_sims_v7.pt, h3_sims_v47.pt and test_opp_sims_v7.npy."""
    full.use_rich()
    model, scaling, _, _ = apply.load_model("style_v7")
    # torch Parameter method; pylint cannot see that it is callable.
    classes = F.normalize(
        model.class_weights.detach().float(), dim=-1  # pylint: disable=not-callable
    )
    saved = torch.load(f"{PROBE}/h3_sims.pt", weights_only=False)
    sims7 = vectors("h3_opp", model, scaling) @ classes.T
    own = torch.from_numpy(saved["own_class"]).to(train.DEVICE)
    keep = own >= 0
    sims7[torch.arange(len(sims7), device=train.DEVICE)[keep], own[keep]] = -1
    sims4 = saved["sims"].float().to(train.DEVICE)
    print(
        "rank agreement of top class v4 vs v7",
        round(float((sims4.argmax(1) == sims7.argmax(1)).float().mean()), 3),
    )
    for tag, sims in (("v7", sims7), ("v47", (sims4 + sims7) / 2)):
        out = dict(saved)
        out["sims"] = sims.half().cpu()
        del out["vectors"]
        torch.save(out, f"{PROBE}/h3_sims_{tag}.pt")
    test7 = vectors("test_opp", model, scaling) @ classes.T
    np.save(f"{PROBE}/test_opp_sims_v7.npy", test7.half().cpu().numpy())
    print("saved")


if __name__ == "__main__":
    main()
