"""Runs the v5 matching recipe on style_v7 (960-D richer tokens), new pools only.

Reproduces Competition 1 style submission v7 (`submission_style_v7_match`),
step 4 of 5 of the v9 chain (after rich_full.py has trained `style_v7`).
Its `rerank_v7/` scores (`pool_scores_{pub2,priv2}.npz`, the `style_v7_match`
head) are what ensemble_v9.py adds to v4's.

Original pools keep the v6 answers (already near perfect). Steps:
cohorts -> probe -> match -> validation -> pools.

Example (repository root), once per step:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.rich_pipeline cohorts
"""
import sys

import numpy as np

from competition1_player_identification.experiments.trunk_style import (
    apply, matching, matching_apply, rerank, rerank_inputs, train,
)
from competition1_player_identification.experiments.trunk_style import rich_full as full
from competition1_player_identification.production_pipeline import open_set_official as official

R = train.ROOT / "rerank_v7"
OUTPUT = train.ROOT / "submission_style_v7_match"


def pool_tokens(name):
    """rerank_inputs.pool_tokens on the `_rich` shards instead of `_all`."""
    games, positions, names = official.pool_games(name)
    parts = {
        key: np.concatenate([
            train.load_shards(train.ROOT / f"{name}_{group}_rich", key, 0)
            for group in ("candidates", "queries")
        ])
        for key in ("tokens", "masks")
    }
    tokens = {
        "trunk": parts["tokens"],
        "engine": train.scatter_engine(games["decisions"], games["masks"], parts["masks"]),
        "masks": parts["masks"],
        "colors": games["colors"].astype(np.int64),
    }
    return tokens, positions, names


def main() -> None:
    """Points the matching modules at style_v7 and rerank_v7, then runs one step."""
    full.use_rich()
    train.use_all_moves = full.use_rich
    # train.np is the numpy module; this duck-typed stand-in only overrides zeros().
    train.np = full.DiskBackedNumpy()  # type: ignore[assignment]
    full.STORE = full.STORE.replace("rich_tokens", "rich_tokens_pipe")
    R.mkdir(exist_ok=True)
    rerank.RERANK = matching.RERANK = matching_apply.RERANK = rerank_inputs.RERANK = R
    load = apply.load_model
    # Every later style_v4 load gets style_v7 instead.
    apply.load_model = lambda name: load("style_v7" if name == "style_v4" else name)
    matching_apply.SOURCE = matching_apply.PREVIOUS = train.ROOT / "submission_style_v6_orig"
    rerank_inputs.pool_tokens = pool_tokens
    step = sys.argv[1]
    if step == "cohorts":
        rerank_inputs.cohorts("style_v7", R)
        link = R / "cohorts_style_v4.npz"
        link.unlink(missing_ok=True)
        link.symlink_to("cohorts_style_v7.npz")
    elif step == "probe":
        matching_apply.probe(["hidden"], R)
    elif step == "match":
        sys.argv = ["matching", "--epochs", "8", "--name", "style_v7_match"]
        matching.main()
    elif step == "validation":
        matching_apply.validation("style_v7_match")
    elif step == "pools":
        matching_apply.pools("style_v7_match", OUTPUT, R)


if __name__ == "__main__":
    main()
