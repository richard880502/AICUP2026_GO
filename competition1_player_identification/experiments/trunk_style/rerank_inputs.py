"""Dumps v4 game embeddings for the second-stage reranker.

Example (repository root), in the order the reranker needs them:
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.rerank_inputs \
      cohorts
    ... rerank_inputs crossfit
    ... rerank_inputs pools

cohorts: validation and final cohorts, [600,15,256] each.
crossfit: every training game encoded by the fold encoder that never saw its
player (style_v4_fold0 / style_v4_fold1, trained with
`train --name style_v4_fold{k} --all-moves --extra --engine-dropout 0.3
--epochs 30 --fold k`).
pools: the official new pools in the same [600,15] layout.
"""

import argparse
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import infer
from competition1_player_identification.experiments.trunk_style import apply
from competition1_player_identification.experiments.trunk_style import rerank
from competition1_player_identification.experiments.trunk_style import train
from competition1_player_identification.production_pipeline import (
    open_set_official as official,
)

RERANK = rerank.RERANK


def pool_tokens(name: str) -> tuple[dict, np.ndarray, dict]:
    """Loads all-moves tokens of one official new pool for train.encode."""
    games, positions, names = official.pool_games(name)
    parts = {
        key: np.concatenate(
            [
                train.load_shards(train.ROOT / f"{name}_{group}_all", key, 0)
                for group in ("candidates", "queries")
            ]
        )
        for key in ("tokens", "masks")
    }
    tokens = {
        "trunk": parts["tokens"],
        "engine": train.scatter_engine(
            games["decisions"], games["masks"], parts["masks"]
        ),
        "masks": parts["masks"],
        "colors": games["colors"].astype(np.int64),
    }
    return tokens, positions, names


def cohorts(name: str, output: pathlib.Path) -> None:
    """Saves game embeddings of the validation and final cohorts."""
    model, scaling, _, _ = apply.load_model(name)
    corpus = train.load_corpus(0)
    _, final = infer.positions(train.OPEN, "final_holdout")
    result: dict = {}
    for key, rows in (("validation", corpus["validation"]), ("final", final)):
        rows = np.asarray(rows)
        print(key, rows.shape, flush=True)
        result[key] = np.reshape(
            train.encode(model, corpus, scaling, rows.ravel()), (600, 15, -1)
        )
    np.savez(output / f"cohorts_{name}.npz", **result)
    print("saved")


def crossfit(output: pathlib.Path) -> None:
    """Encodes every training game with the fold encoder that never saw it."""
    train.use_all_moves()
    corpus = train.load_corpus(0, True)
    groups = corpus["groups"]
    rows = np.concatenate(groups)
    owners = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    embeddings = np.zeros((len(rows), 256), np.float32)
    for fold in (0, 1):
        model, scaling, _, checkpoint = apply.load_model(f"style_v4_fold{fold}")
        print(
            fold, checkpoint["epoch"], checkpoint["validation"]["centroid"], flush=True
        )
        chosen = owners % 2 == fold
        embeddings[chosen] = train.encode(model, corpus, scaling, rows[chosen])
        del model
    np.savez(
        output / "train_embeddings.npz", crossfit=embeddings, owners=owners, rows=rows
    )
    print("saved", embeddings.shape, len(groups))


def pools(name: str, output: pathlib.Path) -> None:
    """Saves game embeddings and question metadata of the official new pools."""
    model, scaling, _, _ = apply.load_model(name)
    for pool in ("pub2", "priv2"):
        tokens, positions, names = pool_tokens(pool)
        embeddings = np.reshape(
            train.encode(model, tokens, scaling, positions.ravel()), (600, 15, -1)
        )
        np.savez(
            output / f"pool_{pool}.npz",
            embeddings=embeddings,
            counts=np.asarray(names["counts"]).astype(np.int64),
            questions=np.asarray(names["questions"]),
            players=np.asarray(names["players"]),
        )
        print(
            pool, embeddings.shape, np.bincount(np.asarray(names["counts"]).astype(int))
        )


def main() -> None:
    """Writes the selected embedding dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", choices=("cohorts", "crossfit", "pools"))
    parser.add_argument("--name", default="style_v4", help="Encoder checkpoint.")
    parser.add_argument("--output", type=pathlib.Path, default=RERANK)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.dump == "cohorts":
        cohorts(arguments.name, arguments.output)
    elif arguments.dump == "crossfit":
        crossfit(arguments.output)
    else:
        pools(arguments.name, arguments.output)


if __name__ == "__main__":
    main()
