"""Trains matched game-coverage arms after complete feature extraction.

Run with PYTHONPATH=competition2_rank_prediction python -m followed by this
module path. This never reads official test features or changes submissions.
"""

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.dense_data_scale import (
    coverage_data,
)
from competition2_rank_prediction.experiments.human_trunk import temporal


def game_pools(source: pathlib.Path, data: dict, optimizer: list) -> dict:
    """Aligns nested manifest keys to feature rows for identical players."""
    small = []
    for path in sorted((source / "cap2").glob("games_*.csv")):
        small.extend(pd.read_csv(path, dtype=str)["game_key"].tolist())
    small_keys = set(small)
    pools = {}
    for cap in (2, 8):
        pools[cap] = np.asarray(
            [
                np.flatnonzero(
                    (data["player_id"] == player)
                    & np.asarray(
                        [
                            cap == 8 or key in small_keys
                            for key in data["game_key"]
                        ]
                    )
                )
                for player in optimizer
            ]
        )
        if pools[cap].shape != (400, cap):
            raise ValueError(f"Incorrect optimizer game pools for cap {cap}")
    return pools


def normalize(data: dict, evaluation: dict, selected: np.ndarray) -> dict:
    """Fits both arms' common channel moments on cap2 optimizer rows only."""
    tokens = transformer_rank.decision_tokens(data["tokens"])
    moments = tokens[selected][data["masks"][selected]].astype(np.float64)
    mean = moments.mean(axis=0).astype(np.float32)
    std = (moments.std(axis=0) + 1e-6).astype(np.float32)
    for payload in (data, evaluation):
        raw = transformer_rank.decision_tokens(payload["tokens"])
        payload["x"] = np.where(
            payload["masks"][..., None], (raw - mean) / std, 0
        ).astype(np.float32)
        payload["y"] = rank_lib.rank_ids(payload["rank"])
    return {"mean": mean, "std": std}


def prepare(root: pathlib.Path) -> tuple[dict, dict, dict]:
    """Aligns nested game pools and fits shared optimizer-only normalization."""
    source = root / "production/game_coverage_manifest_20261109"
    directory = root / "production/game_coverage_pilot_20261109/features"
    manifest = coverage_data.read_manifest(source / "cap8")
    coverage_data.validate_features(directory, manifest)
    split = coverage_data.partitions(manifest)
    data = rank_lib.load_features(directory)
    evaluation = rank_lib.load_features(
        root / "runtime/competition2_research/features/feat96/val"
    )
    if set(data["player_id"]) & set(evaluation["player_id"]):
        raise ValueError("Pilot and evaluation players overlap")
    audit: dict = {
        "pools": game_pools(source, data, sorted(split["optimizer"])),
        "inner": np.flatnonzero(
            np.isin(data["player_id"], list(split["inner"]))
        ),
        "optimizer_players": sorted(split["optimizer"]),
        "inner_players": sorted(split["inner"]),
    }
    audit.update(normalize(data, evaluation, audit["pools"][2].ravel()))
    return data, evaluation, audit


def update(model: torch.nn.Module, optimizer: tuple, batch: tuple) -> None:
    """Applies the fixed NLL augmentation and clipped optimizer update."""
    tokens, masks, labels = batch
    valid = masks & (torch.rand_like(masks, dtype=torch.float32) > 0.15)
    valid[:, 0] |= ~valid.any(1)
    noisy = tokens + 0.05 * torch.randn_like(tokens)
    loss = torch.nn.functional.nll_loss(model(noisy, valid), labels)
    optimizer[0].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer[0].step()
    optimizer[1].step()


def fit(model: torch.nn.Module, context: dict, run: tuple, output) -> dict:
    """Keeps every inner metric and returns the minimum-NLL model state."""
    cap, seed = run
    player_generator = np.random.default_rng(seed + 20261109)
    game_generator = np.random.default_rng(seed + 20261110)
    history: list[float] = []
    best = {}
    for epoch in range(40):
        model.train()
        for _ in range(64):
            players = player_generator.integers(400, size=128)
            # Continuous matched draws prevent coverage from changing player RNG.
            games = (game_generator.random(128) * cap).astype(int)
            rows = context["pool"][players, games]
            update(
                model,
                context["optimizer"],
                tuple(t[rows] for t in context["tensors"]),
            )
        metric = transformer_rank.inner_metric(
            model, context["inner"], "nll", torch.device("cuda:1")
        )
        if not history or metric < min(history):
            best = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        history.append(metric)
        (output / f"cap{cap}_seed{seed}_history.json").write_text(
            json.dumps(history)
        )
        if (epoch + 1) % 5 == 0:
            print(
                f"cap={cap} seed={seed} epoch={epoch+1} nll={metric:.5f}",
                flush=True,
            )
    return {"model": best, "best_epoch": int(np.argmin(history)) + 1}


def train_arm(data: dict, audit: dict, run: tuple, output) -> torch.nn.Module:
    """Trains one fixed-budget arm, selecting on shared inner-player NLL."""
    cap, seed = run
    torch.manual_seed(seed)
    device = torch.device("cuda:1")
    model = transformer_rank.build_model(
        data["x"].shape[2], 96, temporal.recipe(), device, None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.05
    )
    context = {
        "optimizer": (
            optimizer,
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=40 * 64
            ),
        ),
        "pool": audit["pools"][cap],
        "tensors": tuple(
            torch.from_numpy(data[key]).to(device)
            for key in ("x", "masks", "y")
        ),
        "inner": tuple(
            data[key][audit["inner"]] for key in ("x", "masks", "y")
        ),
    }
    selected = fit(model, context, run, output)
    model.load_state_dict(selected["model"])
    torch.save(
        {
            **selected,
            "mean": audit["mean"],
            "std": audit["std"],
            "optimizer_players": audit["optimizer_players"],
            "inner_players": audit["inner_players"],
        },
        output / f"cap{cap}_seed{seed}.pt",
    )
    return model


def main() -> None:
    """Runs all six preregistered arms only after cache validation passes."""
    root = pathlib.Path.cwd()
    output = root / "production/game_coverage_pilot_20261109/training"
    if output.exists():
        raise FileExistsError(f"Refusing existing training output: {output}")
    data, evaluation, audit = prepare(root)
    output.mkdir(parents=True)
    inputs = sorted(output.parent.joinpath("features").glob("feat_*.npz"))
    inputs.extend(sorted(pathlib.Path(__file__).parent.glob("coverage_*.py")))
    inputs.extend(
        sorted(
            (root / "runtime/competition2_research/features/feat96/val").glob(
                "feat_*.npz"
            )
        )
    )
    inputs.extend(
        root / "competition2_rank_prediction" / name
        for name in ("rank_lib.py", "transformer_rank.py")
    )
    inputs.extend(
        sorted(
            (root / "production/game_coverage_manifest_20261109/cap2").glob(
                "games_*.csv"
            )
        )
    )
    (output / "input_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in inputs
            },
            indent=2,
        )
    )
    torch.set_num_threads(4)
    for seed in range(3):
        for cap in (2, 8):
            model = train_arm(data, audit, (cap, seed), output)
            probabilities = transformer_rank.predict(
                model,
                evaluation["x"],
                evaluation["masks"],
                torch.device("cuda:1"),
            )
            np.savez_compressed(
                output / f"cap{cap}_seed{seed}_validation.npz",
                probabilities=probabilities,
                game_key=evaluation["game_key"],
                player=evaluation["player_id"],
                y=evaluation["y"],
            )


if __name__ == "__main__":
    main()
