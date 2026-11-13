"""Trains a per-move style encoder on frozen HumanSL trunk tokens.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.train

Each game is a sequence of up to 64 own decisions. A token concatenates the
769-D trunk descriptor with the 41 engine scalars of the same decision.
Training uses only the open-set v2 training players (ArcFace + supervised
contrastive, P x K batches). The 600-player validation cohort selects the
epoch; the final cohort is never read here.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.open_set_v2 import dataset
from competition1_player_identification.experiments.open_set_v2 import (
    evaluation,
)
from competition1_player_identification.experiments.open_set_v2 import fusion

OPEN = dataset.ROOT
ROOT = pathlib.Path("production/trunk_style_20261113")
DEVICE = "cuda:0"


def load_shards(directory: pathlib.Path, key: str, limit: int) -> np.ndarray:
    """Concatenates one array from complete shards in sorted order."""
    paths = sorted(directory.glob("shard_*.npz"))
    if limit:
        paths = paths[:limit]
    parts = []
    for path in paths:
        with np.load(path) as arrays:
            parts.append(np.asarray(arrays[key]))
    return np.concatenate(parts)


def fill_shards(
    paths: list, tokens: np.ndarray, masks: np.ndarray, offset: int
) -> int:
    """Copies trunk shards into preallocated arrays; returns the next offset."""
    for path in paths:
        with np.load(path) as arrays:
            count = len(arrays["masks"])
            tokens[offset : offset + count] = arrays["tokens"]
            masks[offset : offset + count] = arrays["masks"]
            offset += count
    return offset


def load_corpus(  # pylint: disable=too-many-locals
    limit: int, extra: bool = False
) -> dict:
    """Loads trunk tokens, engine scalars where cached, colors and splits.

    The 16-game corpus has engine scalars; optional extra training games
    (corpus2) have none and are flagged so their engine channels stay zero.
    """
    base_paths = sorted((ROOT / "corpus").glob("shard_*.npz"))
    if limit:
        base_paths = base_paths[:limit]
    extra_paths = sorted((ROOT / "corpus2").glob("shard_*.npz")) if extra else []
    sizes = []
    for path in base_paths + extra_paths:
        with np.load(path) as arrays:
            sizes.append(len(arrays["masks"]))
    count = sum(sizes[: len(base_paths)])
    total = sum(sizes)
    trunk = np.zeros((total, 64, 769), np.float16)
    masks = np.zeros((total, 64), bool)
    fill_shards(extra_paths, trunk, masks, fill_shards(base_paths, trunk, masks, 0))
    engine = np.zeros((total, 64, 41), np.float16)
    engine[:count] = load_shards(
        OPEN / "engine_features/candidates", "tokens", (count + 127) // 128
    )[:count]
    engine_masks = load_shards(
        OPEN / "engine_features/candidates", "masks", (count + 127) // 128
    )[:count]
    agreement = float((masks[:count] == engine_masks).all(1).mean())
    masks[:count] &= engine_masks
    everything = np.load(OPEN / "engine_inputs/global_indices.npy")
    selected = everything[:count]
    if extra:
        selected = np.concatenate(
            (selected, np.load(ROOT / "inputs/corpus2_global_indices.npy"))
        )
    if len(selected) != total:
        raise ValueError("Extra corpus extraction is incomplete")
    raw = dataset.load_arrays(OPEN / "raw_cache", ("players", "colors"))
    with np.load(OPEN / "clean_splits.npz") as splits:
        train_rows = np.flatnonzero(np.isin(selected, splits["train"]))
        validation = np.searchsorted(everything, splits["validation"])
    _, groups = dataset.player_groups(raw["players"][selected[train_rows]])
    groups = [train_rows[group] for group in groups]
    return {
        "trunk": trunk,
        "engine": engine,
        "masks": masks,
        "has_engine": np.arange(total) < count,
        "colors": raw["colors"][selected].astype(np.int64),
        "groups": groups,
        "train_rows": train_rows,
        "validation": validation,
        "mask_agreement": agreement,
    }


def fit_scaling(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fits per-channel statistics on a sample of valid training decisions."""
    generator = np.random.default_rng(20261113)
    eligible = data["train_rows"][data["has_engine"][data["train_rows"]]]
    rows = generator.choice(eligible, min(8192, len(eligible)), replace=False)
    rows.sort()
    valid = data["masks"][rows]
    values = np.concatenate(
        (data["trunk"][rows][valid], data["engine"][rows][valid]), axis=1
    ).astype(np.float32)
    return values.mean(0), values.std(0).clip(min=1e-3)


class StyleEncoder(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """Pools one game's decision tokens into a unit style vector."""

    def __init__(self, classes: int, width: int = 256, dimension: int = 256):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(810, width), torch.nn.LayerNorm(width)
        )
        self.step = torch.nn.Parameter(torch.randn(1, 64, width) * 0.02)
        self.color = torch.nn.Embedding(2, width)
        block = torch.nn.TransformerEncoderLayer(
            width,
            4,
            dim_feedforward=width * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            block, 4, enable_nested_tensor=False
        )
        self.norm = torch.nn.LayerNorm(width)
        self.pool = torch.nn.Linear(width, 1)
        self.output = torch.nn.Linear(width, dimension)
        self.class_weights = torch.nn.Parameter(torch.empty(classes, dimension))
        torch.nn.init.xavier_uniform_(self.class_weights)

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor, colors: torch.Tensor
    ) -> torch.Tensor:
        """Returns L2-normalized game embeddings; padded decisions are ignored."""
        hidden = self.project(tokens) + self.step + self.color(colors)[:, None]
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=~valid))
        weights = self.pool(hidden).squeeze(-1).float()
        weights = weights.masked_fill(~valid, float("-inf")).softmax(1)
        pooled = (weights[..., None] * hidden.float()).sum(1)
        return F.normalize(self.output(pooled), dim=-1)


def batch_tensors(data: dict, scaling: tuple, rows: np.ndarray) -> tuple:
    """Moves standardized tokens for the given sorted-or-not rows to the GPU."""
    order = np.argsort(rows)
    inverse = np.argsort(order)
    ordered = rows[order]
    values = np.concatenate(
        (data["trunk"][ordered], data["engine"][ordered]), axis=2
    )[inverse]
    tokens = torch.from_numpy(values).to(DEVICE).float()
    mean, scale = scaling
    tokens = ((tokens - mean) / scale).clamp(-10, 10)
    valid = torch.from_numpy(data["masks"][rows]).to(DEVICE)
    tokens = tokens * valid[..., None]
    if "has_engine" in data:
        present = torch.from_numpy(data["has_engine"][rows]).to(DEVICE)
        tokens[..., 769:] = tokens[..., 769:] * present[:, None, None]
    colors = torch.from_numpy(data["colors"][rows]).to(DEVICE)
    return tokens, valid, colors


def encode(model: StyleEncoder, data: dict, scaling: tuple, rows: np.ndarray):
    """Encodes rows in bounded batches without dropout or gradient."""
    model.eval()
    outputs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), 1024):
            tokens, valid, colors = batch_tensors(
                data, scaling, rows[start : start + 1024]
            )
            outputs.append(model(tokens, valid, colors).float().cpu().numpy())
    return np.concatenate(outputs)


def losses(model: StyleEncoder, embeddings: torch.Tensor, labels: torch.Tensor):
    """Combines additive-angular classification with supervised contrast."""
    cosine = embeddings @ F.normalize(model.class_weights, dim=-1).T
    margin = torch.cos(
        torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)) + 0.2
    )
    target = F.one_hot(  # pylint: disable=not-callable
        labels, cosine.shape[1]
    ).bool()
    arcface = F.cross_entropy(30 * torch.where(target, margin, cosine), labels)
    similarity = embeddings @ embeddings.T / 0.1
    same = labels[:, None] == labels[None]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    similarity = similarity.masked_fill(eye, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, 1, keepdim=True)
    positives = same & ~eye
    contrast = -(log_probability.masked_fill(~positives, 0).sum(1)) / positives.sum(
        1
    ).clamp(min=1)
    return arcface, contrast.mean()


def sample_batch(data: dict, generator: np.random.Generator, players: int):
    """Draws four distinct games from each of P players with enough games."""
    eligible = [index for index, group in enumerate(data["groups"]) if len(group) >= 4]
    owners = generator.choice(eligible, players, replace=False)
    rows = np.concatenate(
        [generator.choice(data["groups"][owner], 4, replace=False) for owner in owners]
    )
    return rows, np.repeat(owners, 4)


def validate(model: StyleEncoder, data: dict, scaling: tuple) -> dict:
    """Scores the 600-player validation cohort and its fusion with open-set."""
    embeddings = encode(model, data, scaling, data["validation"].ravel())
    scores = evaluation.similarity_scores(
        np.reshape(embeddings, (600, 15, -1)), "cuda:0"
    )
    base = fusion.standardize(np.load(OPEN / "fusion_validation_scores.npy"))
    report: dict = {
        name: fusion.score_value(value) for name, value in scores.items()
    }
    best = max(report, key=lambda name: report[name])
    extra = fusion.standardize(scores[best])
    trials = {
        weight: fusion.score_value(base + weight * extra)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    }
    report["aggregation"] = best
    report["fusion_trials"] = trials
    report["open_set_fusion_only"] = trials[0.0]
    report["fused_best"] = max(trials.values())
    return report


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Trains, selects the epoch on validation and saves the best checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--players", type=int, default=128)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--name", default="style_v1")
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also train on corpus2 games, which have no engine scalars.",
    )
    parser.add_argument("--engine-dropout", type=float, default=0.0)
    parser.add_argument(
        "--no-engine",
        action="store_true",
        help="Ablation: zero the 41 engine scalars, keep trunk tokens only.",
    )
    arguments = parser.parse_args()
    torch.manual_seed(20261113)
    data = load_corpus(arguments.limit_shards, arguments.extra)
    scaling_arrays = fit_scaling(data)
    if arguments.no_engine:
        data["has_engine"] = np.zeros(len(data["trunk"]), bool)
    scaling = tuple(torch.from_numpy(value).to(DEVICE) for value in scaling_arrays)
    model = StyleEncoder(len(data["groups"])).to(DEVICE)
    steps = max(1, len(data["train_rows"]) // (arguments.players * 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 5e-4, total_steps=steps * arguments.epochs, pct_start=0.1
    )
    generator = np.random.default_rng(20261113)
    output = ROOT / arguments.name
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "games": len(data["trunk"]),
                "train_players": len(data["groups"]),
                "steps_per_epoch": steps,
                "mask_agreement": data["mask_agreement"],
            }
        ),
        flush=True,
    )
    history, best = [], -1.0
    complete = not arguments.limit_shards
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        started = time.monotonic()
        totals = np.zeros(2)
        for _ in range(steps):
            rows, labels = sample_batch(data, generator, arguments.players)
            tokens, valid, colors = batch_tensors(data, scaling, rows)
            dropped = torch.rand(len(rows), device=DEVICE) < arguments.engine_dropout
            tokens[dropped, :, 769:] = 0
            keep = torch.rand(valid.shape, device=DEVICE) > 0.15
            valid = valid & (keep | (valid.sum(1, keepdim=True) <= 8))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(tokens, valid, colors)
            arcface, contrast = losses(
                model, embeddings.float(), torch.from_numpy(labels).to(DEVICE)
            )
            optimizer.zero_grad(set_to_none=True)
            (arcface + contrast).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals += (arcface.item(), contrast.item())
        row = {
            "epoch": epoch,
            "arcface": totals[0] / steps,
            "contrast": totals[1] / steps,
            "seconds": round(time.monotonic() - started, 1),
        }
        if complete and (epoch % 2 == 0 or epoch == arguments.epochs):
            row["validation"] = validate(model, data, scaling)
            score = row["validation"][row["validation"]["aggregation"]]
            if score > best:
                best = score
                torch.save(
                    {
                        "state": model.state_dict(),
                        "mean": scaling_arrays[0],
                        "scale": scaling_arrays[1],
                        "classes": len(data["groups"]),
                        "epoch": epoch,
                        "validation": row["validation"],
                    },
                    output / "checkpoint.pt",
                )
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=1))
        print(json.dumps(row), flush=True)
    print(json.dumps({"best_validation": best}), flush=True)


if __name__ == "__main__":
    main()
