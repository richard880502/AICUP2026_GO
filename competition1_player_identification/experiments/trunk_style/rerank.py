"""Trains the listwise second-stage reranker on game embeddings.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.rerank \
      --seed 0 --steps 6000 --name listwise_s0

Episodes mirror an official new pool: 600 candidates with 10 reference games,
400 questions with distinct targets and 1/2/3/5 query games. The first stage
(centroid cosine + CSLS) proposes 20 candidates; a small Transformer compares
the query games with each candidate's reference games, then the 20 candidates
with each other, and is trained with cross-entropy on the true candidate.

Training players come from cross-fitted encoders (each player is embedded by
an encoder that never saw them; see rerank_inputs). The validation cohort
selects the step; the final cohort is reported once at that step.
"""

import argparse
import json
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.trunk_style import train

RERANK = train.ROOT / "rerank"
COUNTS = (1, 2, 3, 5)
TOP = 20
DEVICE = train.DEVICE


def first_stage(
    games: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor
) -> tuple:
    """games [600,15,D] unit; returns standardized scores and CSLS [Q,600]."""
    centroids = F.normalize(games[:, :10].mean(1), dim=-1)
    queries = games[targets, 10:]
    keep = (torch.arange(5, device=games.device)[None] < counts[:, None]).float()
    pooled = F.normalize((queries * keep[..., None]).sum(1), dim=-1)
    scores = pooled @ centroids.T
    z = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)
    row = z.topk(5, dim=1).values.mean(1, keepdim=True)
    column = z.topk(5, dim=0).values.mean(0, keepdim=True)
    return z, 2 * z - row - column, column.expand_as(z), queries, keep


def build(  # pylint: disable=too-many-locals
    games: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor, force: bool
) -> dict:
    """Returns reranker inputs for the 20 proposed candidates of each query.

    With force, a true candidate missing from the proposals replaces the
    last one (training only).
    """
    z, csls, column, queries, keep = first_stage(games, targets, counts)
    order = csls.argsort(dim=1, descending=True)
    chosen = order[:, :TOP].clone()
    hit = chosen == targets[:, None]
    present = hit.any(1)
    if force:
        chosen[~present, TOP - 1] = targets[~present]
        hit = chosen == targets[:, None]
    label = hit.float().argmax(1)
    references = games[chosen][:, :, :10]  # [Q, TOP, 10, D]
    pair = torch.einsum("qcd,qtrd->qtcr", queries, references)  # [Q,TOP,5,10]
    mask = keep[:, None, :, None].expand_as(pair)
    count = counts[:, None].float()
    mean = (pair * mask).sum((2, 3)) / (10 * count)
    low = pair.masked_fill(mask == 0, -9)
    high = pair.masked_fill(mask == 0, 9)
    row_max = (low.max(3).values * keep[:, None]).sum(2) / count
    col_max = low.max(2).values.mean(2)
    scalars = torch.stack(
        (
            z.gather(1, chosen),
            csls.gather(1, chosen),
            column.gather(1, chosen),
            torch.arange(TOP, device=games.device).float().expand_as(chosen) / TOP,
            mean,
            low.amax((2, 3)),
            high.amin((2, 3)),
            row_max,
            col_max,
            (csls.gather(1, chosen) - csls.gather(1, chosen[:, :1])),
        ),
        dim=-1,
    )
    return {
        "queries": queries,
        "keep": keep.bool(),
        "references": references,
        "scalars": scalars,
        "counts": counts,
        "label": label,
        "present": present,
        "csls": csls,
        "chosen": chosen,
    }


class Reranker(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """Pairwise game-set comparison followed by a listwise pass."""

    def __init__(self, dimension: int = 256, width: int = 192):
        super().__init__()
        self.project = torch.nn.Linear(dimension, width)
        self.kind = torch.nn.Embedding(2, width)
        self.count = torch.nn.Embedding(6, width)
        self.scalars = torch.nn.Sequential(
            torch.nn.Linear(10, width), torch.nn.GELU(), torch.nn.Linear(width, width)
        )

        def layer() -> torch.nn.TransformerEncoderLayer:
            return torch.nn.TransformerEncoderLayer(
                width, 4, width * 2, 0.1, "gelu", batch_first=True, norm_first=True
            )

        self.pair = torch.nn.TransformerEncoder(
            layer(), 2, enable_nested_tensor=False
        )
        self.merge = torch.nn.Linear(width * 2, width)
        self.listwise = torch.nn.TransformerEncoder(
            layer(), 2, enable_nested_tensor=False
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(width), torch.nn.Linear(width, 1)
        )
        self.prior = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns one logit per proposed candidate, [Q, TOP]."""
        queries, references = batch["queries"], batch["references"]
        size, top = references.shape[:2]
        left = self.project(queries) + self.kind.weight[0]
        left = left + self.count(batch["counts"])[:, None]
        left = left[:, None].expand(size, top, 5, -1)
        right = self.project(references) + self.kind.weight[1]
        tokens = torch.cat((left, right), 2).reshape(size * top, 15, -1)
        valid = torch.cat(
            (
                batch["keep"][:, None].expand(size, top, 5),
                torch.ones(size, top, 10, dtype=torch.bool, device=queries.device),
            ),
            2,
        ).reshape(size * top, 15)
        hidden = self.pair(tokens, src_key_padding_mask=~valid)
        weight = valid[:, :5, None].float()
        query_side = (hidden[:, :5] * weight).sum(1) / weight.sum(1)
        merged = self.merge(torch.cat((query_side, hidden[:, 5:].mean(1)), -1))
        merged = merged.reshape(size, top, -1) + self.scalars(batch["scalars"])
        logits = self.head(self.listwise(merged)).squeeze(-1)
        return logits + self.prior * batch["scalars"][..., 1]


def utility(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Contest utility after reordering the proposed candidates."""
    order = logits.argsort(dim=1, descending=True)
    position = (order == batch["label"][:, None]).float().argmax(1)
    gain = torch.where(position < 5, torch.exp(-position.float()), 0.0)
    return torch.where(batch["present"], gain, 0.0)


def repeated_targets(seed: int, repeats: int = 0) -> np.ndarray:
    """Draws 400 of 600 targets; `repeats` of them duplicate another target."""
    generator = np.random.default_rng(seed)
    picks = generator.permutation(600)[:400]
    if repeats:
        slots = generator.choice(400, repeats, replace=False)
        others = np.setdiff1d(np.arange(400), slots)
        picks[slots] = picks[generator.choice(others, repeats)]
    return picks


def cohort_episodes(embeddings: np.ndarray, seeds: int) -> Iterator[tuple]:
    """Yields official-like episodes of a [600,15,D] cohort, one per seed."""
    games = torch.from_numpy(embeddings).to(DEVICE)
    for seed in range(seeds):
        targets = torch.from_numpy(repeated_targets(seed)).to(DEVICE)
        counts = torch.tensor(COUNTS, device=DEVICE).repeat_interleave(100)
        yield games, targets, counts


def evaluate(model: Reranker, embeddings: np.ndarray, seeds: int = 20) -> dict:
    """Mean utility of the reranker and of its CSLS proposals on a cohort."""
    model.eval()
    learned_parts, proposed_parts = [], []
    with torch.no_grad():
        for games, targets, counts in cohort_episodes(embeddings, seeds):
            batch = build(games, targets, counts, False)
            learned_parts.append(utility(model(batch), batch))
            proposed_parts.append(utility(batch["scalars"][..., 1], batch))
    learned, proposed = torch.stack(learned_parts), torch.stack(proposed_parts)

    def by_count(values: torch.Tensor) -> list:
        return [round(x, 3) for x in values.reshape(-1, 4, 100).mean((0, 2)).tolist()]

    return {
        "reranked": round(learned.mean().item(), 4),
        "csls": round(proposed.mean().item(), 4),
        "by_count": by_count(learned),
        "csls_by_count": by_count(proposed),
    }


class Sampler:  # pylint: disable=too-few-public-methods
    """Draws 600-player episodes from cross-fitted training embeddings."""

    def __init__(self, source: str):
        if source == "cohort":
            games = np.load(RERANK / "cohorts_style_v4.npz")["validation"]
            self.table = torch.arange(600 * 15).reshape(600, 15).to(DEVICE)
            self.sizes = torch.full((600,), 15, device=DEVICE)
            self.folds = [torch.arange(600, device=DEVICE)]
            self.values = torch.from_numpy(
                games.reshape(-1, games.shape[-1])
            ).to(DEVICE)
            return
        with np.load(RERANK / "train_embeddings.npz") as arrays:
            values, owners = arrays["crossfit"], arrays["owners"]
        self.values = F.normalize(torch.from_numpy(values).to(DEVICE), dim=-1)
        players = int(np.max(owners)) + 1
        sizes = np.bincount(owners, minlength=players)
        table = np.zeros((players, int(np.max(sizes))), np.int64)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        order = np.argsort(owners, kind="stable")
        for player in range(players):
            table[player, : sizes[player]] = order[
                starts[player] : starts[player] + sizes[player]
            ]
        self.table = torch.from_numpy(table).to(DEVICE)
        self.sizes = torch.from_numpy(sizes).to(DEVICE)
        eligible = np.flatnonzero(sizes >= 15)
        self.folds = [
            torch.from_numpy(eligible[eligible % 2 == fold]).to(DEVICE)
            for fold in (0, 1)
        ]
        print(json.dumps({"players": [len(f) for f in self.folds]}), flush=True)

    def episode(self, generator: torch.Generator) -> tuple:
        """Returns games [600,15,D], 400 target rows and their query counts."""
        fold = self.folds[
            int(
                torch.randint(
                    len(self.folds), (1,), generator=generator, device=DEVICE
                )
            )
        ]
        players = fold[
            torch.randperm(len(fold), generator=generator, device=DEVICE)[:600]
        ]
        width = self.table.shape[1]
        noise = torch.rand((600, width), generator=generator, device=DEVICE)
        absent = (
            torch.arange(width, device=DEVICE)[None] >= self.sizes[players][:, None]
        )
        noise[absent] = 2
        slots = noise.argsort(1)[:, :15]
        games = self.values[self.table[players].gather(1, slots)]
        targets = torch.randperm(600, generator=generator, device=DEVICE)[:400]
        counts = torch.tensor(COUNTS, device=DEVICE)[
            torch.randint(4, (400,), generator=generator, device=DEVICE)
        ]
        return games, targets, counts


def main() -> None:  # pylint: disable=too-many-locals
    """Trains on sampled episodes and keeps the best validation step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="crossfit",
        help="crossfit (training players) or cohort (validation, debugging).",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="listwise")
    arguments = parser.parse_args()
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=DEVICE).manual_seed(arguments.seed)
    cohorts = np.load(RERANK / "cohorts_style_v4.npz")
    sampler = Sampler(arguments.source)
    model = Reranker().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 3e-4, total_steps=arguments.steps, pct_start=0.05
    )
    best, total = -1.0, 0.0
    for step in range(1, arguments.steps + 1):
        model.train()
        games, targets, counts = sampler.episode(generator)
        batch = build(games, targets, counts, True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        loss = F.cross_entropy(logits.float(), batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % 250 == 0 or step == arguments.steps:
            report: dict = {"step": step, "loss": round(total / 250, 4)}
            total = 0.0
            report["validation"] = evaluate(model, cohorts["validation"])
            if report["validation"]["reranked"] > best:
                best = report["validation"]["reranked"]
                report["final"] = evaluate(model, cohorts["final"])
                torch.save(model.state_dict(), RERANK / f"{arguments.name}.pt")
            print(json.dumps(report), flush=True)
    print(json.dumps({"best_validation": best}))


if __name__ == "__main__":
    main()
