"""Fine-tunes the v4 style encoder with a per-move matching (MaxSim) head.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.matching \
      --epochs 8 --name style_v5_match

Every own move gets a 64-D unit vector from the encoder's contextual hidden
state. A query game is scored against a player's reference games by letting
each query move find its most similar reference move and averaging those
similarities. The game-vector losses (ArcFace + supervised contrast) are kept
so one model yields both scores. Validation cohort selects the epoch.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from competition1_player_identification.experiments.trunk_style import apply
from competition1_player_identification.experiments.trunk_style import rerank
from competition1_player_identification.experiments.trunk_style import train

DEVICE = train.DEVICE
COUNTS = (1, 2, 3, 5)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
RERANK = rerank.RERANK


def contextual(
    encoder: train.StyleEncoder,
    tokens: torch.Tensor,
    valid: torch.Tensor,
    colors: torch.Tensor,
) -> torch.Tensor:
    """Returns the encoder's per-move hidden states before pooling."""
    hidden = encoder.project(tokens) + encoder.step + encoder.color(colors)[:, None]
    return encoder.norm(encoder.encoder(hidden, src_key_padding_mask=~valid))


class TokenModel(torch.nn.Module):
    """Style encoder plus a per-move matching head."""

    def __init__(self, encoder: train.StyleEncoder, width: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(256, width)
        self.log_scale = torch.nn.Parameter(torch.tensor(np.log(30.0)))

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor, colors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns unit game vectors [N,256] and unit move vectors [N,L,64]."""
        enc = self.encoder
        hidden = contextual(enc, tokens, valid, colors)
        weights = enc.pool(hidden).squeeze(-1).float()
        weights = weights.masked_fill(~valid, float("-inf")).softmax(1)
        pooled = (weights[..., None] * hidden.float()).sum(1)
        games = F.normalize(enc.output(pooled), dim=-1)
        moves = F.normalize(self.head(hidden.float()), dim=-1)
        return games, moves


def match_loss(
    model: TokenModel, moves: torch.Tensor, valid: torch.Tensor, players: int
) -> torch.Tensor:
    """Each player's first game is matched against every player's other three."""
    moves = moves.reshape(players, 4, 160, -1)
    valid = valid.reshape(players, 4, 160)
    query, query_valid = moves[:, 0], valid[:, 0]
    refs = moves[:, 1:].reshape(players, 480, -1)
    ref_valid = valid[:, 1:].reshape(players, 480)
    sims = torch.einsum("qid,pjd->qipj", query, refs)
    sims = sims.masked_fill(~ref_valid[None, None], -4.0)
    best = sims.amax(-1)  # [P,160,P]
    weight = query_valid.float()
    scores = (best * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp(
        min=1
    )
    target = torch.arange(players, device=moves.device)
    return F.cross_entropy(scores * model.log_scale.exp(), target)


def embed(model: TokenModel, data: dict, scaling: tuple, rows: np.ndarray) -> tuple:
    """Returns game vectors, half-precision move vectors and validity."""
    model.eval()
    games, moves, masks = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), 512):
            tokens, valid, colors = train.batch_tensors(
                data, scaling, rows[start : start + 512]
            )
            vectors, tokens_out = model(tokens, valid, colors)
            games.append(vectors.float())
            moves.append(tokens_out.half())
            masks.append(valid)
    return torch.cat(games), torch.cat(moves), torch.cat(masks)


def move_tokens(
    encoder: train.StyleEncoder,
    data: dict,
    scaling: tuple,
    rows: np.ndarray,
    kind: str = "hidden",
) -> tuple:
    """Training-free move vectors [N,160,D] (half), validity and game vectors.

    kind "hidden" takes the encoder's contextual hidden states (256-D);
    "raw" takes the standardized trunk activations (384-D).
    """
    outputs, masks, games = [], [], []
    encoder.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), 512):
            tokens, valid, colors = train.batch_tensors(
                data, scaling, rows[start : start + 512]
            )
            if kind == "raw":
                hidden = tokens[..., :384]
            else:
                hidden = contextual(encoder, tokens, valid, colors)
            outputs.append(F.normalize(hidden.float(), dim=-1).half())
            masks.append(valid)
            games.append(encoder(tokens, valid, colors).float())
    return torch.cat(outputs), torch.cat(masks), torch.cat(games)


def maxsim(tokens: torch.Tensor, valid: torch.Tensor, step: int = 8) -> torch.Tensor:
    """tokens [600,15,160,D] -> per-query-game scores [600,5,600].

    Every valid query move takes its best cosine over all moves of a
    candidate's 10 reference games; the maxima are averaged over moves.
    """
    width = tokens.shape[-1]
    flat = tokens[:, :10].reshape(-1, width)
    penalty = torch.where(valid[:, :10].reshape(600, 1600), 0.0, -4.0).half()[
        None, None
    ]
    queries = tokens[:, 10:].reshape(3000, 160, width)
    query_valid = valid[:, 10:].reshape(3000, 160)
    out = torch.zeros(3000, 600, device=DEVICE)
    for start in range(0, 3000, step):
        block = queries[start : start + step]
        sims = (block.reshape(-1, width) @ flat.T).reshape(len(block), 160, 600, 1600)
        best = (sims + penalty).amax(-1).float()
        weight = query_valid[start : start + step].float()
        out[start : start + step] = (best * weight[..., None]).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp(min=1)
    return out.reshape(600, 5, 600)


def standardize(scores: torch.Tensor) -> torch.Tensor:
    """Standardizes every row (question) of a score matrix."""
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)


def csls(scores: torch.Tensor) -> torch.Tensor:
    """Row-standardizes, then subtracts each row's and column's top-5 mean."""
    z = standardize(scores)
    row = z.topk(5, 1).values.mean(1, keepdim=True)
    return 2 * z - row - z.topk(5, 0).values.mean(0, keepdim=True)


def utility(scores: torch.Tensor) -> float:
    """Contest utility of a square score matrix with truth on the diagonal."""
    own = scores.diag()
    ranks = (scores > own[:, None]).sum(1)
    return torch.where(ranks < 5, torch.exp(-ranks.float()), 0.0).mean().item()


def cosine_scores(games: torch.Tensor, count: int) -> torch.Tensor:
    """Standardized centroid cosine of a [600,15,D] cohort with `count` queries."""
    centroids = F.normalize(games[:, :10].mean(1), dim=-1)
    return standardize(
        F.normalize(games[:, 10 : 10 + count].mean(1), dim=-1) @ centroids.T
    )


def fusion_table(
    games: torch.Tensor, per_game: torch.Tensor, weights: tuple
) -> dict:
    """Utility of cosine + weight x matching per weight; inf is matching alone.

    Each value lists the mean over query counts, then the four counts.
    """
    table = {}
    for weight in weights:
        values = []
        for count in COUNTS:
            extra = standardize(per_game[:, :count].mean(1))
            if weight != float("inf"):
                extra = cosine_scores(games, count) + weight * extra
            values.append(utility(extra))
        table[str(weight)] = [float(np.mean(values))] + values
    return table


def cohort(
    model: TokenModel, data: dict, scaling: tuple, rows: np.ndarray, save: str = ""
) -> dict:
    """Scores a [600,15] cohort: game vectors, move matching and their fusions."""
    games, moves, valid = embed(model, data, scaling, np.asarray(rows).ravel())
    games = games.reshape(600, 15, -1)
    per_game = maxsim(moves.reshape(600, 15, 160, -1), valid.reshape(600, 15, 160))
    if save:
        torch.save({"games": games.cpu(), "per_game": per_game.cpu()}, save)
    table = fusion_table(games, per_game, WEIGHTS + (float("inf"),))
    report: dict = {key: round(values[0], 4) for key, values in table.items()}
    report["by_count_best"] = table["inf"][1:]
    return report


def main() -> None:  # pylint: disable=too-many-locals
    """Fine-tunes, selects the epoch on validation and saves the best state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--players", type=int, default=64)
    parser.add_argument("--name", default="style_v5_match")
    parser.add_argument("--base", default="style_v4")
    parser.add_argument("--limit-steps", type=int, default=0)
    arguments = parser.parse_args()
    torch.manual_seed(20261114)
    encoder, scaling, _, checkpoint = apply.load_model(arguments.base)
    data = train.load_corpus(0, True)
    if len(data["groups"]) != checkpoint["classes"]:
        raise ValueError("Training players differ from the base checkpoint")
    model = TokenModel(encoder).to(DEVICE)
    steps = arguments.limit_steps or len(data["train_rows"]) // (arguments.players * 4)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": list(model.head.parameters()) + [model.log_scale], "lr": 1e-3},
        ],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, [1e-4, 1e-3], total_steps=steps * arguments.epochs, pct_start=0.1
    )
    generator = np.random.default_rng(20261114)
    before = cohort(model, data, scaling, data["validation"])
    print(json.dumps({"steps_per_epoch": steps, "before": before}), flush=True)
    best = -1.0
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        totals = np.zeros(3)
        for _ in range(steps):
            rows, labels = train.sample_batch(data, generator, arguments.players)
            tokens, valid, colors = train.batch_tensors(data, scaling, rows)
            dropped = torch.rand(len(rows), device=DEVICE) < 0.3
            tokens[dropped, :, train.LAYOUT.trunk :] = 0
            keep = torch.rand(valid.shape, device=DEVICE) > 0.15
            valid = valid & (keep | (valid.sum(1, keepdim=True) <= 8))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                games, moves = model(tokens, valid, colors)
            arcface, contrast = train.losses(
                model.encoder, games.float(), torch.from_numpy(labels).to(DEVICE)
            )
            match = match_loss(model, moves.float(), valid, arguments.players)
            optimizer.zero_grad(set_to_none=True)
            (arcface + contrast + 2 * match).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals += (arcface.item(), contrast.item(), match.item())
        row = {
            "epoch": epoch,
            "arcface": totals[0] / steps,
            "contrast": totals[1] / steps,
            "match": totals[2] / steps,
            "scale": model.log_scale.exp().item(),
            "validation": cohort(model, data, scaling, data["validation"]),
        }
        score = max(row["validation"][str(weight)] for weight in WEIGHTS)
        if score > best:
            best = score
            torch.save(
                {
                    "state": model.state_dict(),
                    "validation": row["validation"],
                    "epoch": epoch,
                },
                RERANK / f"{arguments.name}.pt",
            )
        print(json.dumps(row), flush=True)
    print(json.dumps({"best_validation": best}), flush=True)


if __name__ == "__main__":
    main()
