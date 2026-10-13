"""Trains a player-level rank model on bags of frozen game embeddings.

Each optimization example contains 5-20 games from one player. Player groups
are disjoint between training, inner validation, the fixed validation cohort,
and the final holdout. The final holdout is not read by this script.

Example:
    c1/.venv/bin/python -m experiments.hierarchical_move.train --device cuda:1
"""

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn

import rank_lib


@dataclasses.dataclass(frozen=True)
class PlayerData:
    """Stores game rows and one rank label for each eligible player.

    Attributes:
        rows: Mapping from player ID to embedding rows.
        label: Mapping from player ID to integer rank.
        by_rank: Eligible player arrays grouped by rank.
    """

    rows: dict[str, np.ndarray]
    label: dict[str, int]
    by_rank: tuple[np.ndarray, ...]


class BagRanker(torch.nn.Module):
    """Permutation-invariant Transformer over multiple game embeddings."""

    def __init__(
        self,
        input_dimensions: int,
        model_dimensions: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(input_dimensions)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, model_dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        block = torch.nn.TransformerEncoderLayer(
            model_dimensions,
            heads,
            model_dimensions * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            block,
            layers,
            enable_nested_tensor=False,
        )
        self.pool = torch.nn.Linear(model_dimensions, 1)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(model_dimensions * 3),
            torch.nn.Linear(model_dimensions * 3, model_dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dimensions, len(rank_lib.LEVELS)),
        )

    def forward(self, games: torch.Tensor) -> torch.Tensor:
        """Returns rank logits for B x K x D game embeddings."""
        hidden = self.encoder(self.project(self.input_norm(games)))
        weights = self.pool(hidden).squeeze(-1).softmax(dim=1)
        attention = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        mean = hidden.mean(dim=1)
        standard_deviation = hidden.std(dim=1, correction=0)
        return self.head(torch.cat((attention, mean, standard_deviation), dim=1))


class DeepSetsBagRanker(torch.nn.Module):
    """Permutation-invariant per-game network without cross-game attention."""

    def __init__(
        self,
        input_dimensions: int,
        model_dimensions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(input_dimensions)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, model_dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dimensions, model_dimensions),
            torch.nn.GELU(),
        )
        self.pool = torch.nn.Linear(model_dimensions, 1)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(model_dimensions * 3),
            torch.nn.Linear(model_dimensions * 3, model_dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dimensions, len(rank_lib.LEVELS)),
        )

    def forward(self, games: torch.Tensor) -> torch.Tensor:
        """Returns rank logits for B x K x D game embeddings."""
        hidden = self.project(self.input_norm(games))
        weights = self.pool(hidden).squeeze(-1).softmax(dim=1)
        attention = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        mean = hidden.mean(dim=1)
        standard_deviation = hidden.std(dim=1, correction=0)
        return self.head(torch.cat((attention, mean, standard_deviation), dim=1))


def load_metadata(directory: pathlib.Path) -> tuple[np.ndarray, ...]:
    """Loads aligned player IDs, ranks, and game keys."""
    with np.load(directory / "metadata.npz", allow_pickle=True) as data:
        return (
            data["player_id"].astype(str),
            data["rank"].astype(np.int64),
            data["game_key"].astype(str),
        )


def make_player_data(
    players: np.ndarray,
    ranks: np.ndarray,
    excluded: set[str],
    minimum_games: int,
) -> PlayerData:
    """Builds rank-balanced player sampling tables."""
    rows: dict[str, list[int]] = {}
    for index, player in enumerate(players):
        if player not in excluded:
            rows.setdefault(player, []).append(index)
    filtered = {
        player: np.asarray(indices, dtype=np.int64)
        for player, indices in rows.items()
        if len(indices) >= minimum_games
    }
    labels = {
        player: int(ranks[indices[0]])
        for player, indices in filtered.items()
    }
    for player, indices in filtered.items():
        if not np.all(ranks[indices] == labels[player]):
            raise ValueError(f"Player {player!r} has inconsistent ranks")
    by_rank = tuple(
        np.asarray([
            player
            for player in sorted(filtered)
            if labels[player] == rank
        ])
        for rank in range(len(rank_lib.LEVELS))
    )
    return PlayerData(filtered, labels, by_rank)


def split_inner_players(
    data: PlayerData,
    players_per_rank: int,
    seed: int,
) -> tuple[PlayerData, PlayerData]:
    """Creates a stratified player-disjoint inner validation split."""
    rng = np.random.default_rng(seed)
    validation_players: set[str] = set()
    for rank_players in data.by_rank:
        chosen = rng.choice(
            rank_players,
            size=min(players_per_rank, len(rank_players)),
            replace=False,
        )
        validation_players.update(chosen.tolist())
    training_rows = {
        player: rows
        for player, rows in data.rows.items()
        if player not in validation_players
    }
    validation_rows = {
        player: rows
        for player, rows in data.rows.items()
        if player in validation_players
    }
    return (
        subset_player_data(data, training_rows),
        subset_player_data(data, validation_rows),
    )


def subset_player_data(
    source: PlayerData,
    rows: dict[str, np.ndarray],
) -> PlayerData:
    """Creates a PlayerData view containing selected players."""
    labels = {player: source.label[player] for player in rows}
    by_rank = tuple(
        np.asarray([
            player
            for player in sorted(rows)
            if labels[player] == rank
        ])
        for rank in range(len(rank_lib.LEVELS))
    )
    return PlayerData(rows, labels, by_rank)


def sample_training_batch(
    embeddings: np.ndarray,
    data: PlayerData,
    batch_size: int,
    games_per_bag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples a rank-balanced batch of player bags."""
    labels = rng.integers(0, len(rank_lib.LEVELS), size=batch_size)
    result = np.empty(
        (batch_size, games_per_bag, embeddings.shape[1]),
        dtype=np.float32,
    )
    for index, label in enumerate(labels):
        player = str(rng.choice(data.by_rank[int(label)]))
        rows = rng.choice(
            data.rows[player],
            size=games_per_bag,
            replace=False,
        )
        result[index] = embeddings[rows].astype(np.float32)
    return result, labels.astype(np.int64)


def fixed_bags(
    data: PlayerData,
    games_per_bag: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates deterministic evaluation row indices and labels."""
    rng = np.random.default_rng(seed + games_per_bag)
    rows = []
    labels = []
    for player in sorted(data.rows):
        for _ in range(repeats):
            rows.append(
                rng.choice(
                    data.rows[player],
                    size=games_per_bag,
                    replace=False,
                ),
            )
            labels.append(data.label[player])
    return np.asarray(rows), np.asarray(labels, dtype=np.int64)


def predict_bags(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predicts class probabilities for prepared bags."""
    model.eval()
    probabilities = []
    with torch.no_grad(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for start in range(0, len(rows), batch_size):
            games = torch.from_numpy(
                embeddings[rows[start:start + batch_size]].astype(np.float32),
            ).to(device)
            probabilities.append(model(games).softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def evaluate(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    data: PlayerData,
    device: torch.device,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluates fixed 5, 10, 15, and 20-game questions."""
    report: dict[str, Any] = {"by_games": {}}
    all_probabilities = []
    all_labels = []
    for games_per_bag in (5, 10, 15, 20):
        rows, labels = fixed_bags(
            data,
            games_per_bag,
            repeats,
            seed,
        )
        probabilities = predict_bags(
            model,
            embeddings,
            rows,
            device,
            512,
        )
        prediction = rank_lib.decode(probabilities)
        report["by_games"][str(games_per_bag)] = {
            "score": rank_lib.competition_score(prediction, labels),
            "top1": float((prediction == labels).mean()),
            "nll": float(
                -np.log(
                    np.clip(
                        probabilities[np.arange(len(labels)), labels],
                        1e-8,
                        1.0,
                    ),
                ).mean(),
            ),
        }
        all_probabilities.append(probabilities)
        all_labels.append(labels)
    combined_probabilities = np.concatenate(all_probabilities)
    combined_labels = np.concatenate(all_labels)
    report["score"] = rank_lib.competition_score(
        rank_lib.decode(combined_probabilities),
        combined_labels,
    )
    report["nll"] = float(
        -np.log(
            np.clip(
                combined_probabilities[
                    np.arange(len(combined_labels)),
                    combined_labels,
                ],
                1e-8,
                1.0,
            ),
        ).mean(),
    )
    return report


def training_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Combines class NLL with a small ordinal-distance penalty."""
    cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
    rank_axis = torch.arange(
        len(rank_lib.LEVELS),
        device=logits.device,
        dtype=logits.dtype,
    )
    expected_rank = (logits.softmax(dim=1) * rank_axis).sum(dim=1)
    ordinal = torch.nn.functional.smooth_l1_loss(
        expected_rank,
        labels.to(logits.dtype),
    )
    return cross_entropy + 0.05 * ordinal


def create_model(
    input_dimensions: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """Creates the configured permutation-invariant bag architecture."""
    architecture = getattr(args, "architecture", "transformer")
    if architecture == "transformer":
        return BagRanker(
            input_dimensions,
            args.dimensions,
            args.heads,
            args.layers,
            args.dropout,
        ).to(device)
    if architecture == "deepsets":
        return DeepSetsBagRanker(
            input_dimensions,
            args.dimensions,
            args.dropout,
        ).to(device)
    raise ValueError(f"Unknown bag architecture {architecture!r}")


def train_model(
    embeddings: np.ndarray,
    training: PlayerData,
    validation: PlayerData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """Trains with fresh random player bags and inner early stopping."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = create_model(embeddings.shape[1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    requested_steps = getattr(args, "steps_per_epoch", 0)
    steps_per_epoch = max(
        1,
        requested_steps or len(training.rows) // args.batch,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
    )
    history = []
    best_nll = float("inf")
    best_state = None
    epochs_without_improvement = 0
    patience = getattr(args, "patience", 0)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            games_per_bag = int(rng.integers(5, 21))
            games, labels = sample_training_batch(
                embeddings,
                training,
                args.batch,
                games_per_bag,
                rng,
            )
            game_tensor = torch.from_numpy(games).to(device)
            label_tensor = torch.from_numpy(labels).to(device)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = training_loss(model(game_tensor), label_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        validation_report = evaluate(
            model,
            embeddings,
            validation,
            device,
            2,
            args.seed,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_nll": float(validation_report["nll"]),
            "validation_score": float(validation_report["score"]),
        }
        history.append(row)
        if row["validation_nll"] < best_nll:
            best_nll = row["validation_nll"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {row['train_loss']:.4f} "
            f"inner nll {row['validation_nll']:.4f} "
            f"score {row['validation_score']:.4f} "
            f"elapsed {time.time() - started:.0f}s",
            flush=True,
        )
        if patience and epochs_without_improvement >= patience:
            print(
                f"early stop after {epoch + 1} epochs; "
                f"best inner nll {best_nll:.4f}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


def external_validation_data(
    players: np.ndarray,
    ranks: np.ndarray,
    game_keys: np.ndarray,
    validation_directory: pathlib.Path,
) -> PlayerData:
    """Builds PlayerData for the fixed external validation cohort."""
    validation = rank_lib.load_features(validation_directory)
    key_to_row = {key: index for index, key in enumerate(game_keys)}
    selected_rows = np.asarray([
        key_to_row[key]
        for key in validation["game_key"]
    ])
    validation_players = players[selected_rows]
    validation_ranks = ranks[selected_rows]
    rows = {
        player: selected_rows[validation_players == player]
        for player in np.unique(validation_players)
    }
    labels = {
        player: int(validation_ranks[validation_players == player][0])
        for player in rows
    }
    by_rank = tuple(
        np.asarray([
            player
            for player in sorted(rows)
            if labels[player] == rank
        ])
        for rank in range(len(rank_lib.LEVELS))
    )
    return PlayerData(rows, labels, by_rank)


def main() -> None:
    """Trains and evaluates the hierarchical move-bag model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "move_embeddings_h",
    )
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "feat" / "val",
    )
    parser.add_argument(
        "--holdout-players",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "holdout" / "players.txt",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--inner-players-per-rank", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    embeddings = np.load(
        args.embeddings / "embeddings.npy",
        mmap_mode="r",
    )
    players, ranks, game_keys = load_metadata(args.embeddings)
    validation = rank_lib.load_features(args.validation)
    excluded = set(validation["player_id"])
    excluded.update(args.holdout_players.read_text(encoding="utf-8").split())
    all_training = make_player_data(players, ranks, excluded, 20)
    training, inner_validation = split_inner_players(
        all_training,
        args.inner_players_per_rank,
        args.seed,
    )
    external_validation = external_validation_data(
        players,
        ranks,
        game_keys,
        args.validation,
    )
    print(
        f"players train={len(training.rows)} inner={len(inner_validation.rows)} "
        f"external={len(external_validation.rows)}",
        flush=True,
    )
    model, history = train_model(
        embeddings,
        training,
        inner_validation,
        args,
        device,
    )
    external_report = evaluate(
        model,
        embeddings,
        external_validation,
        device,
        10,
        args.seed + 10000,
    )
    checkpoint = {
        "state": model.state_dict(),
        "args": vars(args) | {
            "embeddings": str(args.embeddings),
            "validation": str(args.validation),
            "holdout_players": str(args.holdout_players),
            "output": str(args.output),
        },
    }
    torch.save(checkpoint, args.output / "model.pt")
    report = {
        "protocol": {
            "training_players": len(training.rows),
            "inner_validation_players": len(inner_validation.rows),
            "external_validation_players": len(external_validation.rows),
            "holdout_read": False,
            "minimum_games_per_training_player": 20,
        },
        "history": history,
        "external_validation": external_report,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(external_report, indent=2))


if __name__ == "__main__":
    main()
