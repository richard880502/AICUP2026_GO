"""Builds an untouched internal 'official-format' test set (our own public/private).

Picks --players players that (a) have at least --min-games games, (b) are not in the validation
cohort nor in any --exclude cohort, and writes
  <out>/players.txt          players that every retrained model must exclude
  <out>/tests/rank_prediction_test_{public,private}.csv  + labels.csv   (official layout)
Each player gives one question of k ~ U{5..20} games, like the official 5-20 game questions.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--players", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/holdout")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--exclude", type=pathlib.Path, nargs="*", default=[],
                    help="players.txt files of earlier cohorts to skip")
    return ap.parse_args()


def excluded_players(exclude: list[pathlib.Path]) -> set[str]:
    """Validation-cohort players plus every --exclude cohort."""
    banned: set[str] = set()
    for path in sorted((ROOT / "features/shards/val").glob("games_*.csv")):
        banned |= set(pd.read_csv(path, usecols=["player_id"])["player_id"])
    for path in exclude:
        banned |= set(path.read_text(encoding="utf-8").split())
    return banned


def pick_players(args: argparse.Namespace, rng: np.random.Generator,
                 banned: set[str]) -> tuple[list[str], pd.DataFrame]:
    """Rank-balanced random players (like the official sets) and all their games."""
    per_rank = args.players // len(LEVELS)
    chosen: list[str] = []
    frames = []
    for rank in LEVELS:
        data = pd.read_csv(ROOT / f"data/training/train_{rank}.csv")
        counts = data.groupby("player_id").size()
        eligible = np.array([p for p in counts[counts >= args.min_games].index if p not in banned],
                            dtype=object)
        pick = rng.choice(eligible, size=per_rank, replace=False)
        chosen += list(pick)
        frames.append(data[data["player_id"].isin(set(pick))])
        print(rank, "eligible", len(eligible), "picked", len(pick), flush=True)
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order], pd.concat(frames, ignore_index=True)


def build_questions(chosen: list[str], games: pd.DataFrame,
                    rng: np.random.Generator) -> tuple[dict[str, list[dict]], list[tuple]]:
    """One question per player: first half public, second half private."""
    labels: list[tuple] = []
    rows: dict[str, list[dict]] = {"public": [], "private": []}
    for q, pid in enumerate(chosen):
        split = "public" if q < len(chosen) // 2 else "private"
        shuffled = games[games["player_id"] == pid].sample(frac=1,
                                                           random_state=int(rng.integers(1 << 30)))
        picked = shuffled.head(int(rng.integers(5, 21)))
        qid = f"{'pub' if split == 'public' else 'priv'}_q_{len(rows[split]) + 1:04d}"
        row = {"question_id": qid, "num_games": len(picked)}
        for i in range(20):
            row[f"sgf_{i + 1}"] = picked.iloc[i]["sgf_content"] if i < len(picked) else ""
        for i in range(20):
            row[f"color_{i + 1}"] = picked.iloc[i]["color"] if i < len(picked) else ""
        rows[split].append(row)
        labels.append((qid, picked.iloc[0]["rank"]))
    return rows, labels


def main() -> None:
    """Writes players.txt and the official-layout test files of the new cohort."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    chosen, games = pick_players(args, rng, excluded_players(args.exclude))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "players.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    (args.out / "tests").mkdir(exist_ok=True)
    rows, labels = build_questions(chosen, games, rng)
    for split, split_rows in rows.items():
        pd.DataFrame(split_rows).to_csv(args.out / f"tests/rank_prediction_test_{split}.csv",
                                        index=False)
    pd.DataFrame(labels, columns=["question_id", "rank"]).to_csv(args.out / "tests/labels.csv",
                                                                 index=False)
    print("wrote", len(labels), "questions;", sum(len(r) for r in rows.values()),
          "total; games:", sum(int(x["num_games"]) for r in rows.values() for x in r))


if __name__ == "__main__":
    main()
