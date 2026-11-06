"""Explodes official rank_prediction_test_{public,private}.csv into shard CSVs.

Each question becomes one pseudo-player (player_id = question_id) so the same extractor and
the same question-level pooling code apply. Example:
    c1/.venv/bin/python scripts/test_to_games.py --tests-dir /path/to/tests

Every SGF goes through sgf_check.sanitize first, so one malformed game cannot break a shard of the
KataGo extraction: games that are fine (all known data) are written unchanged; repaired games are
written as a canonical SGF; unusable games (no SGF, board size not 19, colour not B / W) are left
out. Every repair or omission is listed in <out>/input_report.csv, and predict_test.py falls back
per question when games are missing.
"""
import argparse
import pathlib
import sys

import pandas as pd  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sgf_check  # noqa: E402  pylint: disable=wrong-import-position,import-error

COLOURS = {"B": "B", "W": "W", "BLACK": "B", "WHITE": "W"}


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--tests-dir", type=pathlib.Path, required=True)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "features/shards/test")
    return ap.parse_args()


def read_tests(tests_dir: pathlib.Path) -> pd.DataFrame:
    """Both test files, question ids kept as text (leading zeros survive)."""
    parts = [pd.read_csv(tests_dir / f"rank_prediction_test_{split}.csv",
                         dtype={"question_id": str}, keep_default_na=False, na_values=[""])
             for split in ("public", "private")]
    return pd.concat(parts, ignore_index=True)


def question_games(row: dict, report: list[dict]) -> list[dict]:
    """The usable games of one question as shard rows; problems go to report."""
    qid = row["question_id"]
    columns = sorted((int(c[4:]) for c in row if c.startswith("sgf_") and c[4:].isdigit()))
    present = [i for i in columns if isinstance(row.get(f"sgf_{i}"), str)
               and row[f"sgf_{i}"].strip()]
    listed = pd.to_numeric(row.get("num_games"), errors="coerce")
    if pd.isna(listed) or int(listed) != len(present):
        report.append({"question_id": qid, "game": "", "issue":
                       f"num_games {row.get('num_games')} but {len(present)} non-empty SGF columns",
                       "action": "used every non-empty SGF column"})
    rows = []
    for i in present:
        colour = COLOURS.get(str(row.get(f"color_{i}", "")).strip().upper())
        checked = sgf_check.sanitize(row[f"sgf_{i}"])
        if colour is None:
            report.append({"question_id": qid, "game": i,
                           "issue": f"colour {row.get(f'color_{i}')!r}", "action": "game dropped"})
            continue
        for issue in checked.issues:
            report.append({"question_id": qid, "game": i, "issue": issue,
                           "action": "game repaired" if checked.usable else "game dropped"})
        if checked.usable:
            rows.append({"game_key": f"{qid}#{i}", "player_id": qid, "rank": "",
                         "color": colour, "sgf_content": checked.sgf})
    if not rows:
        report.append({"question_id": qid, "game": "", "issue": "no usable game",
                       "action": "predict_test.py uses its prior fallback"})
    return rows


def main() -> None:
    """Writes the shards and the input report."""
    args = parse_args()
    tests = read_tests(args.tests_dir)
    report: list[dict] = []
    rows: list[dict] = []
    seen: set[str] = set()
    for record in tests.to_dict("records"):
        qid = record["question_id"]
        if qid in seen:
            report.append({"question_id": qid, "game": "", "issue": "duplicate question_id",
                           "action": "later row ignored"})
            continue
        seen.add(qid)
        rows.extend(question_games(record, report))
    games = pd.DataFrame(rows, columns=["game_key", "player_id", "rank", "color", "sgf_content"])
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(games), args.shard_size):
        games.iloc[i:i + args.shard_size].to_csv(args.out / f"games_{i // args.shard_size:04d}.csv",
                                                 index=False)
    pd.DataFrame(report, columns=["question_id", "game", "issue", "action"]).to_csv(
        args.out / "input_report.csv", index=False)
    print(len(games), "games,", games.player_id.nunique(), "questions;", len(report),
          "input problems (see", args.out / "input_report.csv", ")")


if __name__ == "__main__":
    main()
