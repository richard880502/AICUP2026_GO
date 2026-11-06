"""Validates submission_rank.csv against the official format, and diffs two submissions.

Official format (aicup_tutorial_review/README.md, 競賽提交): one file covering the public and the
private questions (400 + 400 = 800 rows), columns question_id, pred_rank; the pub_ / priv_ prefixes
copied verbatim; a missing question makes the whole submission invalid.

Checks (FAIL makes the file unsubmittable):
  FAIL  unreadable / not UTF-8, header not exactly question_id,pred_rank, empty or NaN cells,
        pred_rank not one of D C B A 1D 2D 3D 4D 5D 6D (exact spelling), duplicated question_id,
        test question missing, question_id not in the test files, row count != test questions
  WARN  UTF-8 BOM, surrounding whitespace, rows not in the order of the test files, CRLF endings
Diff (--diff OTHER): how many questions change, from -> to counts, 1-step vs larger moves, mean
signed shift, and both predicted-rank distributions.

Usage (from the project root):
  c1/.venv/bin/python scripts/check_submission.py SUB.csv --tests-dir DIR [--diff OTHER.csv]
  c1/.venv/bin/python scripts/check_submission.py SUB.csv --expect 800      (no test files at hand)
Exit code 1 if any check FAILs, else 0.
"""
import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
HEADER = ["question_id", "pred_rank"]


class Checks:
    """PASS / WARN / FAIL lines."""

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        """One line."""
        self.items.append({"status": status, "check": name, "detail": detail})
        print(f"{status:4s}  {name:22s} {detail}")

    def failed(self) -> bool:
        """Any FAIL."""
        return any(item["status"] == "FAIL" for item in self.items)


def test_ids(tests_dir: pathlib.Path) -> list[str]:
    """Question ids of the official files, public first, in file order."""
    ids: list[str] = []
    for split in ("public", "private"):
        frame = pd.read_csv(tests_dir / f"rank_prediction_test_{split}.csv",
                            usecols=["question_id"], dtype=str, keep_default_na=False)
        ids.extend(frame["question_id"])
    return ids


def read_raw(chk: Checks, path: pathlib.Path) -> pd.DataFrame | None:
    """Encoding, line endings, header; the table as text."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        chk.add("FAIL", "encoding", f"not UTF-8: {err}")
        return None
    chk.add("WARN" if text.startswith("﻿") else "PASS", "encoding",
            "UTF-8 with BOM (strip it to be safe)" if text.startswith("﻿") else "UTF-8")
    if "\r\n" in text:
        chk.add("WARN", "line_endings", "CRLF line endings")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    header = list(frame.columns)
    chk.add("PASS" if header == HEADER else "FAIL", "header",
            f"{header}" + ("" if header == HEADER else f" (required {HEADER})"))
    return frame if header == HEADER else None


def check_rows(chk: Checks, sub: pd.DataFrame, ids: list[str] | None, expect: int) -> None:
    """Cells, labels, ids, count, order."""
    stripped = sub.apply(lambda col: col.str.strip())
    spaced = int((stripped != sub).to_numpy().sum())
    chk.add("WARN" if spaced else "PASS", "whitespace", f"{spaced} cells with surrounding spaces")
    empty = int((stripped == "").to_numpy().sum())
    chk.add("FAIL" if empty else "PASS", "empty_cells", f"{empty} empty / NaN cells")
    bad = sorted(set(stripped["pred_rank"]) - set(LEVELS))
    chk.add("FAIL" if bad else "PASS", "labels",
            f"invalid pred_rank values {bad[:8]}" if bad else "all in D..6D")
    dup = int(stripped["question_id"].duplicated().sum())
    chk.add("FAIL" if dup else "PASS", "duplicates", f"{dup} duplicated question_id")
    want = len(ids) if ids is not None else expect
    chk.add("PASS" if len(sub) == want else "FAIL", "row_count", f"{len(sub)} rows, need {want}")
    prefixed = int(stripped["question_id"].str.match(r"^(pub|priv)_").sum())
    chk.add("PASS" if prefixed == len(sub) else "WARN", "id_prefix",
            f"{len(sub) - prefixed} ids without pub_ / priv_ prefix")
    if ids is None:
        return
    have = set(stripped["question_id"])
    missing = [q for q in ids if q not in have]
    unknown = sorted(have - set(ids))
    chk.add("FAIL" if missing else "PASS", "missing_questions",
            f"{len(missing)} test questions without a prediction {missing[:5]}")
    chk.add("FAIL" if unknown else "PASS", "unknown_questions",
            f"{len(unknown)} ids not in the test files {unknown[:5]}")
    same_order = list(stripped["question_id"]) == ids
    chk.add("PASS" if same_order else "WARN", "row_order",
            "public then private, as in the test files" if same_order
            else "rows not in test-file order (the spec does not require it)")


def distribution(sub: pd.DataFrame) -> dict[str, int]:
    """Predicted-rank counts."""
    counts = sub["pred_rank"].str.strip().value_counts()
    return {r: int(counts.get(r, 0)) for r in LEVELS}


def diff(sub: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    """Changes between two submissions (aligned on question_id)."""
    a = sub.assign(q=sub["question_id"].str.strip()).set_index("q")["pred_rank"].str.strip()
    b = other.assign(q=other["question_id"].str.strip()).set_index("q")["pred_rank"].str.strip()
    common = a.index.intersection(b.index)
    level = {r: i for i, r in enumerate(LEVELS)}
    ia = a[common].map(level)
    ib = b[common].map(level)
    ok = ia.notna() & ib.notna()
    step = (ib[ok] - ia[ok]).astype(int)
    moved = step != 0
    pairs = pd.Series(list(zip(a[common][ok][moved], b[common][ok][moved]))).value_counts()
    return {"common": len(common), "only_in_first": len(a.index.difference(b.index)),
            "only_in_second": len(b.index.difference(a.index)),
            "changed": int(moved.sum()), "changed_share": float(moved.mean()) if len(step) else 0.0,
            "one_step": int((step.abs() == 1).sum()), "two_plus": int((step.abs() >= 2).sum()),
            "mean_signed_shift": float(step.mean()) if len(step) else 0.0,
            "changes_from_to": {f"{x}->{y}": int(n) for (x, y), n in pairs.items()},
            "changed_by_first_rank": {r: int(((a[common][ok] == r) & moved).sum())
                                      for r in LEVELS},
            "first": distribution(sub), "second": distribution(other)}


def print_diff(result: dict[str, Any], first: pathlib.Path, second: pathlib.Path) -> None:
    """Human-readable diff."""
    print(f"\ndiff {first.name} -> {second.name}: {result['changed']} of {result['common']} "
          f"questions change ({100 * result['changed_share']:.1f}%), {result['one_step']} by one "
          f"rank, {result['two_plus']} by two or more, mean signed shift "
          f"{result['mean_signed_shift']:+.3f} ranks")
    if result["only_in_first"] or result["only_in_second"]:
        print(f"  ids only in first {result['only_in_first']}, only in second "
              f"{result['only_in_second']}")
    print("  changed, by first rank:", result["changed_by_first_rank"])
    top = sorted(result["changes_from_to"].items(), key=lambda kv: -kv[1])[:12]
    print("  most common changes:", dict(top))
    print("  " + " ".join(f"{r:>4s}" for r in ("rank",) + LEVELS))
    for name, label in (("first", "1st"), ("second", "2nd")):
        print(f"  {label:>4s} " + " ".join(f"{result[name][r]:4d}" for r in LEVELS))


def parse_args() -> argparse.Namespace:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("submission", type=pathlib.Path)
    ap.add_argument("--tests-dir", type=pathlib.Path, default=None,
                    help="official test files (exact id set and order)")
    ap.add_argument("--expect", type=int, default=800, help="row count without --tests-dir")
    ap.add_argument("--diff", type=pathlib.Path, default=None, help="second submission to compare")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    return ap.parse_args()


def main() -> None:
    """Validate (and diff)."""
    args = parse_args()
    chk = Checks()
    report: dict[str, Any] = {"submission": str(args.submission)}
    sub = read_raw(chk, args.submission)
    if sub is not None:
        ids = test_ids(args.tests_dir) if args.tests_dir is not None else None
        check_rows(chk, sub, ids, args.expect)
        report["distribution"] = distribution(sub)
        print("predicted ranks:", report["distribution"])
        if args.diff is not None:
            other = pd.read_csv(args.diff, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            report["diff"] = diff(sub, other)
            print_diff(report["diff"], args.submission, args.diff)
    verdict = "FAIL" if chk.failed() else "PASS"
    print(f"\nsubmission: {verdict}"
          + ("" if verdict == "PASS" else " - do NOT upload this file"))
    if args.json is not None:
        report.update(verdict=verdict, checks=chk.items)
        args.json.write_text(json.dumps(report, indent=2, default=lambda o: o.item()
                                        if isinstance(o, np.generic) else str(o)))
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
