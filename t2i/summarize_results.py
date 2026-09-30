#!/usr/bin/env python3
"""Aggregate the fixed ten-condition Proposition-2 T2I experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONDITIONS = [
    (prompt, kind, f"p{prompt}_{kind}")
    for prompt in range(1, 6)
    for kind in ("related", "unrelated")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--condition-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def rows_from_run(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for prompt, kind, condition in CONDITIONS:
        path = run_root / condition / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing condition summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"condition is not complete: {condition}")
        rows.append(
            {
                "condition": condition,
                "prompt_index": str(prompt),
                "negative_kind": kind,
                "positive": str(payload["clip_positive_mean"]),
                "negative": str(payload["clip_used_negative_mean"]),
                "margin": str(payload["clip_margin_pos_minus_used_negative_mean"]),
                "resamples": str(payload["resamples"]),
                "unique_roots": str(payload["final_unique_roots"]),
            }
        )
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {(row["prompt_index"], row["negative_kind"]) for row in rows}
    expected = {(str(prompt), kind) for prompt, kind, _ in CONDITIONS}
    if len(rows) != 10 or observed != expected:
        raise RuntimeError("expected exactly five related and five unrelated rows")
    return rows


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    groups: dict[str, dict[str, float]] = {}
    for kind in ("related", "unrelated"):
        selected = [row for row in rows if row["negative_kind"] == kind]
        groups[kind] = {
            "clip_positive": mean(selected, "positive"),
            "clip_negative": mean(selected, "negative"),
            "margin": mean(selected, "margin"),
        }
    return {
        "method": "Ours (Proposition 2)",
        "conditions": len(rows),
        **groups,
        "overall_margin": mean(rows, "margin"),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = rows_from_run(args.run_root) if args.run_root else read_rows(args.condition_csv)
    table = summarize(rows)
    if args.output_dir or args.run_root:
        output = args.output_dir or args.run_root
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "condition_results.csv", rows)
        (output / "table_summary.json").write_text(
            json.dumps(table, indent=2) + "\n", encoding="utf-8"
        )
    related = table["related"]
    unrelated = table["unrelated"]
    print(
        "Ours (Proposition 2)"
        f" & {related['clip_positive']:.4f} & {related['clip_negative']:.4f}"
        f" & {related['margin']:.4f} & {unrelated['clip_positive']:.4f}"
        f" & {unrelated['clip_negative']:.4f} & {unrelated['margin']:.4f}"
        f" & {table['overall_margin']:.4f} \\\\"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

