#!/usr/bin/env python3
"""Recompute the matched pMHC binder table from per-binder CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    ("boltzgen_target_a_only", "BoltzGen (target A only)"),
    ("fixed_cfg", "Fixed CFG"),
    ("dng", "DNG"),
    ("ours_prop2", "Ours (Proposition 2)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--target-a-root", type=Path)
    parser.add_argument("--cfg-root", type=Path)
    parser.add_argument("--dng-root", type=Path)
    parser.add_argument("--ours-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect(root: Path) -> list[dict[str, str]]:
    if root.is_file():
        rows = read_csv(root)
    else:
        paths = sorted(root.glob("shard*/specificity_summary.csv"))
        if len(paths) != 8:
            raise RuntimeError(f"{root}: expected 8 shard summaries, found {len(paths)}")
        rows = [row for path in paths for row in read_csv(path)]
    if len(rows) != 64:
        raise RuntimeError(f"{root}: expected 64 binders, found {len(rows)}")
    required = {
        "binder_id",
        "wanted_binder_peptide_iptm_mean",
        "unwanted_binder_peptide_iptm_mean",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"{root}: missing columns {sorted(missing)}")
    return rows


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    gaps = [
        float(row["wanted_binder_peptide_iptm_mean"])
        - float(row["unwanted_binder_peptide_iptm_mean"])
        for row in rows
    ]
    q25, _, q75 = statistics.quantiles(gaps, n=4, method="inclusive")
    return {
        "n": len(gaps),
        "mean": statistics.fmean(gaps),
        "median": statistics.median(gaps),
        "q25": q25,
        "q75": q75,
        "positive_fraction": statistics.fmean(gap > 0 for gap in gaps),
        "max": max(gaps),
        "population_std": statistics.pstdev(gaps),
    }


def sources(args: argparse.Namespace) -> dict[str, Path]:
    if args.reference:
        root = Path(__file__).resolve().parent / "reference"
        return {
            "boltzgen_target_a_only": root / "boltzgen_target_a_only.csv",
            "fixed_cfg": root / "fixed_cfg.csv",
            "dng": root / "dng.csv",
            "ours_prop2": root / "ours_prop2.csv",
        }
    supplied = {
        "boltzgen_target_a_only": args.target_a_root,
        "fixed_cfg": args.cfg_root,
        "dng": args.dng_root,
        "ours_prop2": args.ours_root,
    }
    if any(path is None for path in supplied.values()):
        raise SystemExit(
            "provide all four run roots, or use --reference"
        )
    return supplied


def main() -> int:
    args = parse_args()
    paths = sources(args)
    payload = {
        "metric": "mean wanted binder--peptide iPTM minus mean unwanted binder--peptide iPTM",
        "design": {
            "seed_base_start": 20261230,
            "seed_base_increment": 1000,
            "shards": 8,
            "backbones_per_shard": 8,
            "inverse_folds_per_backbone": 1,
            "boltz2_predictions_per_condition": 3,
        },
        "statistics": {key: metrics(collect(paths[key])) for key, _ in METHODS},
    }
    print("Method & N & Mean & Median & Q75 & $\\Delta>0$ \\\\")
    for key, label in METHODS:
        row = payload["statistics"][key]
        print(
            f"{label} & {row['n']} & {row['mean']:.4f} & {row['median']:.4f}"
            f" & {row['q75']:.4f} & {row['positive_fraction']:.4f} \\\\"
        )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "binder_table.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

