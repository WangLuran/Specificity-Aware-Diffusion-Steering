#!/usr/bin/env python3
"""Aggregate binder-level specificity results by CFG setting."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows = read_rows(run_root / "specificity_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["guidance_scale_tag"]].append(row)

    output: list[dict] = []
    for scale, group in grouped.items():
        robust = values(group, "robust_wanted_minus_unwanted_gap")
        mean_gap = values(group, "mean_wanted_minus_unwanted_gap")
        max_gap = values(
            group,
            "best_sample_wanted_minus_worst_offtarget_gap",
        )
        best = group[int(np.argmax(robust))]
        output.append(
            {
                "guidance_scale_tag": scale,
                "binder_count": len(group),
                "robust_gap_mean": float(np.mean(robust)),
                "robust_gap_median": float(np.median(robust)),
                "robust_gap_max_for_lead_selection": float(np.max(robust)),
                "robust_gap_positive_fraction": float(np.mean(robust > 0)),
                "mean_gap_mean": float(np.mean(mean_gap)),
                "mean_gap_median": float(np.median(mean_gap)),
                "mean_gap_max_for_lead_selection": float(np.max(mean_gap)),
                "best_sample_gap_mean": float(np.mean(max_gap)),
                "best_sample_gap_max_for_lead_selection": float(
                    np.max(max_gap)
                ),
                "wanted_pair_iptm_mean_over_binders": float(
                    np.mean(values(group, "wanted_binder_peptide_iptm_mean"))
                ),
                "unwanted_pair_iptm_mean_over_binders": float(
                    np.mean(
                        values(group, "unwanted_binder_peptide_iptm_mean")
                    )
                ),
                "mutation_position": group[0].get("mutation_position", "4"),
                "wanted_mutation_contact_fraction_mean": float(
                    np.mean(
                        values(
                            group,
                            "wanted_mutation_contact_model_fraction",
                        )
                    )
                ),
                "unwanted_mutation_contact_fraction_mean": float(
                    np.mean(
                        values(
                            group,
                            "unwanted_mutation_contact_model_fraction",
                        )
                    )
                ),
                "best_binder_id_by_robust_gap": best["binder_id"],
                "best_binder_sequence": best["binder_sequence"],
            }
        )
    output.sort(key=lambda row: row["guidance_scale_tag"])
    destination = run_root / "specificity_by_setting.csv"
    write_rows(destination, output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
