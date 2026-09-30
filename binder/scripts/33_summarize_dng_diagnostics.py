#!/usr/bin/env python3
"""Summarize BoltzGen DNG posterior and dynamic-guidance trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_max": max(values),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    paths = sorted(
        (run_root / "generation").glob(
            "lambda_*/trial_*/negative_guidance_diagnostics/sample_*.json"
        )
    )
    if not paths:
        raise FileNotFoundError("No DNG diagnostic JSON files found.")

    sample_rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("guidance_mode") != "dng":
            raise ValueError(f"Expected DNG diagnostics in {path}.")
        dng = payload["dng"]
        steps = payload["steps"]
        posterior_before = [
            float(step["dng_posterior_before"]) for step in steps
        ]
        posterior_after = [
            float(step["dng_posterior_after"]) for step in steps
        ]
        requested = [
            float(step["dng_dynamic_scale_requested"]) for step in steps
        ]
        effective = [
            float(step["effective_guidance_scale"]) for step in steps
        ]
        p_min = float(dng["p_min"])
        p_max = float(dng["p_max"])
        lambda_tag = next(
            part.removeprefix("lambda_")
            for part in path.parts
            if part.startswith("lambda_")
        )
        trial = next(
            part.removeprefix("trial_")
            for part in path.parts
            if part.startswith("trial_")
        )
        sample_rows.append(
            {
                "lambda0_tag": lambda_tag,
                "trial": trial,
                "sample": path.stem.removeprefix("sample_"),
                "step_count": len(steps),
                "lambda0": float(dng["lambda0"]),
                "prior": float(dng["prior"]),
                "temperature": float(dng["temperature"]),
                "offset": float(dng["offset"]),
                "posterior_initial": posterior_before[0],
                "posterior_final": posterior_after[-1],
                **summarize(
                    posterior_before + [posterior_after[-1]],
                    "posterior",
                ),
                **summarize(requested, "requested_scale"),
                **summarize(effective, "effective_scale"),
                "posterior_pmin_fraction": statistics.mean(
                    value <= p_min * (1 + 1e-6) for value in posterior_after
                ),
                "posterior_pmax_fraction": statistics.mean(
                    value >= p_max * (1 - 1e-6) for value in posterior_after
                ),
                "scale_clipped_fraction": statistics.mean(
                    abs(a - b) > 1e-8
                    for a, b in zip(requested, effective, strict=True)
                ),
                "diagnostics_path": str(path),
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["lambda0_tag"]].append(row)
    setting_rows: list[dict] = []
    for tag, group in sorted(grouped.items()):
        setting_rows.append(
            {
                "lambda0_tag": tag,
                "sample_count": len(group),
                "lambda0": group[0]["lambda0"],
                "posterior_final_mean": statistics.mean(
                    row["posterior_final"] for row in group
                ),
                "posterior_final_median": statistics.median(
                    row["posterior_final"] for row in group
                ),
                "posterior_trajectory_mean": statistics.mean(
                    row["posterior_mean"] for row in group
                ),
                "requested_scale_mean": statistics.mean(
                    row["requested_scale_mean"] for row in group
                ),
                "requested_scale_median": statistics.median(
                    row["requested_scale_median"] for row in group
                ),
                "requested_scale_max": max(
                    row["requested_scale_max"] for row in group
                ),
                "effective_scale_mean": statistics.mean(
                    row["effective_scale_mean"] for row in group
                ),
                "effective_scale_max": max(
                    row["effective_scale_max"] for row in group
                ),
                "posterior_pmin_fraction_mean": statistics.mean(
                    row["posterior_pmin_fraction"] for row in group
                ),
                "posterior_pmax_fraction_mean": statistics.mean(
                    row["posterior_pmax_fraction"] for row in group
                ),
                "scale_clipped_fraction_mean": statistics.mean(
                    row["scale_clipped_fraction"] for row in group
                ),
            }
        )

    sample_path = run_root / "dng_diagnostics_by_sample.csv"
    setting_path = run_root / "dng_diagnostics_by_setting.csv"
    write_csv(sample_path, sample_rows)
    write_csv(setting_path, setting_rows)
    print(sample_path)
    print(setting_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
