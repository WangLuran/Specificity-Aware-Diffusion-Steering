#!/usr/bin/env python3
"""Summarize HLA rotations used by frame-consistent CFG generation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--low-noise-start-progress", type=float, default=0.75)
    parser.add_argument("--max-low-noise-median-deg", type=float, default=0.2)
    return parser.parse_args()


def stats(values: list[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_min_deg": float(np.min(array)),
        f"{prefix}_median_deg": float(np.median(array)),
        f"{prefix}_mean_deg": float(np.mean(array)),
        f"{prefix}_max_deg": float(np.max(array)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
        raise SystemExit("No negative-guidance diagnostic JSON files found")

    sample_rows: list[dict] = []
    setting_steps: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        scale = next(
            part.removeprefix("lambda_")
            for part in path.parts
            if part.startswith("lambda_")
        )
        trial = next(
            part.removeprefix("trial_")
            for part in path.parts
            if part.startswith("trial_")
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_scale = float(
            payload.get(
                "base_guidance_scale",
                payload["steps"][0]["requested_guidance_scale"],
            )
        )
        all_angles = [
            float(step["rotation_angle_degrees"])
            for step in payload["steps"]
        ]
        active_angles = [
            float(step["rotation_angle_degrees"])
            for step in payload["steps"]
            if abs(float(step["requested_guidance_scale"])) > 0
        ]
        full_angles = [
            float(step["rotation_angle_degrees"])
            for step in payload["steps"]
            if base_scale != 0
            and np.isclose(
                float(step["requested_guidance_scale"]),
                base_scale,
                rtol=1e-6,
                atol=1e-8,
            )
        ]
        step_count = len(payload["steps"])
        low_noise_angles = [
            float(step["rotation_angle_degrees"])
            for step_index, step in enumerate(payload["steps"])
            if float(
                step.get(
                    "progress",
                    step_index / max(step_count - 1, 1),
                )
            )
            >= args.low_noise_start_progress
        ]
        sample_rows.append(
            {
                "guidance_scale_tag": scale,
                "trial": trial,
                "sample": int(re.search(r"(\d+)$", path.stem).group(1)),
                "base_guidance_scale": base_scale,
                "input_HLA_raw_rmsd_A": payload[
                    "input_HLA_raw_rmsd_angstrom"
                ],
                "input_HLA_fit_rmsd_A": payload[
                    "input_HLA_fit_rmsd_angstrom"
                ],
                **stats(all_angles, "all_steps"),
                **stats(active_angles, "active_steps"),
                **stats(full_angles, "full_guidance_steps"),
                **stats(low_noise_angles, "low_noise_final_quarter_steps"),
                "diagnostics_path": str(path),
            }
        )
        setting_steps[scale]["all"].extend(all_angles)
        setting_steps[scale]["active"].extend(active_angles)
        setting_steps[scale]["full"].extend(full_angles)
        setting_steps[scale]["low_noise"].extend(low_noise_angles)

    setting_rows = []
    for scale, groups in setting_steps.items():
        setting_rows.append(
            {
                "guidance_scale_tag": scale,
                "sample_count": sum(
                    row["guidance_scale_tag"] == scale
                    for row in sample_rows
                ),
                **stats(groups["all"], "all_steps"),
                **stats(groups["active"], "active_steps"),
                **stats(groups["full"], "full_guidance_steps"),
                **stats(
                    groups["low_noise"],
                    "low_noise_final_quarter_steps",
                ),
            }
        )
    setting_rows.sort(key=lambda row: row["guidance_scale_tag"])

    sample_path = run_root / "alignment_rotation_by_sample.csv"
    setting_path = run_root / "alignment_rotation_by_setting.csv"
    write_csv(sample_path, sample_rows)
    write_csv(setting_path, setting_rows)
    print(sample_path)
    print(setting_path)
    failed = [
        row
        for row in sample_rows
        if float(
            row["low_noise_final_quarter_steps_median_deg"]
        )
        > args.max_low_noise_median_deg
    ]
    if failed:
        descriptions = ", ".join(
            f"scale={row['guidance_scale_tag']} trial={row['trial']} "
            f"median={float(row['low_noise_final_quarter_steps_median_deg']):.4f}"
            for row in failed
        )
        raise SystemExit(
            "Low-noise alignment rotation quality control failed: "
            + descriptions
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
