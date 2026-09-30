#!/usr/bin/env python3
"""Summarize the two-field Proposition-2 PA-gate particle system."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_max": max(values),
    }


def main() -> int:
    args = parse_args()
    paths = sorted(
        args.run_root.resolve().glob(
            "generation/lambda_*/trial_*/"
            "negative_guidance_diagnostics/sample_*.json"
        )
    )
    if not paths:
        raise FileNotFoundError("No PA-gate SMC diagnostics found.")
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("guidance_mode") != "pa_gate_smc":
            raise ValueError(f"Expected pa_gate_smc diagnostics in {path}.")
        steps = payload["steps"]
        target = [float(step["target_scale_mean"]) for step in steps]
        proposal = [float(step["proposal_scale_mean"]) for step in steps]
        rho1 = [float(step.get("rho1", 0.0)) for step in steps]
        rho2 = [float(step.get("rho2", step.get("rho", 0.0))) for step in steps]
        ess = [
            float(step["incremental_ess_fraction"]) for step in steps
        ]
        cumulative_ess = [
            float(step.get("cumulative_ess_fraction", value))
            for step, value in zip(steps, ess)
        ]
        clip_fraction = [
            float(step.get("gate_logw_clip_fraction", 0.0))
            for step in steps
        ]
        total_clip_fraction = [
            float(step.get("logw_clip_fraction", 0.0)) for step in steps
        ]
        jvp_clip_fraction = [
            float(step.get("jvp_logw_clip_fraction", 0.0))
            for step in steps
        ]
        kernel_clip_fraction = [
            float(step.get("kernel_logw_clip_fraction", 0.0))
            for step in steps
        ]
        ancestor_clip_fraction = [
            float(step.get("ancestor_logw_clip_fraction", 0.0))
            for step in steps
        ]
        theta = [float(step["theta_mean"]) for step in steps]
        rows.append(
            {
                "particle_count": int(payload["particle_count"]),
                "step_count": len(steps),
                "resample_count": int(payload["resample_count"]),
                "final_unique_initial_roots": int(
                    payload["final_unique_initial_roots"]
                ),
                **describe(target, "target_scale"),
                **describe(proposal, "proposal_scale"),
                **describe(rho1, "rho1"),
                **describe(rho2, "rho2"),
                **describe(ess, "incremental_ess"),
                **describe(cumulative_ess, "cumulative_ess"),
                **describe(theta, "theta"),
                "ess_objective": steps[-1].get(
                    "ess_objective", "incremental"
                ),
                "rho2_selection_objective": steps[-1].get(
                    "rho2_selection_objective", "full_ess"
                ),
                "gate_logw_clip_fraction_mean": statistics.mean(
                    clip_fraction
                ),
                "logw_clip_fraction_mean": statistics.mean(
                    total_clip_fraction
                ),
                "jvp_logw_clip_fraction_mean": statistics.mean(
                    jvp_clip_fraction
                ),
                "kernel_logw_clip_fraction_mean": statistics.mean(
                    kernel_clip_fraction
                ),
                "ancestor_logw_clip_fraction_mean": statistics.mean(
                    ancestor_clip_fraction
                ),
                "rho2_nonzero_fraction": statistics.mean(
                    abs(value) > 1e-8 for value in rho2
                ),
                "resampled_step_fraction": statistics.mean(
                    bool(step["resampled"]) for step in steps
                ),
                "lhat_final_mean": float(steps[-1]["lhat_mean_after"]),
                "diagnostics_path": str(path),
            }
        )
    output = (
        args.run_root.resolve() / "pa_gate_smc_diagnostics_summary.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
