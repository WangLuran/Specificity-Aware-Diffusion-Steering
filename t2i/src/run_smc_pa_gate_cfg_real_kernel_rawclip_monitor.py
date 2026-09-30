#!/usr/bin/env python3
"""pA-gate SMC monitor with adaptive gate tempering.

This variant removes the unconditional density factor from the target and uses
only the positive-prompt path as the reference:

    gamma_t(x) ∝ p_t^A(x) (1 - theta_t(x))^lambda_s.

The gate is still built from the raw A/B Gaussian-kernel log-ratio estimator:

    theta_t = sigmoid(log c_t + eta_t lhat_t),
    lhat_t ≈ log p_t^B - log p_t^A.

The finite practical diagnostic weight is

    log omega =
        lambda_s log[(1-theta_t(x))/(1-theta_tau(y))]
      + log L^{nu_A}(x|y)
      - log L^{a_r}(x|y)
      + log K^{b_r}(y|x)
      - log K^mu(y|x).

The local rho surrogate and a_r are the retained-order expansion of exactly
that finite Gaussian estimator.  We optimize a dimensionless rho ratio
r = rho_t / s_t.

This variant keeps the retained-order rho optimization from
run_smc_pa_gate_rho_monitor.py, but protects the gate-selection feedback loop by
tempering the gate log-weight used for the immediate resampling decision.  If
the full gate increment would push the active cumulative ESS below
--gate-temper-ess, only an alpha fraction of the gate increment is used for the
current selection weight and the remaining (1-alpha) fraction is carried as a
cumulative residual weight.  This is a bridge-style alternative to raw clipping:
the gate contribution is delayed rather than discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "HF_HOME",
    str(Path.home() / ".cache/huggingface"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prop2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_dng_eval import PROMPTS_FILE, ddpm_reverse_mean_and_variance  # noqa: E402
from run_smc_cfg_effective_monitor import (  # noqa: E402
    all_gather_cat,
    assemble_grid_from_saved_images,
    decode_and_save_images,
    distributed_barrier,
    global_ess_frac_from_logw,
    global_normalized_weights_from_logw,
    load_prompts,
    load_sd_pipeline,
    parse_devices,
    resolve_prompt,
    should_skip_smc_weight,
    wait_for_decode_markers,
    write_decode_done_marker,
    write_rows,
)
from run_smc_gaussian_monitor import (  # noqa: E402
    encode_text,
    finite_corr,
    gaussian_log_prob_isotropic,
    log1m_theta_from_lhat,
    model_output_to_epsilon,
    schedule_at_state as base_schedule_at_state,
    systematic_resample,
    theta_from_lhat,
)

HF_CACHE = Path(os.environ["HF_HOME"]) / "hub"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def schedule_at_state(k: int, n_steps: int, args: argparse.Namespace) -> tuple[float, float]:
    """Return the base schedules with an optional endpoint-preserving θ shift.

    Since θ = sigmoid(log(c_t) + eta_t * lhat), multiplying c_t by
    exp(delta_t) shifts the θ logit by delta_t.

    ``geometric`` interpolation is handled here (rather than by the shared
    legacy scheduler) because it makes log(c_t) exactly linear in diffusion
    progress.  This is the natural "uniform gate schedule" coordinate: c_t
    enters the gate only through log(c_t).

    The optional endpoint-preserving shift uses

        delta_t = shift * sin(pi * progress)^2,

    which is exactly zero at both endpoints and largest in the middle.  A
    negative shift lowers mid-trajectory θ and the local sensitivity
    gate_power * eta_t * θ without changing either endpoint target.
    """
    progress = min(1.0, max(0.0, float(k) / max(int(n_steps) - 1, 1)))

    # The shared scheduler predates geometric interpolation.  Give it a valid
    # placeholder mode for any component that we replace below.
    c_mode = str(getattr(args, "c_schedule", "exp"))
    eta_mode = str(getattr(args, "bs_schedule", "exp"))
    base_args = args
    if c_mode == "geometric" or eta_mode == "geometric":
        base_args = argparse.Namespace(**vars(args))
        if c_mode == "geometric":
            base_args.c_schedule = "linear"
        if eta_mode == "geometric":
            base_args.bs_schedule = "linear"
    c, eta = base_schedule_at_state(k, n_steps, base_args)

    if c_mode == "geometric":
        if float(args.c0) <= 0.0 or float(args.c1) <= 0.0:
            raise ValueError("--c-schedule geometric requires positive --c0 and --c1")
        c = math.exp(
            math.log(float(args.c1))
            + progress * (math.log(float(args.c0)) - math.log(float(args.c1)))
        )
    if eta_mode == "geometric":
        if float(args.bs0) <= 0.0 or float(args.bs1) <= 0.0:
            raise ValueError("--bs-schedule geometric requires positive --bs0 and --bs1")
        eta = math.exp(
            math.log(float(args.bs1))
            + progress * (math.log(float(args.bs0)) - math.log(float(args.bs1)))
        )

    shift = float(getattr(args, "theta_mid_logit_shift", 0.0))
    if shift != 0.0:
        delta = shift * math.sin(math.pi * progress) ** 2
        c = float(c) * math.exp(delta)
    return float(c), float(eta)


def gate_power_at_state(
    k: int, n_steps: int, args: argparse.Namespace
) -> float:
    """Fixed annealing path for the gate potential exponent.

    This is deliberately independent of ESS and particle values.  A
    nonconstant path changes only the intermediate Feynman--Kac targets; the
    final state always uses ``args.gate_power``.  Exact incremental weights
    account for the exponent change, so no gate residual is discarded.
    """
    terminal = float(args.gate_power)
    mode = str(getattr(args, "gate_power_schedule", "constant"))
    if mode == "constant":
        return terminal
    progress = min(
        1.0, max(0.0, float(k) / max(int(n_steps) - 1, 1))
    )
    end_frac = min(
        1.0,
        max(1e-8, float(getattr(args, "gate_power_end_frac", 1.0))),
    )
    progress = min(1.0, progress / end_frac)
    start_frac = min(
        1.0, max(0.0, float(getattr(args, "gate_power_start_frac", 0.0)))
    )
    gamma = max(
        1e-8, float(getattr(args, "gate_power_schedule_gamma", 1.0))
    )
    if mode == "linear":
        weight = progress
    elif mode == "power":
        weight = progress**gamma
    elif mode == "cosine":
        weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unknown --gate-power-schedule={mode!r}")
    return terminal * (start_frac + (1.0 - start_frac) * weight)


def linear_baseline_kappa_at_state(
    k: int, n_steps: int, args: argparse.Namespace
) -> float:
    """Return the fixed, ESS-independent centered-reference schedule.

    ``--linear-baseline-kappa`` remains the backward-compatible constant
    default.  ``run`` resolves omitted schedule endpoints to that value before
    workers are spawned, so this helper always sees explicit start/end values.
    A varying kappa changes only the reference/proposal factorization; exact
    incremental weights retain the same pure-PA target.
    """
    start = float(getattr(args, "kappa_start", args.linear_baseline_kappa))
    end = float(getattr(args, "kappa_end", args.linear_baseline_kappa))
    mode = str(getattr(args, "kappa_schedule", "constant"))
    if mode == "constant":
        return start
    progress = min(
        1.0, max(0.0, float(k) / max(int(n_steps) - 1, 1))
    )
    end_frac = min(
        1.0, max(1e-8, float(getattr(args, "kappa_end_frac", 1.0)))
    )
    progress = min(1.0, progress / end_frac)
    gamma = max(1e-8, float(getattr(args, "kappa_gamma", 1.0)))
    if mode == "linear":
        weight = progress
    elif mode == "power":
        weight = progress**gamma
    elif mode == "cosine":
        weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unknown --kappa-schedule={mode!r}")
    return start + (end - start) * weight


def centered_lhat_increment(
    lhat_new: torch.Tensor,
    lhat: torch.Tensor,
    *,
    kappa_new: float,
    kappa: float,
    lhat_temp: float,
) -> torch.Tensor:
    """Exact residual increment induced by the centered linear reference."""
    temp = float(lhat_temp)
    if float(kappa_new) == float(kappa):
        # Preserve the legacy constant-kappa evaluation order and avoid
        # subtracting two large kappa*lhat products.
        coeff = float(kappa) / max(abs(temp), 1e-30)
        return coeff * (lhat_new.float() - lhat.float())
    if abs(temp) <= 1e-30:
        raise ValueError("Centered lhat compensation requires nonzero lhat_temp")
    return (
        (float(kappa) / temp) * (lhat_new.float() - lhat.float())
        + ((float(kappa_new) - float(kappa)) / temp) * lhat_new.float()
    )


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"mean_{prefix}": float(x.mean().item()),
        f"std_{prefix}": float(x.std(unbiased=False).item()),
        f"min_{prefix}": float(x.min().item()),
        f"max_{prefix}": float(x.max().item()),
    }


def timing_step_enabled(args: argparse.Namespace, step: int, n_steps: int) -> bool:
    spec = str(getattr(args, "timing_steps", "") or "").strip().lower()
    if spec in {"", "none", "off", "false", "0:false"}:
        return False
    if spec == "all":
        return True
    for raw in spec.replace(";", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "last":
            idx = n_steps - 1
        else:
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx < 0:
                idx = n_steps + idx
        if step == idx:
            return True
    return False


def jvp_step_window_active(
    args: argparse.Namespace,
    step: int,
    n_steps: int,
) -> bool:
    """Return whether the expensive JVP term is retained at this reverse step.

    The window is fixed before sampling and is independent of ESS, rho, and
    particle values. Both boundaries use reverse-step indices: with 100
    steps, ``first=20`` and ``last=10`` retains JVPs exactly on steps 20..89.
    """
    first = max(0, int(getattr(args, "no_jvp_first_steps", 0)))
    last = max(0, int(getattr(args, "no_jvp_last_steps", 0)))
    return first <= int(step) < int(n_steps) - last


def timing_now(device: str | torch.device, enabled: bool) -> float:
    if enabled:
        device_s = str(device)
        if device_s.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    return time.perf_counter()


TIMING_PHASES = (
    "model_eval",
    "base_setup",
    "surrogate_rho",
    "proposal_weight",
    "clip_and_select",
    "weight_temper",
    "distributed_metrics",
    "state_update",
    "step_total",
    # Nested sub-phases of surrogate_rho; do not add these to the phases above.
    "jvp",
    "rho_search",
)


def summarize_timing_records(records: list[dict[str, float]]) -> dict[str, Any]:
    """Summarize per-step slowest-rank timings."""
    summary: dict[str, Any] = {
        "timed_steps": len(records),
        "aggregation": "mean across timed steps of the per-step maximum across ranks",
        "units": "milliseconds",
        "phase_mean_ms": {},
        "phase_std_ms": {},
        "phase_min_ms": {},
        "phase_max_ms": {},
    }
    if not records:
        return summary
    for phase in TIMING_PHASES:
        values = np.asarray([row.get(phase, np.nan) for row in records], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary["phase_mean_ms"][phase] = float(np.mean(finite))
        summary["phase_std_ms"][phase] = float(np.std(finite))
        summary["phase_min_ms"][phase] = float(np.min(finite))
        summary["phase_max_ms"][phase] = float(np.max(finite))
    total_mean = float(summary["phase_mean_ms"].get("step_total", float("nan")))
    additive = TIMING_PHASES[:8]
    summary["additive_phase_share"] = {
        phase: float(summary["phase_mean_ms"][phase] / total_mean)
        for phase in additive
        if phase in summary["phase_mean_ms"] and math.isfinite(total_mean) and total_mean > 0.0
    }
    rho_active = [row["rho_search"] for row in records if row.get("rho_search_active", 0.0) > 0.5]
    summary["rho_search_active_steps"] = len(rho_active)
    summary["rho_search_active_mean_ms"] = (
        float(np.mean(rho_active)) if rho_active else 0.0
    )
    summary["rho_search_active_total_ms"] = (
        float(np.sum(rho_active)) if rho_active else 0.0
    )
    step_totals = np.asarray(
        [row.get("step_total", np.nan) for row in records], dtype=float
    )
    finite_step_totals = step_totals[np.isfinite(step_totals)]
    summary["rho_search_share_of_timed_denoising"] = (
        float(np.sum(rho_active) / np.sum(finite_step_totals))
        if rho_active
        and finite_step_totals.size
        and float(np.sum(finite_step_totals)) > 0.0
        else 0.0
    )
    jvp_active = [
        row["jvp"]
        for row in records
        if row.get("jvp_step_window_active", 0.0) > 0.5
    ]
    summary["jvp_active_steps"] = len(jvp_active)
    summary["jvp_active_mean_ms"] = (
        float(np.mean(jvp_active)) if jvp_active else 0.0
    )
    summary["nested_phases"] = ["jvp", "rho_search"]
    return summary


def weighted_scalar_stats(prefix: str, x: torch.Tensor, logw: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    w = global_normalized_weights_from_logw(logw.detach().float().reshape(-1)).to(x.device)
    mean = torch.sum(w * x)
    var = torch.sum(w * (x - mean).pow(2))
    return {
        f"{prefix}_mean": float(mean.item()),
        f"{prefix}_std": float(var.clamp_min(0.0).sqrt().item()),
    }


def genealogy_stats(origin_ids: torch.Tensor, n_particles: int) -> dict[str, float | int]:
    """Root-ancestor diagnostics for the current particle population."""
    n = int(n_particles)
    if n <= 0 or origin_ids.numel() == 0:
        return {
            "unique_initial_ancestors": 0,
            "unique_initial_ancestor_frac": float("nan"),
            "max_initial_ancestor_copies": 0,
            "max_initial_ancestor_copy_frac": float("nan"),
            "mean_initial_ancestor_copies_nonzero": float("nan"),
            "initial_ancestor_entropy_frac": float("nan"),
            "initial_ancestor_effective_roots": float("nan"),
            "initial_ancestor_effective_root_frac": float("nan"),
        }

    ids = origin_ids.detach().long().reshape(-1).cpu().clamp(0, n - 1)
    counts = torch.bincount(ids, minlength=n).float()
    nonzero = counts > 0
    probs = counts / max(float(ids.numel()), 1.0)
    nz_probs = probs[probs > 0]
    entropy = -torch.sum(nz_probs * torch.log(nz_probs.clamp_min(1e-30)))
    ess_roots = 1.0 / torch.sum(probs.pow(2)).clamp_min(1e-30)
    unique = int(torch.count_nonzero(nonzero).item())
    max_copies = int(counts.max().item())
    return {
        "unique_initial_ancestors": unique,
        "unique_initial_ancestor_frac": float(unique / max(n, 1)),
        "max_initial_ancestor_copies": max_copies,
        "max_initial_ancestor_copy_frac": float(max_copies / max(float(ids.numel()), 1.0)),
        "mean_initial_ancestor_copies_nonzero": float(
            counts[nonzero].mean().item() if bool(nonzero.any()) else float("nan")
        ),
        "initial_ancestor_entropy_frac": float((entropy / math.log(max(n, 2))).item()),
        "initial_ancestor_effective_roots": float(ess_roots.item()),
        "initial_ancestor_effective_root_frac": float(ess_roots.item() / max(n, 1)),
    }


def distributed_logsumexp(local_x: torch.Tensor) -> torch.Tensor:
    """Logsumexp over the global particle set from local rank tensors."""
    x = torch.nan_to_num(local_x.detach().float().reshape(-1), nan=0.0, posinf=80.0, neginf=-80.0)
    if x.numel() == 0:
        local_max = torch.tensor(-float("inf"), device=local_x.device, dtype=torch.float32)
    else:
        local_max = torch.max(x)
    global_max = local_max.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    if not torch.isfinite(global_max):
        return global_max
    local_sum = torch.exp(x - global_max).sum()
    global_sum = local_sum.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    return global_max + torch.log(global_sum.clamp_min(torch.finfo(global_sum.dtype).tiny))


def distributed_ess_frac_from_local_logw(local_logw: torch.Tensor) -> float:
    """ESS/N for global particles, computed without gathering full vectors."""
    x = torch.nan_to_num(local_logw.detach().float().reshape(-1), nan=0.0, posinf=80.0, neginf=-80.0)
    local_n = torch.tensor(float(x.numel()), device=x.device, dtype=torch.float32)
    total_n = local_n.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
    if float(total_n.item()) <= 0.0:
        return float("nan")
    log_z1 = distributed_logsumexp(x)
    log_z2 = distributed_logsumexp(2.0 * x)
    ess = torch.exp(2.0 * log_z1 - log_z2) / total_n
    return float(ess.clamp(0.0, 1.0).item())


def all_gather_particle_rows(local_rows: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather several equal-length particle vectors with one collective.

    all_gather_cat concatenates on its first dimension.  Flattening the local
    [n_rows, local_particles] block lets one collective replace one collective
    per row; the reshape below restores global particle order within each row.
    """
    rows = local_rows.contiguous()
    if rows.ndim != 2:
        raise ValueError(f"Expected [rows, local_particles], got shape={tuple(rows.shape)}")
    if int(world_size) <= 1:
        return rows
    n_rows, local_n = rows.shape
    gathered = all_gather_cat(rows.reshape(-1), int(world_size))
    return (
        gathered.reshape(int(world_size), n_rows, local_n)
        .permute(1, 0, 2)
        .reshape(n_rows, int(world_size) * local_n)
        .contiguous()
    )


def choose_gate_temper_alpha(
    base_logw: torch.Tensor,
    gate_logw: torch.Tensor,
    ess_threshold: float,
    *,
    max_iter: int = 32,
    grid_size: int = 65,
) -> tuple[float, float, float, float]:
    """Choose an active gate bridge fraction.

    Prefer the largest alpha in [0, 1] with
    ESS(base + alpha * gate) >= threshold.  If the threshold is unreachable,
    return the alpha with the largest ESS.  The ESS curve need not be monotone:
    in this sampler the gate term often cancels degeneracy already present in
    the non-gate terms, so checking only the two endpoints can wrongly choose
    alpha=0.

    base_logw may already include carried residual/old particle weights.  In
    that mode the protected quantity is the active cumulative selection weight,
    not just the latest incremental bridge.

    Returns (alpha, ess_base, ess_active, ess_full).
    """
    threshold = float(ess_threshold)
    base = base_logw.detach().float()
    gate = gate_logw.detach().float()
    ess_base = global_ess_frac_from_logw(base)
    ess_full = global_ess_frac_from_logw(base + gate)
    if threshold <= 0.0 or ess_full >= threshold:
        return 1.0, ess_base, ess_full, ess_full

    n_grid = max(3, int(grid_size))
    grid: list[tuple[float, float]] = []
    best_alpha = 0.0
    best_ess = ess_base
    feasible_alpha = None
    for i in range(n_grid):
        alpha = float(i) / float(n_grid - 1)
        ess = ess_base if i == 0 else ess_full if i == n_grid - 1 else global_ess_frac_from_logw(base + alpha * gate)
        grid.append((alpha, ess))
        if ess > best_ess:
            best_alpha = alpha
            best_ess = ess
        if ess >= threshold:
            feasible_alpha = alpha

    if feasible_alpha is None:
        return float(best_alpha), ess_base, float(best_ess), ess_full

    lo = float(feasible_alpha)
    hi = 1.0
    for alpha, ess in grid:
        if alpha > lo and ess < threshold:
            hi = float(alpha)
            break
    else:
        ess_active = global_ess_frac_from_logw(base + lo * gate)
        return float(lo), ess_base, ess_active, ess_full

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ess_mid = global_ess_frac_from_logw(base + mid * gate)
        if ess_mid >= threshold:
            lo = mid
        else:
            hi = mid
    ess_active = global_ess_frac_from_logw(base + lo * gate)
    return float(lo), ess_base, ess_active, ess_full


def clip_final_surrogate_logw(logw: torch.Tensor, clip: float) -> torch.Tensor:
    """Clamp the final surrogate log-weight used by the sampler.

    This is intentionally applied after the retained-order surrogate polynomial
    is evaluated.  It is therefore a propagation/selection stabilizer, not a
    change to the local rho objective or to the decomposition diagnostics.
    """
    clip_f = float(clip)
    if clip_f <= 0.0:
        return logw
    return torch.nan_to_num(
        logw.float(),
        nan=0.0,
        posinf=clip_f,
        neginf=-clip_f,
    ).clamp(-clip_f, clip_f)


def _softmax_mass_np(logw: torch.Tensor, mask: torch.Tensor) -> float:
    logw = torch.nan_to_num(logw.detach().float().reshape(-1), nan=0.0, posinf=80.0, neginf=-80.0)
    mask = mask.detach().reshape(-1).to(device=logw.device, dtype=torch.bool)
    if int(logw.numel()) < 1 or int(mask.sum().item()) < 1:
        return 0.0
    w = torch.softmax(logw - logw.max(), dim=0)
    return float(w[mask].sum().item())


def _global_centered_clip(
    x: torch.Tensor,
    clip: float,
    *,
    center_mode: str,
    count_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    if int(finite.sum().item()) < 1 or float(clip) <= 0.0:
        residual = torch.zeros_like(x)
        return x.clone(), residual, torch.tensor(float("nan"), device=x.device), torch.zeros_like(finite)
    xv = x[finite]
    if center_mode == "mean":
        center = xv.mean()
    elif center_mode == "zero":
        center = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    elif center_mode == "median":
        center = xv.median()
    else:
        raise ValueError(f"Unknown raw clip center mode {center_mode!r}")
    clipped = x.clone()
    clipped_finite = center + (xv - center).clamp(-float(clip), float(clip))
    clipped[finite] = clipped_finite
    residual = x - clipped
    clipped_mask = finite & (residual.abs() > float(count_eps))
    return clipped, residual, center, clipped_mask


def _global_centered_rank_clip(
    x: torch.Tensor,
    topk: int,
    *,
    center_mode: str,
    count_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Winsorize the largest centered absolute deviations to the next rank.

    With topk=4, the four largest |x-center| values are clipped to the 5th
    largest value.  This is intentionally rank based so each step clips the
    same number of JVP outliers when enough finite particles are present.
    """
    x = x.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    n_finite = int(finite.sum().item())
    if n_finite < 2 or int(topk) <= 0:
        residual = torch.zeros_like(x)
        return (
            x.clone(),
            residual,
            torch.tensor(float("nan"), device=x.device),
            torch.zeros_like(finite),
            torch.tensor(float("nan"), device=x.device),
        )
    xv = x[finite]
    if center_mode == "mean":
        center = xv.mean()
    elif center_mode == "zero":
        center = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    elif center_mode == "median":
        center = xv.median()
    else:
        raise ValueError(f"Unknown raw clip center mode {center_mode!r}")

    centered = xv - center
    abs_centered = centered.abs()
    if n_finite <= int(topk):
        threshold = torch.zeros((), device=x.device, dtype=torch.float32)
    else:
        threshold = torch.topk(abs_centered, k=int(topk) + 1, largest=True).values[-1]
    clipped = x.clone()
    clipped_finite = center + centered.sign() * torch.minimum(abs_centered, threshold)
    clipped[finite] = clipped_finite
    residual = x - clipped
    clipped_mask = finite & (residual.abs() > float(count_eps))
    return clipped, residual, center, clipped_mask, threshold


def _rho_gradient_centered_clip_residual(
    x: torch.Tensor,
    *,
    clip: float,
    rank_topk: int,
    center_mode: str,
) -> torch.Tensor:
    """Differentiable-a.e. counterpart of the JVP clipping residual.

    The ordinary global clipping helpers intentionally detach their input
    because they are used by selection/monitoring code.  Rho gradient search
    instead needs the derivative of the *same clipped objective*.  This helper
    reproduces the mean/median/zero centering and fixed-rank winsorization
    algebra without detaching.  Rank selection and the median are piecewise
    differentiable; PyTorch therefore supplies the local (selected-rank)
    derivative, which is the appropriate gradient away from ties.
    """
    values = x.float().reshape(-1)
    finite = torch.isfinite(values)
    n_finite = int(finite.sum().item())
    if n_finite < 1:
        return torch.zeros_like(values)
    finite_values = values[finite]
    if center_mode == "mean":
        center = finite_values.mean()
    elif center_mode == "zero":
        center = torch.zeros((), device=values.device, dtype=torch.float32)
    elif center_mode == "median":
        center = finite_values.median()
    else:
        raise ValueError(f"Unknown raw clip center mode {center_mode!r}")

    centered = finite_values - center
    if int(rank_topk) > 0 and n_finite >= 2:
        if n_finite <= int(rank_topk):
            threshold = torch.zeros((), device=values.device, dtype=torch.float32)
        else:
            threshold = torch.topk(
                centered.abs(),
                k=int(rank_topk) + 1,
                largest=True,
            ).values[-1]
        clipped_values = center + centered.sign() * torch.minimum(
            centered.abs(), threshold
        )
    elif float(clip) > 0.0:
        clipped_values = center + centered.clamp(-float(clip), float(clip))
    else:
        return torch.zeros_like(values)

    # index_copy keeps the dependence on finite_values/clipped_values in the
    # autograd graph while leaving nonfinite positions with zero residual, as
    # in _global_centered_clip/_global_centered_rank_clip.
    finite_indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
    return torch.zeros_like(values).index_copy(
        0,
        finite_indices,
        finite_values - clipped_values,
    )


def _rho_gradient_ess_frac(logw: torch.Tensor) -> torch.Tensor:
    """Differentiable ESS/N on an already gathered global population."""
    values = torch.nan_to_num(
        logw.float().reshape(-1),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    )
    if int(values.numel()) < 1:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float32)
    weights = torch.softmax(values, dim=0)
    return (1.0 / (float(values.numel()) * torch.sum(weights.square()))).clamp(
        0.0, 1.0
    )


def _parse_float_csv(value: str | None) -> list[float]:
    if value is None:
        return []
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def adaptive_rank_clip_topk_from_ess(
    ess_frac: float,
    *,
    enabled: bool,
    fallback_topk: int,
    thresholds: str | None,
    topks: str | None,
) -> int:
    """Choose a rank-clip K from the current unclipped surrogate ESS/N."""
    fallback = max(0, int(fallback_topk))
    if not enabled:
        return fallback
    ess = float(ess_frac)
    if not math.isfinite(ess):
        return fallback
    cuts = _parse_float_csv(thresholds or "0.97,0.93,0.88")
    values = _parse_int_csv(topks or "0,2,4,8")
    if len(values) != len(cuts) + 1:
        raise ValueError(
            "--adaptive-rank-clip-jvp-topks must have exactly one more entry "
            "than --adaptive-rank-clip-jvp-ess-thresholds"
        )
    for left, right in zip(cuts, cuts[1:]):
        if left < right:
            raise ValueError("--adaptive-rank-clip-jvp-ess-thresholds must be high-to-low")
    for threshold, topk in zip(cuts, values):
        if ess >= float(threshold):
            return max(0, int(topk))
    return max(0, int(values[-1]))


def raw_clip_surrogate_terms_global(
    *,
    logw_surrogate: torch.Tensor,
    local_decomp_terms: dict[str, torch.Tensor] | None,
    args: argparse.Namespace,
    world_size: int,
    global_start: int,
    local_n: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Raw biased clipping for selected additive surrogate terms.

    This replaces the active surrogate log-weight term directly.  Unlike the
    gate temper path, no residual is carried forward; the diagnostics quantify
    how large that raw replacement is.
    """
    stats: dict[str, float] = {
        "raw_clip_jvp_threshold": float(getattr(args, "raw_clip_jvp", 0.0)),
        "rank_clip_jvp_topk": float(getattr(args, "rank_clip_jvp_topk", 0)),
        "adaptive_rank_clip_jvp": float(bool(getattr(args, "adaptive_rank_clip_jvp", False))),
        "raw_clip_count_eps": float(getattr(args, "raw_clip_count_eps", 1e-6)),
        "raw_clip_reverse_kernel_threshold": float(getattr(args, "raw_clip_reverse_kernel", 0.0)),
        "raw_clip_center_mode": str(getattr(args, "raw_clip_center", "median")),
    }
    rank_clip_jvp_topk = int(getattr(args, "rank_clip_jvp_topk", 0))
    count_eps = float(getattr(args, "raw_clip_count_eps", 1e-6))
    clip_specs = (
        ("jvp", "proposal_hessian_jvp", float(getattr(args, "raw_clip_jvp", 0.0))),
        ("reverse_kernel", "group_reverse_kernel_alpha", float(getattr(args, "raw_clip_reverse_kernel", 0.0))),
    )
    if local_decomp_terms is None:
        orig_ess = global_ess_frac_from_logw(logw_surrogate.float())
        effective_rank_clip_jvp_topk = adaptive_rank_clip_topk_from_ess(
            orig_ess,
            enabled=bool(getattr(args, "adaptive_rank_clip_jvp", False)),
            fallback_topk=rank_clip_jvp_topk,
            thresholds=str(getattr(args, "adaptive_rank_clip_jvp_ess_thresholds", "0.97,0.93,0.88")),
            topks=str(getattr(args, "adaptive_rank_clip_jvp_topks", "0,2,4,8")),
        )
        for prefix, _, _ in clip_specs:
            stats.update(
                {
                    f"raw_clip_{prefix}_enabled": 0.0,
                    f"raw_clip_{prefix}_frac": 0.0,
                    f"raw_clip_{prefix}_residual_std": 0.0,
                    f"raw_clip_{prefix}_residual_abs_mean": 0.0,
                    f"raw_clip_{prefix}_residual_abs_max": 0.0,
                    f"raw_clip_{prefix}_term_std": float("nan"),
                    f"raw_clip_{prefix}_clipped_term_std": float("nan"),
                    f"raw_clip_{prefix}_center": float("nan"),
                    f"raw_clip_{prefix}_effective_threshold": float("nan"),
                    f"raw_clip_{prefix}_rank_topk": (
                        float(effective_rank_clip_jvp_topk) if prefix == "jvp" else 0.0
                    ),
                    f"raw_clip_{prefix}_orig_mass": 0.0,
                    f"raw_clip_{prefix}_clipped_mass": 0.0,
                }
            )
        stats.update(
            {
                "raw_clip_any_frac": 0.0,
                "raw_clip_total_residual_std": 0.0,
                "raw_clip_total_residual_abs_mean": 0.0,
                "raw_clip_total_residual_abs_max": 0.0,
                "raw_clip_surrogate_orig_std": float(logw_surrogate.float().std(unbiased=False).item()),
                "raw_clip_surrogate_clipped_std": float(logw_surrogate.float().std(unbiased=False).item()),
                "raw_clip_surrogate_orig_ess": orig_ess,
                "raw_clip_surrogate_clipped_ess": orig_ess,
                "raw_clip_surrogate_corr": 1.0,
                "rank_clip_jvp_effective_topk": float(effective_rank_clip_jvp_topk),
                "raw_clip_any_orig_mass": 0.0,
                "raw_clip_any_clipped_mass": 0.0,
            }
        )
        return logw_surrogate, stats

    local_logw = logw_surrogate.float().reshape(-1)
    local_terms: list[torch.Tensor] = []
    for _, key, _ in clip_specs:
        term = local_decomp_terms.get(key)
        local_terms.append(
            torch.full_like(local_logw, float("nan"))
            if term is None
            else term.float().reshape(-1)
        )
    gathered_rows = all_gather_particle_rows(
        torch.stack([local_logw, *local_terms], dim=0),
        world_size,
    ).float()
    global_logw_orig = gathered_rows[0]
    gathered_terms = {
        key: gathered_rows[index + 1]
        for index, (_, key, _) in enumerate(clip_specs)
    }
    global_logw_clipped = global_logw_orig.clone()
    total_residual = torch.zeros_like(global_logw_orig)
    any_mask = torch.zeros_like(global_logw_orig, dtype=torch.bool)
    center_mode = str(getattr(args, "raw_clip_center", "median"))
    orig_ess = global_ess_frac_from_logw(global_logw_orig)
    effective_rank_clip_jvp_topk = adaptive_rank_clip_topk_from_ess(
        orig_ess,
        enabled=bool(getattr(args, "adaptive_rank_clip_jvp", False)),
        fallback_topk=rank_clip_jvp_topk,
        thresholds=str(getattr(args, "adaptive_rank_clip_jvp_ess_thresholds", "0.97,0.93,0.88")),
        topks=str(getattr(args, "adaptive_rank_clip_jvp_topks", "0,2,4,8")),
    )

    for prefix, key, clip in clip_specs:
        term = local_decomp_terms.get(key)
        effective_rank_topk = effective_rank_clip_jvp_topk if prefix == "jvp" else rank_clip_jvp_topk
        use_rank_clip = prefix == "jvp" and effective_rank_topk > 0
        enabled = float(((use_rank_clip and effective_rank_topk > 0) or clip > 0.0) and term is not None)
        global_term = gathered_terms[key]
        if use_rank_clip:
            clipped_term, residual, center, clipped_mask, effective_threshold = _global_centered_rank_clip(
                global_term,
                effective_rank_topk,
                center_mode=center_mode,
                count_eps=count_eps,
            )
        else:
            clipped_term, residual, center, clipped_mask = _global_centered_clip(
                global_term,
                clip,
                center_mode=center_mode,
                count_eps=count_eps,
            )
            effective_threshold = torch.tensor(float(clip) if clip > 0.0 else float("nan"), device=global_term.device)
        if enabled:
            global_logw_clipped = global_logw_clipped - residual
            total_residual = total_residual + residual
            any_mask = any_mask | clipped_mask

        finite_term = torch.isfinite(global_term)
        finite_clipped = torch.isfinite(clipped_term)
        stats.update(
            {
                f"raw_clip_{prefix}_enabled": enabled,
                f"raw_clip_{prefix}_frac": float(clipped_mask.float().mean().item()) if clipped_mask.numel() else 0.0,
                f"raw_clip_{prefix}_residual_std": float(residual.std(unbiased=False).item()),
                f"raw_clip_{prefix}_residual_abs_mean": float(residual.abs().mean().item()),
                f"raw_clip_{prefix}_residual_abs_max": float(residual.abs().max().item()),
                f"raw_clip_{prefix}_term_std": (
                    float(global_term[finite_term].std(unbiased=False).item())
                    if bool(finite_term.any().item())
                    else float("nan")
                ),
                f"raw_clip_{prefix}_clipped_term_std": (
                    float(clipped_term[finite_clipped].std(unbiased=False).item())
                    if bool(finite_clipped.any().item())
                    else float("nan")
                ),
                f"raw_clip_{prefix}_center": float(center.item()) if torch.isfinite(center) else float("nan"),
                f"raw_clip_{prefix}_effective_threshold": (
                    float(effective_threshold.item()) if torch.isfinite(effective_threshold) else float("nan")
                ),
                f"raw_clip_{prefix}_rank_topk": float(effective_rank_topk if use_rank_clip else 0),
                f"raw_clip_{prefix}_orig_mass": _softmax_mass_np(global_logw_orig, clipped_mask),
                f"raw_clip_{prefix}_clipped_mass": _softmax_mass_np(global_logw_clipped, clipped_mask),
            }
        )

    local_slice = slice(global_start, global_start + local_n)
    clipped_local_logw = global_logw_clipped[local_slice].to(device=logw_surrogate.device, dtype=torch.float32)
    finite_pair = torch.isfinite(global_logw_orig) & torch.isfinite(global_logw_clipped)
    if int(finite_pair.sum().item()) >= 2:
        raw_corr = finite_corr(global_logw_orig[finite_pair], global_logw_clipped[finite_pair])
    else:
        raw_corr = float("nan")
    stats.update(
        {
            "raw_clip_any_frac": float(any_mask.float().mean().item()) if any_mask.numel() else 0.0,
            "raw_clip_total_residual_std": float(total_residual.std(unbiased=False).item()),
            "raw_clip_total_residual_abs_mean": float(total_residual.abs().mean().item()),
            "raw_clip_total_residual_abs_max": float(total_residual.abs().max().item()),
            "raw_clip_surrogate_orig_std": float(global_logw_orig.std(unbiased=False).item()),
            "raw_clip_surrogate_clipped_std": float(global_logw_clipped.std(unbiased=False).item()),
            "raw_clip_surrogate_orig_ess": orig_ess,
            "raw_clip_surrogate_clipped_ess": global_ess_frac_from_logw(global_logw_clipped),
            "raw_clip_surrogate_corr": raw_corr,
            "rank_clip_jvp_effective_topk": float(effective_rank_clip_jvp_topk),
            "raw_clip_any_orig_mass": _softmax_mass_np(global_logw_orig, any_mask),
            "raw_clip_any_clipped_mass": _softmax_mass_np(global_logw_clipped, any_mask),
        }
    )
    return clipped_local_logw.reshape_as(logw_surrogate.float()), stats


def raw_clipped_local_decomp_term(
    *,
    local_decomp_terms: dict[str, torch.Tensor] | None,
    key: str,
    clip: float,
    rank_topk: int = 0,
    center_mode: str = "median",
    count_eps: float = 1e-6,
    world_size: int = 1,
    global_start: int = 0,
    local_n: int = 0,
) -> torch.Tensor | None:
    """Return the local slice of a decomp term after the configured raw clip.

    This mirrors raw_clip_surrogate_terms_global for one additive term.  It is
    used by JVP tempering so the bridge tempers the already-robustified JVP
    term rather than reintroducing the unclipped tail.
    """
    if local_decomp_terms is None:
        return None
    term = local_decomp_terms.get(key)
    if term is None:
        return None
    local_term = term.float()
    global_term = all_gather_cat(local_term.reshape(-1).contiguous(), world_size).float()
    if int(rank_topk) > 0:
        clipped, _, _, _, _ = _global_centered_rank_clip(
            global_term,
            int(rank_topk),
            center_mode=str(center_mode),
            count_eps=float(count_eps),
        )
    else:
        clipped, _, _, _ = _global_centered_clip(
            global_term,
            float(clip),
            center_mode=str(center_mode),
            count_eps=float(count_eps),
        )
    local_slice = slice(global_start, global_start + local_n)
    return clipped[local_slice].to(device=local_term.device, dtype=torch.float32).reshape_as(local_term)


def lhat_resampling_diagnostics(
    *,
    lhat: torch.Tensor,
    lhat_new: torch.Tensor,
    kernel_lr: torch.Tensor,
    logw_used: torch.Tensor,
    logw_true: torch.Tensor,
    logw_surrogate: torch.Tensor,
    ancestor_idx: torch.Tensor | None,
) -> dict[str, float]:
    """Diagnostics for how lhat increments and resampling bend the trajectory."""
    lhat_f = lhat.detach().float().reshape(-1)
    lhat_new_f = lhat_new.detach().float().reshape(-1)
    kernel_f = kernel_lr.detach().float().reshape(-1)
    inc = lhat_new_f - lhat_f

    out: dict[str, float] = {}
    out.update(tensor_stats("lhat_inc", inc))
    out.update(weighted_scalar_stats("resample_expected_lhat_new", lhat_new_f, logw_used))
    out.update(weighted_scalar_stats("resample_expected_kernel_lr", kernel_f, logw_used))
    out["resample_expected_mean_lhat_shift_vs_current"] = (
        out["resample_expected_lhat_new_mean"] - float(lhat_f.mean().item())
    )
    out["resample_expected_mean_lhat_shift_vs_unweighted_new"] = (
        out["resample_expected_lhat_new_mean"] - float(lhat_new_f.mean().item())
    )
    out["resample_expected_mean_kernel_lr_shift_vs_unweighted"] = (
        out["resample_expected_kernel_lr_mean"] - float(kernel_f.mean().item())
    )

    out["corr_lhat_logw_used"] = finite_corr(lhat_f, logw_used)
    out["corr_lhat_new_logw_used"] = finite_corr(lhat_new_f, logw_used)
    out["corr_kernel_lr_logw_used"] = finite_corr(kernel_f, logw_used)
    out["corr_lhat_inc_logw_used"] = finite_corr(inc, logw_used)
    out["corr_lhat_new_logw_true"] = finite_corr(lhat_new_f, logw_true)
    out["corr_kernel_lr_logw_true"] = finite_corr(kernel_f, logw_true)
    out["corr_lhat_new_logw_surrogate"] = finite_corr(lhat_new_f, logw_surrogate)
    out["corr_kernel_lr_logw_surrogate"] = finite_corr(kernel_f, logw_surrogate)

    n = int(lhat_f.numel())
    if ancestor_idx is None or n == 0:
        for key in [
            "selected_lhat_new_mean",
            "selected_lhat_new_std",
            "selected_kernel_lr_mean",
            "selected_kernel_lr_std",
            "selected_mean_lhat_shift_vs_current",
            "selected_mean_lhat_shift_vs_unweighted_new",
            "selected_std_lhat_shift_vs_unweighted_new",
            "ancestor_unique_frac",
            "ancestor_max_count_frac",
            "ancestor_entropy_frac",
        ]:
            out[key] = float("nan")
        return out

    idx = ancestor_idx.detach().long().reshape(-1).to(lhat_f.device)
    idx = idx.clamp(0, max(n - 1, 0))
    selected_lhat = lhat_new_f[idx]
    selected_kernel = kernel_f[idx]
    out["selected_lhat_new_mean"] = float(selected_lhat.mean().item())
    out["selected_lhat_new_std"] = float(selected_lhat.std(unbiased=False).item())
    out["selected_kernel_lr_mean"] = float(selected_kernel.mean().item())
    out["selected_kernel_lr_std"] = float(selected_kernel.std(unbiased=False).item())
    out["selected_mean_lhat_shift_vs_current"] = (
        out["selected_lhat_new_mean"] - float(lhat_f.mean().item())
    )
    out["selected_mean_lhat_shift_vs_unweighted_new"] = (
        out["selected_lhat_new_mean"] - float(lhat_new_f.mean().item())
    )
    out["selected_std_lhat_shift_vs_unweighted_new"] = (
        out["selected_lhat_new_std"] - float(lhat_new_f.std(unbiased=False).item())
    )

    counts = torch.bincount(idx, minlength=n).float()
    probs = counts / max(float(idx.numel()), 1.0)
    nz = probs > 0
    entropy = -torch.sum(probs[nz] * torch.log(probs[nz].clamp_min(1e-30)))
    out["ancestor_unique_frac"] = float((counts > 0).float().mean().item())
    out["ancestor_max_count_frac"] = float((counts.max() / max(float(idx.numel()), 1.0)).item())
    out["ancestor_entropy_frac"] = float((entropy / math.log(max(n, 2))).item())
    return out


def normalize_clip_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_clip_score_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.asarray([int(r["particle_index"]) for r in rows], dtype=int)
    metric_labels = [
        ("clip_positive", "positive"),
        ("clip_related_negative", "related negative"),
        ("clip_unrelated_negative", "unrelated negative"),
        ("clip_used_negative", "used negative"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    for key, label in metric_labels:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
        if np.isfinite(vals).any():
            axes[0].plot(idx, vals, marker="o", ms=3, lw=1.2, label=label)
    axes[0].set_title("CLIP Scores Per Particle")
    axes[0].set_xlabel("particle")
    axes[0].set_ylabel("cosine similarity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    means = []
    names = []
    for key, label in metric_labels:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            names.append(label)
            means.append(float(finite.mean()))
    axes[1].bar(np.arange(len(names)), means, color=["tab:blue", "tab:red", "tab:purple", "tab:orange"][: len(names)])
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("mean cosine similarity")
    axes[1].set_title("Mean CLIP Scores")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "clip_scores.png", bbox_inches="tight")
    plt.close(fig)


def score_final_images_with_clip(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    negative: str,
) -> dict[str, Any]:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    img_dir = args.out_dir / "images"
    image_paths = [img_dir / f"particle_{idx:03d}.png" for idx in range(int(args.n_particles))]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(
            f"CLIP scoring needs saved final images, but {len(missing)} are missing. "
            f"First missing paths:\n{preview}"
        )

    devices = parse_devices(args.devices, args.device)
    score_device = str(args.clip_score_device or (devices[0] if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if score_device.startswith("cuda") else torch.float32
    model_id = str(args.clip_model_id)
    print(f"[clip] loading {model_id} on {score_device}", flush=True)
    model = CLIPModel.from_pretrained(
        model_id,
        cache_dir=str(HF_CACHE),
        local_files_only=bool(args.clip_local_files_only),
        torch_dtype=dtype,
    ).to(score_device)
    processor = CLIPProcessor.from_pretrained(
        model_id,
        cache_dir=str(HF_CACHE),
        local_files_only=bool(args.clip_local_files_only),
    )
    model.eval()

    text_by_label = {
        "positive": str(prompt["positive"]),
        "related_negative": str(prompt.get("related_negative", "")),
        "unrelated_negative": str(prompt.get("unrelated_negative", "")),
        "used_negative": str(negative),
    }
    unique_texts = sorted({text for text in text_by_label.values() if text})

    text_features: dict[str, torch.Tensor] = {}
    text_batch = max(1, int(args.clip_text_batch_size))
    for start in range(0, len(unique_texts), text_batch):
        batch_texts = unique_texts[start : start + text_batch]
        inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(score_device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = normalize_clip_features(model.get_text_features(**inputs))
        for text, feat in zip(batch_texts, feats.detach().cpu()):
            text_features[text] = feat.float()

    image_features: dict[str, torch.Tensor] = {}
    image_batch = max(1, int(args.clip_batch_size))
    for start in range(0, len(image_paths), image_batch):
        batch_paths = image_paths[start : start + image_batch]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(score_device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            feats = normalize_clip_features(model.get_image_features(**inputs))
        for path, feat in zip(batch_paths, feats.detach().cpu()):
            image_features[str(path)] = feat.float()
        print(
            f"[clip] image features {min(start + image_batch, len(image_paths))}/{len(image_paths)}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths):
        image_feat = image_features[str(path)]
        pos_feat = text_features[text_by_label["positive"]]
        related_feat = text_features[text_by_label["related_negative"]]
        unrelated_feat = text_features[text_by_label["unrelated_negative"]]
        used_feat = text_features[text_by_label["used_negative"]]
        clip_positive = float(image_feat @ pos_feat)
        clip_related = float(image_feat @ related_feat)
        clip_unrelated = float(image_feat @ unrelated_feat)
        clip_used = float(image_feat @ used_feat)
        rows.append(
            {
                "particle_index": idx,
                "image_path": str(path),
                "negative_kind": str(args.negative_kind),
                "positive_prompt": text_by_label["positive"],
                "related_negative_prompt": text_by_label["related_negative"],
                "unrelated_negative_prompt": text_by_label["unrelated_negative"],
                "used_negative_prompt": text_by_label["used_negative"],
                "clip_positive": clip_positive,
                "clip_related_negative": clip_related,
                "clip_unrelated_negative": clip_unrelated,
                "clip_used_negative": clip_used,
                "clip_margin_pos_minus_related_negative": clip_positive - clip_related,
                "clip_margin_pos_minus_unrelated_negative": clip_positive - clip_unrelated,
                "clip_margin_pos_minus_used_negative": clip_positive - clip_used,
            }
        )

    metrics_dir = args.out_dir / "metrics"
    scores_path = metrics_dir / "clip_scores.csv"
    write_dict_rows(scores_path, rows)
    make_clip_score_plot(rows, metrics_dir)

    summary: dict[str, Any] = {
        "clip_score_model": model_id,
        "clip_score_device": score_device,
        "clip_score_n_images": len(rows),
    }
    numeric_keys = [key for key in rows[0] if key.startswith("clip_")]
    for key in numeric_keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            summary[f"{key}_mean"] = float(finite.mean())
            summary[f"{key}_std"] = float(finite.std())
            summary[f"{key}_min"] = float(finite.min())
            summary[f"{key}_max"] = float(finite.max())

    summary_path = metrics_dir / "clip_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    run_summary_path = args.out_dir / "summary.json"
    if run_summary_path.exists():
        with run_summary_path.open() as f:
            run_summary = json.load(f)
        run_summary.update(summary)
        run_summary["clip_scores_csv"] = str(scores_path)
        run_summary["clip_summary_json"] = str(summary_path)
        with run_summary_path.open("w") as f:
            json.dump(run_summary, f, indent=2)

    print(f"[clip] wrote {scores_path}", flush=True)
    print(f"[clip] wrote {summary_path}", flush=True)
    print(f"[clip] wrote {metrics_dir / 'clip_scores.png'}", flush=True)
    return summary


def beta_variance_stats(beta_flat: torch.Tensor, *, lhat_temp: float) -> dict[str, float]:
    """Cross-particle heterogeneity of the tempered A/B score difference."""
    beta = float(lhat_temp) * beta_flat.detach().float()
    n, dim = beta.shape
    dim = max(int(dim), 1)
    norm2 = beta.pow(2).sum(dim=1)
    out = {
        "mean_gl2_per_dim": float(norm2.mean().item() / dim),
        "std_gl2_per_dim": float(norm2.std(unbiased=False).item() / dim),
        "cv_gl2": float((norm2.std(unbiased=False) / norm2.mean().clamp_min(1e-30)).item()),
        "beta_coord_var_mean": float(beta.var(dim=0, unbiased=False).mean().item()),
        "beta_coord_std_mean": float(beta.std(dim=0, unbiased=False).mean().item()),
        "beta_mean_norm2_per_dim": float(beta.mean(dim=0).pow(2).sum().item() / dim),
    }
    if n >= 2:
        d = torch.pdist(beta, p=2)
        out["beta_pair_dist_mean_per_sqrt_dim"] = float((d.mean() / math.sqrt(dim)).item())
        out["beta_pair_dist_min_per_sqrt_dim"] = float((d.min() / math.sqrt(dim)).item())
        out["beta_pair_dist_max_per_sqrt_dim"] = float((d.max() / math.sqrt(dim)).item())
    else:
        out["beta_pair_dist_mean_per_sqrt_dim"] = float("nan")
        out["beta_pair_dist_min_per_sqrt_dim"] = float("nan")
        out["beta_pair_dist_max_per_sqrt_dim"] = float("nan")
    return out


def component_norm_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    """Particlewise displacement magnitudes, normalized to be comparable across latent sizes."""
    flat = x.detach().float().flatten(1)
    dim = max(int(flat.shape[1]), 1)
    norm = flat.norm(dim=1) / math.sqrt(dim)
    norm2 = flat.pow(2).sum(dim=1) / dim
    return {
        f"{prefix}_norm_mean_per_sqrt_dim": float(norm.mean().item()),
        f"{prefix}_norm_std_per_sqrt_dim": float(norm.std(unbiased=False).item()),
        f"{prefix}_norm_min_per_sqrt_dim": float(norm.min().item()),
        f"{prefix}_norm_max_per_sqrt_dim": float(norm.max().item()),
        f"{prefix}_norm2_mean_per_dim": float(norm2.mean().item()),
    }


def drift_component_stats(components: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in components.items():
        out.update(component_norm_stats(f"drift_{name}", value))

    def mean_norm(name: str) -> float:
        return out.get(f"drift_{name}_norm_mean_per_sqrt_dim", float("nan"))

    eps = 1e-30
    base = mean_norm("base")
    brownian = mean_norm("brownian")
    guidance_rho0 = mean_norm("guidance_rho0")
    guidance_proposal = mean_norm("guidance_proposal")
    guidance_rho_correction = mean_norm("guidance_rho_correction")
    deterministic = mean_norm("deterministic_total")
    sample_total = mean_norm("sample_total")
    out.update(
        {
            "drift_guidance_proposal_over_base": float(guidance_proposal / max(base, eps)),
            "drift_guidance_proposal_over_brownian": float(guidance_proposal / max(brownian, eps)),
            "drift_rho_correction_over_guidance_rho0": float(
                guidance_rho_correction / max(guidance_rho0, eps)
            ),
            "drift_deterministic_over_brownian": float(deterministic / max(brownian, eps)),
            "drift_sample_over_brownian": float(sample_total / max(brownian, eps)),
        }
    )
    return out


def conditional_ess_frac_from_logw(logw_prev: torch.Tensor, logw_inc: torch.Tensor) -> float:
    """Conditional ESS/N for the latest incremental weights.

    Unlike pure incESS, this respects the current normalized particle weights
    before the latest increment.  If the previous weights are uniform, it
    reduces to the relative ESS of the incremental weights.
    """
    logw_prev = torch.nan_to_num(logw_prev.float(), nan=0.0, posinf=80.0, neginf=-80.0)
    logw_inc = torch.nan_to_num(logw_inc.float(), nan=0.0, posinf=80.0, neginf=-80.0)
    log_wprev = logw_prev - torch.logsumexp(logw_prev, dim=0)
    log_num = 2.0 * torch.logsumexp(log_wprev + logw_inc, dim=0)
    log_den = torch.logsumexp(log_wprev + 2.0 * logw_inc, dim=0)
    return float(torch.exp(log_num - log_den).clamp(0.0, 1.0).item())


def should_resample_from_ess(
    args: argparse.Namespace,
    *,
    step: int,
    n_steps: int,
    cum_ess: float,
    inc_ess: float,
    cess_ess: float,
    resample_armed: bool = True,
) -> tuple[bool, float, bool, bool, bool, str]:
    mode = str(getattr(args, "resample_ess_mode", "cum"))
    if mode == "cum":
        score = float(cum_ess)
    elif mode == "inc":
        score = float(inc_ess)
    elif mode == "cess":
        score = float(cess_ess)
    elif mode == "cum_or_inc":
        score = min(float(cum_ess), float(inc_ess))
    elif mode == "cum_or_cess":
        score = min(float(cum_ess), float(cess_ess))
    else:
        raise ValueError(f"Unknown --resample-ess-mode={mode!r}")

    rearm_threshold = float(getattr(args, "resample_ess_rearm", 0.0))
    hysteresis_enabled = rearm_threshold > 0.0
    rearmed = False
    armed_now = bool(resample_armed) if hysteresis_enabled else True
    if hysteresis_enabled and not armed_now and score >= rearm_threshold:
        armed_now = True
        rearmed = True

    no_resample_last = max(0, int(getattr(args, "no_resample_last_steps", 0)))
    in_final_block = no_resample_last > 0 and int(step) >= max(0, int(n_steps) - no_resample_last)
    below_trigger = bool(args.resample_ess <= 1.0 and score < float(args.resample_ess))
    do_resample = bool(below_trigger and armed_now and not in_final_block)
    armed_after = False if do_resample and hysteresis_enabled else armed_now

    if in_final_block:
        block_reason = "final"
    elif below_trigger and not armed_now:
        block_reason = "hysteresis"
    else:
        block_reason = ""
    resample_allowed = bool(not in_final_block and armed_now)
    return do_resample, score, resample_allowed, armed_after, rearmed, block_reason


def forward_kernel_mean_variance(scheduler, timestep: int, x_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate DDPM forward K(y_t | x_prev) for the inference jump."""
    t = int(timestep)
    prev_t = int(scheduler.previous_timestep(t))
    if prev_t < 0:
        var = torch.tensor(1e-20, device=x_prev.device, dtype=torch.float32)
        return x_prev, var
    alpha_t = scheduler.alphas_cumprod[t].to(device=x_prev.device, dtype=x_prev.dtype)
    alpha_prev = scheduler.alphas_cumprod[prev_t].to(device=x_prev.device, dtype=x_prev.dtype)
    ratio = (alpha_t / alpha_prev).clamp(min=0.0, max=1.0)
    mean = ratio.sqrt() * x_prev
    var = (1.0 - ratio).to(dtype=torch.float32).clamp_min(1e-20)
    return mean, var


def forward_kernel_displacement_jacobian(scheduler, timestep: int, device: str | torch.device) -> float:
    """Jacobian scalar of K^mu mean displacement, mean(x)-x, for DDPM forward kernel."""
    t = int(timestep)
    prev_t = int(scheduler.previous_timestep(t))
    if prev_t < 0:
        return 0.0
    alpha_t = scheduler.alphas_cumprod[t].to(device=device, dtype=torch.float32)
    alpha_prev = scheduler.alphas_cumprod[prev_t].to(device=device, dtype=torch.float32)
    ratio = (alpha_t / alpha_prev).clamp(min=0.0, max=1.0)
    return float(ratio.sqrt().item() - 1.0)


def predicted_noise_guidance_factor(scheduler, timestep: int) -> float:
    """Convert a score-guidance coefficient into a DDPM epsilon coefficient.

    The pA proposal adds sigma_reverse^2 * lambda_s * (score_A - score_B) to
    the reverse mean.  For epsilon prediction,

        score_A - score_B = -(eps_A - eps_B) / sqrt(1 - alpha_bar_t).

    This returns the multiplier f_t such that

        lambda_eps * (eps_A - eps_B)

    inside a standard DDPMScheduler epsilon model output gives the same reverse
    mean shift, with lambda_eps = f_t * lambda_s.
    """
    t = int(timestep)
    prev_t = int(scheduler.previous_timestep(t))
    alpha_t = float(scheduler.alphas_cumprod[t].detach().float().item())
    alpha_prev = (
        float(scheduler.alphas_cumprod[prev_t].detach().float().item())
        if prev_t >= 0
        else 1.0
    )
    beta_t = max(1.0 - alpha_t, 1e-30)
    current_alpha = max(alpha_t / max(alpha_prev, 1e-30), 1e-30)
    current_beta = max(1.0 - current_alpha, 0.0)
    pred_x0_coeff = math.sqrt(alpha_prev) * current_beta / beta_t
    dmean_deps = pred_x0_coeff * math.sqrt(beta_t / max(alpha_t, 1e-30))
    if dmean_deps <= 0.0:
        return 0.0
    variance = float(scheduler._get_variance(t).detach().float().item())
    return float(variance / (math.sqrt(beta_t) * dmean_deps))


def ab_outputs_and_beta(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
    *,
    linear_baseline_kappa: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base, gate-A/B epsilon outputs and beta_l.

    prompt_embeds is packed as [unconditional, A, B].  The configurable base
    branch is

        eps_base = eps_u
                   + lambda_cfg * (eps_A - eps_u)
                   + lambda_neg * (eps_B - eps_u).

    ``lambda_neg`` defaults to zero, exactly recovering the prior base path.
    It affects only the base reverse kernel.  In the default cfg-real
    gate-ratio mode, the gate retains its separate powered A and B branches, so

        beta_l = -lambda_cfg * (eps_B - eps_A) / sqrt(1 - alpha_bar_t).

    In raw gate-ratio mode, the gate ratio is the unpowered A/B prompt ratio:

        beta_l = -(eps_B - eps_A) / sqrt(1 - alpha_bar_t).
    """
    chunk_size = int(getattr(pipe, "_dng_unet_chunk_size", 0) or 0)
    if chunk_size > 0 and latents.shape[0] > chunk_size:
        uncond_embeds, pos_embeds, neg_embeds = prompt_embeds.chunk(3)
        outs_a, outs_gate_a, outs_b, betas = [], [], [], []
        for start in range(0, latents.shape[0], chunk_size):
            end = min(start + chunk_size, latents.shape[0])
            chunk_embeds = torch.cat(
                [
                    uncond_embeds[start:end],
                    pos_embeds[start:end],
                    neg_embeds[start:end],
                ],
                dim=0,
            )
            out_a_base, out_a_gate, out_b_gate, beta = ab_outputs_and_beta(
                pipe,
                latents[start:end],
                timestep,
                chunk_embeds,
                linear_baseline_kappa=linear_baseline_kappa,
            )
            outs_a.append(out_a_base)
            outs_gate_a.append(out_a_gate)
            outs_b.append(out_b_gate)
            betas.append(beta)
        return (
            torch.cat(outs_a, dim=0),
            torch.cat(outs_gate_a, dim=0),
            torch.cat(outs_b, dim=0),
            torch.cat(betas, dim=0),
        )

    t_int = int(timestep)
    latent_model_input = torch.cat([latents, latents, latents], dim=0)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    model_out = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    out_uncond, out_pos, out_neg = model_out.chunk(3)

    eps_u = model_output_to_epsilon(pipe.scheduler, out_uncond, t_int, latents)
    eps_a = model_output_to_epsilon(pipe.scheduler, out_pos, t_int, latents)
    eps_b = model_output_to_epsilon(pipe.scheduler, out_neg, t_int, latents)
    cfg_scale = float(getattr(pipe, "_pa_gate_cfg_scale", 1.0))
    base_lambda_neg = float(getattr(pipe, "_pa_gate_base_lambda_neg", 0.0))
    prediction_type = str(getattr(pipe.scheduler.config, "prediction_type", "epsilon"))
    if prediction_type != "epsilon":
        raise ValueError(
            "CFG-real pA-gate runner currently expects an epsilon-prediction scheduler; "
            f"got prediction_type={prediction_type!r}"
        )
    out_a_real = eps_u + cfg_scale * (eps_a - eps_u)
    out_b_real = eps_u + cfg_scale * (eps_b - eps_u)
    if linear_baseline_kappa is None:
        linear_baseline_kappa = float(
            getattr(pipe, "_pa_gate_linear_baseline_kappa", 0.0)
        )
    else:
        linear_baseline_kappa = float(linear_baseline_kappa)
    if linear_baseline_kappa > 0.0:
        # Algebraic pure-PA factorization:
        #   p_A V_PA = [p_A (p_A/p_B)^kappa] *
        #              [V_PA (p_B/p_A)^kappa].
        # The first bracket supplies the NP-equivalent DDPM reference mean;
        # the second is the centered nonlinear PA residual handled below.
        out_base = eps_a + linear_baseline_kappa * (eps_a - eps_b)
    else:
        out_base = out_a_real + base_lambda_neg * (eps_b - eps_u)
    gate_ratio_mode = str(getattr(pipe, "_pa_gate_gate_ratio_mode", "cfg-real"))
    if gate_ratio_mode == "cfg-real":
        out_a_gate = out_a_real
        out_b_gate = out_b_real
    elif gate_ratio_mode == "raw":
        out_a_gate = eps_a
        out_b_gate = eps_b
    else:
        raise ValueError(f"Unknown gate ratio mode {gate_ratio_mode!r}")
    alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)
    beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
    beta_l = -(out_b_gate - out_a_gate) / beta_prod_t.sqrt()
    return out_base, out_a_gate, out_b_gate, beta_l.float()


def beta_only_ab(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    """Return beta_l using only the positive-A and negative-B UNet branches.

    The unconditional branch cancels from both supported gate ratios:

      raw:      eps_B - eps_A
      cfg-real: cfg * (eps_B - eps_A)

    This specialized path is used by JVP calculations, where propagating a
    dual tensor through the unused unconditional branch is especially costly.
    The ordinary denoising evaluation still uses ``ab_outputs_and_beta``
    because it needs the unconditional branch to construct the configured
    base mean.
    """
    if int(prompt_embeds.shape[0]) != 3 * int(latents.shape[0]):
        raise ValueError(
            "beta_only_ab expects prompt embeddings packed as "
            f"[unconditional, A, B]: got {prompt_embeds.shape[0]} embeddings "
            f"for {latents.shape[0]} latents"
        )

    chunk_size = int(getattr(pipe, "_dng_unet_chunk_size", 0) or 0)
    if chunk_size > 0 and latents.shape[0] > chunk_size:
        _, pos_embeds, neg_embeds = prompt_embeds.chunk(3)
        betas = []
        for start in range(0, latents.shape[0], chunk_size):
            end = min(start + chunk_size, latents.shape[0])
            # Preserve the public [unconditional, A, B] packing expected by
            # the recursive call.  The first block is only a placeholder and
            # is discarded before the UNet invocation.
            chunk_prompt_embeds = torch.cat(
                [
                    pos_embeds[start:end],
                    pos_embeds[start:end],
                    neg_embeds[start:end],
                ],
                dim=0,
            )
            betas.append(
                beta_only_ab(
                    pipe,
                    latents[start:end],
                    timestep,
                    chunk_prompt_embeds,
                )
            )
        return torch.cat(betas, dim=0)

    _, pos_embeds, neg_embeds = prompt_embeds.chunk(3)
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    model_out = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=torch.cat([pos_embeds, neg_embeds], dim=0),
        return_dict=False,
    )[0]
    out_pos, out_neg = model_out.chunk(2)

    t_int = int(timestep)
    eps_a = model_output_to_epsilon(pipe.scheduler, out_pos, t_int, latents)
    eps_b = model_output_to_epsilon(pipe.scheduler, out_neg, t_int, latents)
    prediction_type = str(getattr(pipe.scheduler.config, "prediction_type", "epsilon"))
    if prediction_type != "epsilon":
        raise ValueError(
            "CFG-real pA-gate runner currently expects an epsilon-prediction scheduler; "
            f"got prediction_type={prediction_type!r}"
        )
    gate_ratio_mode = str(getattr(pipe, "_pa_gate_gate_ratio_mode", "cfg-real"))
    if gate_ratio_mode == "cfg-real":
        gate_scale = float(getattr(pipe, "_pa_gate_cfg_scale", 1.0))
    elif gate_ratio_mode == "raw":
        gate_scale = 1.0
    else:
        raise ValueError(f"Unknown gate ratio mode {gate_ratio_mode!r}")
    alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(
        device=latents.device,
        dtype=latents.dtype,
    )
    beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
    return (-gate_scale * (eps_b - eps_a) / beta_prod_t.sqrt()).float()


def finite_difference_jbeta_z(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
    z: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Central finite-difference directional JVP J beta_l(latents)[z].

    The plus/minus perturbations are packed into one UNet call.  This is the
    cheapest robust JVP path for SD here because exact autograd JVP would need
    gradient-enabled UNet execution through attention blocks.
    """
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("--jvp-eps must be positive")
    lat_plus = latents + eps * z.to(dtype=latents.dtype)
    lat_minus = latents - eps * z.to(dtype=latents.dtype)
    uncond_embeds, pos_embeds, neg_embeds = prompt_embeds.chunk(3)
    pair_embeds = torch.cat(
        [
            torch.cat([uncond_embeds, uncond_embeds], dim=0),
            torch.cat([pos_embeds, pos_embeds], dim=0),
            torch.cat([neg_embeds, neg_embeds], dim=0),
        ],
        dim=0,
    )
    beta_pair = beta_only_ab(
        pipe,
        torch.cat([lat_plus, lat_minus], dim=0),
        timestep,
        pair_embeds,
    )
    beta_plus, beta_minus = beta_pair.chunk(2)
    return (beta_plus - beta_minus) / (2.0 * eps)


def forward_ad_jbeta_z(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Forward-mode AD directional JVP J beta_l(latents)[z]."""
    tangent = z.detach().to(dtype=latents.dtype, device=latents.device)
    primal = latents.detach()
    with torch.autograd.forward_ad.dual_level():
        dual_latents = torch.autograd.forward_ad.make_dual(primal, tangent)
        beta_dual = beta_only_ab(pipe, dual_latents, timestep, prompt_embeds)
        _, jbeta_z = torch.autograd.forward_ad.unpack_dual(beta_dual)
    if jbeta_z is None:
        raise RuntimeError("forward-mode AD did not produce a tangent for beta_l")
    return jbeta_z.detach().float()


def compute_jbeta_z(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
    z: torch.Tensor,
    *,
    mode: str,
    fd_eps: float,
    rank: int,
) -> tuple[torch.Tensor, str]:
    if mode in {"forward-ad", "auto"}:
        try:
            return (
                forward_ad_jbeta_z(pipe, latents, timestep, prompt_embeds, z),
                "forward-ad",
            )
        except Exception as exc:
            if mode == "forward-ad":
                raise
            if rank == 0:
                print(f"[jvp] forward-ad failed; falling back to finite-diff: {exc}", flush=True)
    return (
        finite_difference_jbeta_z(pipe, latents, timestep, prompt_embeds, z, fd_eps),
        "finite-diff",
    )


def _broadcast_particle_scalar(x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or per-particle scalar to a flattened particle tensor."""
    x = x.float()
    if x.ndim == 0:
        return x
    return x.reshape(-1, *([1] * (flat.ndim - 1)))


def local_surrogate_terms(
    rho_ratio: torch.Tensor | float,
    *,
    latents: torch.Tensor,
    mu_a: torch.Tensor,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    forward_displacement: torch.Tensor,
    z: torch.Tensor,
    beta_l: torch.Tensor,
    jbeta_z: torch.Tensor | None,
    jbeta_base_brownian: torch.Tensor | None = None,
    jbeta_base_guidance0: torch.Tensor | None = None,
    jbeta_full_deterministic: torch.Tensor | None = None,
    jbeta_guidance_rho0: torch.Tensor | None = None,
    gamma_delta_mode: str = "brownian",
    gamma_jvp: bool = True,
    theta: torch.Tensor,
    eta: float,
    lhat_temp: float,
    gate_power: float,
    h_alpha_s: torch.Tensor,
    h_lhat_s: torch.Tensor | None = None,
    lhat: torch.Tensor | None = None,
    linear_baseline_kappa: float = 0.0,
    linear_baseline_kappa_new: float | None = None,
    pa_residual_alpha: float = 1.0,
    jvp_shrink_alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """pA-gate retained-order rho local surrogate terms.

    rho_ratio is the dimensionless ratio r = rho_t / s_t.  The local alpha
    formula is the DDPM-unit version of the 1D Gaussian pA-gate check:

        p_A * (1 - theta)^lambda_s,

    with theta using the raw A/B kernel log-ratio.
    """
    if not torch.is_tensor(rho_ratio):
        rho_ratio = torch.tensor(float(rho_ratio), device=latents.device, dtype=torch.float32)
    rho_ratio = rho_ratio.to(device=latents.device, dtype=torch.float32)
    gamma_delta_mode = str(gamma_delta_mode)

    y = latents.float().flatten(1)
    zf = z.float().flatten(1)
    temp = float(lhat_temp)
    beta = temp * beta_l.float().flatten(1)
    use_gamma_jvp = bool(gamma_jvp)
    if use_gamma_jvp:
        if jbeta_z is None:
            raise ValueError("Gamma JVP is enabled but jbeta_z is unavailable")
        jbeta = temp * jbeta_z.float().flatten(1)
    else:
        jbeta = torch.zeros_like(zf)
    mu_a_f = mu_a.float().flatten(1)
    hmu = forward_displacement.float().flatten(1)

    theta_f = theta.float()
    eta_f = float(eta)
    gate_pow = float(gate_power)
    residual_alpha = float(pa_residual_alpha)
    var_f = variance.float()
    fvar = forward_variance.float()

    nu_a = y - mu_a_f
    beta_dot_z = torch.sum(beta * zf, dim=1)
    baseline_lhat_coeff = float(linear_baseline_kappa) / max(abs(temp), 1e-30)
    if linear_baseline_kappa_new is None:
        linear_baseline_kappa_new = float(linear_baseline_kappa)
    baseline_lhat_coeff_new = float(linear_baseline_kappa_new) / max(
        abs(temp), 1e-30
    )
    residual_gate_coeff = residual_alpha * (
        gate_pow * theta_f[:, None] * eta_f - baseline_lhat_coeff
    )
    psi = -residual_gate_coeff * beta
    q = residual_gate_coeff * beta + rho_ratio * psi

    if gamma_delta_mode == "brownian":
        gamma_delta = zf
        gamma_jbeta_delta = jbeta
    elif gamma_delta_mode == "base_brownian":
        if use_gamma_jvp and jbeta_base_brownian is None:
            raise ValueError("gamma_delta_mode='base_brownian' requires base+Brownian JVP")
        gamma_delta = (mu_a_f - y) + zf
        gamma_jbeta_delta = (
            temp * jbeta_base_brownian.float().flatten(1)
            if use_gamma_jvp
            else torch.zeros_like(gamma_delta)
        )
    elif gamma_delta_mode == "base_guidance0":
        if use_gamma_jvp and jbeta_base_guidance0 is None:
            raise ValueError("gamma_delta_mode='base_guidance0' requires base+Brownian+guidance_rho0 JVP")
        base_plus_brownian = (mu_a_f - y) + zf
        guidance_rho0 = _broadcast_particle_scalar(var_f, y) * psi
        gamma_delta = base_plus_brownian + guidance_rho0
        gamma_jbeta_delta = (
            temp * jbeta_base_guidance0.float().flatten(1)
            if use_gamma_jvp
            else torch.zeros_like(gamma_delta)
        )
    elif gamma_delta_mode == "full_deterministic":
        if use_gamma_jvp and jbeta_full_deterministic is None:
            raise ValueError("gamma_delta_mode='full_deterministic' requires base+Brownian+guidance_rho0 JVP")
        guidance_rho0 = _broadcast_particle_scalar(var_f, y) * psi
        gamma_delta = (mu_a_f - y) + zf + guidance_rho0
        gamma_jbeta_delta = (
            temp * jbeta_full_deterministic.float().flatten(1)
            if use_gamma_jvp
            else torch.zeros_like(gamma_delta)
        )
    elif gamma_delta_mode == "full":
        if use_gamma_jvp and (jbeta_base_brownian is None or jbeta_guidance_rho0 is None):
            raise ValueError("gamma_delta_mode='full' requires base+Brownian and guidance JVPs")
        base_plus_brownian = (mu_a_f - y) + zf
        guidance_rho0 = _broadcast_particle_scalar(var_f, y) * psi
        gamma_delta = base_plus_brownian + (1.0 - rho_ratio) * guidance_rho0
        gamma_jbeta_delta = (
            (
                temp * jbeta_base_brownian.float().flatten(1)
                + (1.0 - rho_ratio) * temp * jbeta_guidance_rho0.float().flatten(1)
            )
            if use_gamma_jvp
            else torch.zeros_like(gamma_delta)
        )
    else:
        raise ValueError(f"Unknown gamma_delta_mode={gamma_delta_mode!r}")

    beta_dot_gamma_delta = torch.sum(beta * gamma_delta, dim=1)
    quad_hpsi_rank1 = (
        rho_ratio
        * residual_alpha
        * gate_pow
        * (eta_f**2)
        * theta_f
        * (1.0 - theta_f)
        * beta_dot_gamma_delta.pow(2)
    )
    quad_hpsi_jvp_raw = (
        rho_ratio
        * residual_gate_coeff.squeeze(1)
        * torch.sum(gamma_delta * gamma_jbeta_delta, dim=1)
    )
    # The fixed-noise quadratic-form approximation is substantially noisier
    # for a coarse reverse-time discretization.  Apply one predeclared,
    # ESS-independent shrinkage factor at the source so rho selection and the
    # propagated weight use exactly the same approximation.  This is not a
    # bridge: no omitted residual is carried into later particle weights.
    quad_hpsi_jvp = float(jvp_shrink_alpha) * quad_hpsi_jvp_raw
    gamma_jbeta_inner = torch.sum(gamma_delta * gamma_jbeta_delta, dim=1)
    gamma_delta_norm = torch.linalg.vector_norm(gamma_delta, dim=1)
    gamma_jbeta_delta_norm = torch.linalg.vector_norm(gamma_jbeta_delta, dim=1)
    gamma_jbeta_cos = gamma_jbeta_inner / (
        gamma_delta_norm * gamma_jbeta_delta_norm
    ).clamp_min(1e-20)
    jvp_scale = rho_ratio * residual_gate_coeff.squeeze(1)
    # The reference reverse kernel and the proposal base kernel match at
    # rho=0, so the particle-dependent base normalization term is zero.
    nu0 = nu_a
    alpha_base = torch.zeros_like(theta_f)
    alpha_forward_linear = -rho_ratio * torch.sum(hmu * psi, dim=1)
    alpha_forward_quadratic = -0.5 * (rho_ratio**2) * fvar * torch.sum(psi * psi, dim=1)
    alpha_reverse_linear = torch.sum(nu0 * q, dim=1)
    alpha_reverse_quadratic = 0.5 * var_f * torch.sum(q * q, dim=1)
    alpha_gate_time = -gate_pow * theta_f * h_alpha_s.float()
    if float(linear_baseline_kappa) != 0.0:
        if h_lhat_s is None:
            raise ValueError("Centered PA baseline requires h_lhat_s")
        alpha_gate_time = (
            alpha_gate_time
            + baseline_lhat_coeff * h_lhat_s.float()
        )
    baseline_schedule_delta = baseline_lhat_coeff_new - baseline_lhat_coeff
    if abs(float(baseline_schedule_delta)) > 0.0:
        if lhat is None:
            raise ValueError("Scheduled centered PA baseline requires lhat")
        # With Delta-kappa=O(h), the omitted product
        # Delta-kappa * Delta-lhat is O(h^(3/2)) under the retained local
        # order.  The operational hybrid path replaces this entire gate block
        # by the exact finite increment.
        alpha_gate_time = (
            alpha_gate_time
            + baseline_schedule_delta * lhat.float()
        )
    alpha_gate_time = residual_alpha * alpha_gate_time
    lam = (
        residual_alpha
        * gate_pow
        * 0.5
        * theta_f
        * (1.0 - theta_f)
        * (eta_f**2)
    )
    # The base target has total reference exponent one, so common Gaussian
    # quadratic parts of L and K cancel against the proposal kernels at
    # retained order.
    quad_variance_schedule = torch.zeros_like(alpha_base)
    quad_mu_jac = torch.zeros_like(alpha_base)
    quad_hpsi = quad_hpsi_rank1 + quad_hpsi_jvp
    quad_gate_z = -lam * beta_dot_gamma_delta.pow(2)
    quad_gamma_rank1 = quad_hpsi_rank1 + quad_gate_z
    quad_gamma_total = quad_hpsi_jvp + quad_gamma_rank1

    total = (
        alpha_base
        + alpha_forward_linear
        + alpha_forward_quadratic
        + alpha_reverse_linear
        + alpha_reverse_quadratic
        + alpha_gate_time
        + quad_variance_schedule
        + quad_mu_jac
        + quad_hpsi
        + quad_gate_z
    )
    return {
        "alpha_base": alpha_base,
        "alpha_forward_linear": alpha_forward_linear,
        "alpha_forward_quadratic": alpha_forward_quadratic,
        "alpha_reverse_linear": alpha_reverse_linear,
        "alpha_reverse_quadratic": alpha_reverse_quadratic,
        "alpha_gate_time": alpha_gate_time,
        "quad_variance_schedule": quad_variance_schedule,
        "quad_mu_jac": quad_mu_jac,
        "quad_hpsi": quad_hpsi,
        "quad_hpsi_rank1": quad_hpsi_rank1,
        "quad_hpsi_jvp": quad_hpsi_jvp,
        "quad_hpsi_jvp_raw": quad_hpsi_jvp_raw,
        "quad_gate_z": quad_gate_z,
        "quad_gamma_rank1": quad_gamma_rank1,
        "quad_gamma_total": quad_gamma_total,
        "beta_dot_z": beta_dot_z,
        "beta_dot_gamma_delta": beta_dot_gamma_delta,
        "theta": theta_f,
        "psi_norm": torch.linalg.vector_norm(psi, dim=1),
        "q_norm": torch.linalg.vector_norm(q, dim=1),
        "nu0_norm": torch.linalg.vector_norm(nu0, dim=1),
        "jbeta_z_norm": torch.linalg.vector_norm(jbeta, dim=1),
        "gamma_delta_norm": gamma_delta_norm,
        "jbeta_gamma_delta_norm": gamma_jbeta_delta_norm,
        "gamma_jbeta_inner": gamma_jbeta_inner,
        "gamma_jbeta_cos": gamma_jbeta_cos,
        "jvp_scale": jvp_scale,
        "total": total,
    }


def local_surrogate_logw(
    rho_ratio: torch.Tensor | float,
    **kwargs,
) -> torch.Tensor:
    """Retained-order rho local surrogate log weight in DDPM latent units."""
    return local_surrogate_terms(rho_ratio, **kwargs)["total"]


SURROGATE_DECOMP_NAMES = (
    "forward_linear",
    "forward_quadratic",
    "reverse_linear",
    "reverse_quadratic",
    "gate_time",
    "gate_curvature",
    "proposal_hessian",
    "proposal_hessian_rank1",
    "proposal_hessian_jvp",
    "proposal_hessian_jvp_raw",
    "gamma_rank1",
    "gamma_quadratic",
    "group_forward_kernel_alpha",
    "group_reverse_kernel_alpha",
    "group_kernel_alpha",
    "group_gate",
    "gate_retained",
    "reverse_retained",
    "gate_exact",
    "reverse_exact",
    "group_quadratic",
    "total",
)

FINITE_DECOMP_NAMES = (
    "theta",
    "reverse_kernel",
    "forward_kernel",
    "total",
)

MATH_COMPONENT_TARGETS = (
    ("s_local", "s_exact"),
    ("gate_taylor2_exact_s", "theta"),
    ("gate_taylor2_local_s", "theta"),
    ("gate_retained_delta", "theta"),
    ("gate_retained_brownian", "theta"),
    ("reverse_raw_delta", "reverse_kernel"),
    ("forward_alpha", "forward_kernel"),
    ("forward_alpha_hessian", "forward_kernel"),
)

MATH_TOTAL_NAMES = (
    "total_gate_taylor2_exact_s",
    "total_gate_taylor2_local_s",
    "total_gate_retained_delta",
    "total_gate_retained_brownian",
    "total_forward_alpha",
    "total_forward_alpha_hessian",
)

MATH_STANDALONE_NAMES = (
    "s_exact",
    "s_local",
    "s_local_error",
    "linear_gate_reverse_delta",
    "linear_gate_reverse_brownian",
)

MATH_AUX_NAMES = (
    "gate_linear_delta",
    "gate_linear_brownian",
    "reverse_linear_delta",
    "reverse_linear_brownian",
    "reverse_retained_alpha",
)

MATH_ALL_NAMES = tuple(
    dict.fromkeys(
        list(MATH_STANDALONE_NAMES)
        + [name for name, _ in MATH_COMPONENT_TARGETS]
        + list(MATH_TOTAL_NAMES)
        + list(MATH_AUX_NAMES)
    )
)


def surrogate_decomposition_from_terms(terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Grouped retained-order log-weight terms for variance diagnostics."""
    forward_linear = terms["alpha_forward_linear"]
    forward_quadratic = terms["alpha_forward_quadratic"]
    reverse_linear = terms["alpha_reverse_linear"]
    reverse_quadratic = terms["alpha_reverse_quadratic"]
    gate_time = terms["alpha_gate_time"]
    gate_curvature = terms["quad_gate_z"]
    proposal_hessian = terms["quad_hpsi"]
    proposal_hessian_rank1 = terms["quad_hpsi_rank1"]
    proposal_hessian_jvp = terms["quad_hpsi_jvp"]
    proposal_hessian_jvp_raw = terms["quad_hpsi_jvp_raw"]
    gamma_rank1 = terms["quad_gamma_rank1"]
    gamma_quadratic = terms["quad_gamma_total"]
    group_forward_kernel_alpha = forward_linear + forward_quadratic
    group_reverse_kernel_alpha = reverse_linear + reverse_quadratic
    group_kernel_alpha = (
        group_forward_kernel_alpha
        + group_reverse_kernel_alpha
    )
    group_gate = gate_time + gate_curvature
    group_quadratic = forward_quadratic + reverse_quadratic + gate_curvature + proposal_hessian
    out = {
        "forward_linear": forward_linear,
        "forward_quadratic": forward_quadratic,
        "reverse_linear": reverse_linear,
        "reverse_quadratic": reverse_quadratic,
        "gate_time": gate_time,
        "gate_curvature": gate_curvature,
        "proposal_hessian": proposal_hessian,
        "proposal_hessian_rank1": proposal_hessian_rank1,
        "proposal_hessian_jvp": proposal_hessian_jvp,
        "proposal_hessian_jvp_raw": proposal_hessian_jvp_raw,
        "gamma_rank1": gamma_rank1,
        "gamma_quadratic": gamma_quadratic,
        "group_forward_kernel_alpha": group_forward_kernel_alpha,
        "group_reverse_kernel_alpha": group_reverse_kernel_alpha,
        "group_kernel_alpha": group_kernel_alpha,
        "group_gate": group_gate,
        "group_quadratic": group_quadratic,
        "total": terms["total"],
    }
    for src, dst in (
        ("beta_dot_z", "diag_beta_dot_z"),
        ("beta_dot_gamma_delta", "diag_beta_dot_gamma_delta"),
        ("psi_norm", "diag_psi_norm"),
        ("q_norm", "diag_q_norm"),
        ("nu0_norm", "diag_nu0_norm"),
        ("jbeta_z_norm", "diag_jbeta_z_norm"),
        ("gamma_delta_norm", "diag_gamma_delta_norm"),
        ("jbeta_gamma_delta_norm", "diag_jbeta_gamma_delta_norm"),
        ("gamma_jbeta_inner", "diag_gamma_jbeta_inner"),
        ("gamma_jbeta_cos", "diag_gamma_jbeta_cos"),
        ("jvp_scale", "diag_jvp_scale"),
    ):
        if src in terms:
            out[dst] = terms[src]
    return out


def surrogate_coefficients(**kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e0 = local_surrogate_logw(0.0, **kwargs)
    e1 = local_surrogate_logw(1.0, **kwargs)
    em1 = local_surrogate_logw(-1.0, **kwargs)
    a = 0.5 * (e1 + em1) - e0
    b = 0.5 * (e1 - em1)
    c = e0
    return a.detach(), b.detach(), c.detach()


def surrogate_cubic_coefficients(**kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return c0..c3 for the full-Delta Gamma expansion ell(r)=sum_k c_k r^k."""
    e0 = local_surrogate_logw(0.0, **kwargs)
    e1 = local_surrogate_logw(1.0, **kwargs)
    em1 = local_surrogate_logw(-1.0, **kwargs)
    e2 = local_surrogate_logw(2.0, **kwargs)
    c0 = e0
    c2 = 0.5 * (e1 + em1) - e0
    c1_plus_c3 = 0.5 * (e1 - em1)
    two_c1_plus_eight_c3 = e2 - c0 - 4.0 * c2
    c3 = (two_c1_plus_eight_c3 - 2.0 * c1_plus_c3) / 6.0
    c1 = c1_plus_c3 - c3
    return c0.detach(), c1.detach(), c2.detach(), c3.detach()


def evaluate_surrogate_polynomial(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    rho: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(rho):
        device = coeffs_low_to_high[0].device
        rho = torch.tensor(float(rho), device=device, dtype=torch.float32)
    rr = rho.to(device=coeffs_low_to_high[0].device, dtype=torch.float32)
    out = torch.zeros_like(coeffs_low_to_high[0].float())
    for coeff in reversed(coeffs_low_to_high):
        out = out * rr + coeff.float()
    return out


def evaluate_surrogate_polynomial_with_jvp_clip(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    rho: torch.Tensor | float,
    *,
    surrogate_kwargs: dict[str, Any] | None,
    jvp_clip: float,
    jvp_rank_topk: int = 0,
    adaptive_rank_clip_jvp: bool = False,
    adaptive_rank_clip_jvp_ess_thresholds: str = "0.97,0.93,0.88",
    adaptive_rank_clip_jvp_topks: str = "0,2,4,8",
    center_mode: str = "median",
    count_eps: float = 1e-6,
    world_size: int,
    global_start: int,
    local_n: int,
) -> torch.Tensor:
    """Evaluate ell(rho), optionally replacing the JVP addend by its clipped value.

    This is used only for rho selection.  The final sampled log weight still
    goes through raw_clip_surrogate_terms_global, so the objective and the
    propagated surrogate use the same JVP robustification.
    """
    inc = evaluate_surrogate_polynomial(coeffs_low_to_high, rho).float()
    if float(jvp_clip) <= 0.0 and int(jvp_rank_topk) <= 0 and not bool(adaptive_rank_clip_jvp):
        return inc
    if surrogate_kwargs is None:
        return inc

    terms = local_surrogate_terms(rho, **surrogate_kwargs)
    jvp_term = terms.get("quad_hpsi_jvp")
    if jvp_term is None:
        return inc

    local_inc = inc.reshape(-1).contiguous()
    global_inc = all_gather_cat(local_inc, world_size).float()
    global_jvp = all_gather_cat(jvp_term.float().reshape(-1).contiguous(), world_size).float()
    effective_rank_topk = adaptive_rank_clip_topk_from_ess(
        global_ess_frac_from_logw(global_inc),
        enabled=bool(adaptive_rank_clip_jvp),
        fallback_topk=int(jvp_rank_topk),
        thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
        topks=str(adaptive_rank_clip_jvp_topks),
    )
    if int(effective_rank_topk) > 0:
        _, residual, _, _, _ = _global_centered_rank_clip(
            global_jvp,
            int(effective_rank_topk),
            center_mode=center_mode,
            count_eps=float(count_eps),
        )
    else:
        _, residual, _, _ = _global_centered_clip(
            global_jvp,
            float(jvp_clip),
            center_mode=center_mode,
            count_eps=float(count_eps),
        )
    global_clipped = global_inc - residual
    local_slice = slice(global_start, global_start + local_n)
    return global_clipped[local_slice].to(device=inc.device, dtype=torch.float32).reshape_as(inc)


def _polynomial_coefficients_from_samples(
    samples: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Recover low-to-high degree-2/3 coefficients from fixed-node samples."""
    if len(samples) not in {3, 4}:
        raise ValueError(f"Expected 3 or 4 polynomial samples, got {len(samples)}")
    e0, e1, em1 = (value.float() for value in samples[:3])
    c0 = e0
    c2 = 0.5 * (e1 + em1) - e0
    c1_plus_c3 = 0.5 * (e1 - em1)
    if len(samples) == 3:
        return c0, c1_plus_c3, c2
    e2 = samples[3].float()
    two_c1_plus_eight_c3 = e2 - c0 - 4.0 * c2
    c3 = (two_c1_plus_eight_c3 - 2.0 * c1_plus_c3) / 6.0
    c1 = c1_plus_c3 - c3
    return c0, c1, c2, c3


def prepare_global_rho_ess_state(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    carry_logw: torch.Tensor,
    *,
    surrogate_kwargs: dict[str, Any] | None,
    raw_clip_jvp: float,
    rank_clip_jvp_topk: int,
    adaptive_rank_clip_jvp: bool,
    raw_clip_jvp_in_rho_objective: bool,
    world_size: int,
) -> dict[str, Any]:
    """Gather everything needed by the rho ESS search exactly once.

    The previous implementation gathered the surrogate and JVP vectors, then
    ran four all-reduces, for every one of 100+ rho candidates.  Because both
    vectors are low-degree polynomials in rho, gathering their coefficients
    once is algebraically equivalent and removes the synchronization-heavy
    inner loop.
    """
    local_coeffs = tuple(value.detach().float().reshape(-1) for value in coeffs_low_to_high)
    local_carry = carry_logw.detach().float().reshape(-1)
    use_jvp_clip = bool(
        raw_clip_jvp_in_rho_objective
        and (
            float(raw_clip_jvp) > 0.0
            or int(rank_clip_jvp_topk) > 0
            or bool(adaptive_rank_clip_jvp)
        )
        and surrogate_kwargs is not None
    )

    local_jvp_coeffs: tuple[torch.Tensor, ...] = ()
    if use_jvp_clip:
        nodes = (0.0, 1.0, -1.0) if len(local_coeffs) == 3 else (0.0, 1.0, -1.0, 2.0)
        jvp_samples = tuple(
            local_surrogate_terms(rho, **surrogate_kwargs)["quad_hpsi_jvp"]
            .detach()
            .float()
            .reshape(-1)
            for rho in nodes
        )
        local_jvp_coeffs = _polynomial_coefficients_from_samples(jvp_samples)

    gathered = all_gather_particle_rows(
        torch.stack([local_carry, *local_coeffs, *local_jvp_coeffs], dim=0),
        int(world_size),
    ).float()
    coeff_start = 1
    coeff_end = coeff_start + len(local_coeffs)
    jvp_end = coeff_end + len(local_jvp_coeffs)
    return {
        "carry_logw": gathered[0],
        "coeffs": tuple(gathered[index] for index in range(coeff_start, coeff_end)),
        "jvp_coeffs": tuple(gathered[index] for index in range(coeff_end, jvp_end)),
    }


def rho_ess_from_prepared_global_state(
    prepared: dict[str, Any],
    rho: float,
    *,
    carry_logw: torch.Tensor | None = None,
    surrogate_logw_clip: float = 0.0,
    raw_clip_jvp: float = 0.0,
    rank_clip_jvp_topk: int = 0,
    adaptive_rank_clip_jvp: bool = False,
    adaptive_rank_clip_jvp_ess_thresholds: str = "0.97,0.93,0.88",
    adaptive_rank_clip_jvp_topks: str = "0,2,4,8",
    raw_clip_center: str = "median",
    raw_clip_count_eps: float = 1e-6,
) -> float:
    """Evaluate global ESS/N without a distributed collective."""
    global_coeffs = tuple(prepared["coeffs"])
    inc = evaluate_surrogate_polynomial(global_coeffs, float(rho)).float()
    global_jvp_coeffs = tuple(prepared.get("jvp_coeffs", ()))
    if global_jvp_coeffs:
        jvp_term = evaluate_surrogate_polynomial(global_jvp_coeffs, float(rho)).float()
        effective_rank_topk = adaptive_rank_clip_topk_from_ess(
            global_ess_frac_from_logw(inc),
            enabled=bool(adaptive_rank_clip_jvp),
            fallback_topk=int(rank_clip_jvp_topk),
            thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
            topks=str(adaptive_rank_clip_jvp_topks),
        )
        if int(effective_rank_topk) > 0:
            _, residual, _, _, _ = _global_centered_rank_clip(
                jvp_term,
                int(effective_rank_topk),
                center_mode=str(raw_clip_center),
                count_eps=float(raw_clip_count_eps),
            )
        else:
            _, residual, _, _ = _global_centered_clip(
                jvp_term,
                float(raw_clip_jvp),
                center_mode=str(raw_clip_center),
                count_eps=float(raw_clip_count_eps),
            )
        inc = inc - residual

    inc = clip_final_surrogate_logw(inc, float(surrogate_logw_clip))
    carry = prepared["carry_logw"] if carry_logw is None else carry_logw
    logw = torch.nan_to_num(
        carry.detach().float().reshape(-1) + inc.reshape(-1),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    )
    log_z1 = torch.logsumexp(logw, dim=0)
    log_z2 = torch.logsumexp(2.0 * logw, dim=0)
    ess = torch.exp(2.0 * log_z1 - log_z2) / max(int(logw.numel()), 1)
    return float(ess.clamp(0.0, 1.0).item())


def global_weighted_poly_objective(
    rho: float,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
    over1_penalty: float = 0.0,
) -> tuple[float, float]:
    """Return J_var + alpha max(0, rho - 1)^2 and derivative."""
    r = torch.tensor(float(rho), device=a.device, dtype=torch.float64)
    aa = a.double()
    bb = b.double()
    cc = c.double()
    ww = w.double()
    ell = aa * r * r + bb * r + cc
    dell = 2.0 * aa * r + bb
    local = torch.stack(
        [
            torch.sum(ww * ell),
            torch.sum(ww * ell * ell),
            torch.sum(ww * dell),
            torch.sum(ww * ell * dell),
        ]
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    mean, second, dmean, dell_mean = local
    var = (second - mean * mean).clamp_min(0.0)
    grad = 2.0 * (dell_mean - mean * dmean)
    alpha = float(over1_penalty)
    if alpha > 0.0 and float(rho) > 1.0:
        excess = float(rho) - 1.0
        var = var + alpha * excess * excess
        grad = grad + 2.0 * alpha * excess
    return float(var.item()), float(grad.item())


def global_weighted_poly_variance_coefficients(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
) -> tuple[float, float, float, float, float]:
    """Return coefficients of Var_w[a rho^2 + b rho + c]."""
    aa = a.double()
    bb = b.double()
    cc = c.double()
    ww = w.double()
    local = torch.stack(
        [
            torch.sum(ww * aa),
            torch.sum(ww * bb),
            torch.sum(ww * cc),
            torch.sum(ww * aa * aa),
            torch.sum(ww * aa * bb),
            torch.sum(ww * aa * cc),
            torch.sum(ww * bb * bb),
            torch.sum(ww * bb * cc),
            torch.sum(ww * cc * cc),
        ]
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    ea, eb, ec, ea2, eab, eac, eb2, ebc, ec2 = [float(x.item()) for x in local]
    p4 = ea2 - ea * ea
    p3 = 2.0 * (eab - ea * eb)
    p2 = eb2 + 2.0 * eac - eb * eb - 2.0 * ea * ec
    p1 = 2.0 * (ebc - eb * ec)
    p0 = ec2 - ec * ec
    return p4, p3, p2, p1, p0


def rho_variance_poly_value(
    coeffs: tuple[float, float, float, float, float],
    rho: float,
    *,
    over1_penalty: float = 0.0,
) -> float:
    p4, p3, p2, p1, p0 = coeffs
    r = float(rho)
    val = (((p4 * r + p3) * r + p2) * r + p1) * r + p0
    alpha = float(over1_penalty)
    if alpha > 0.0 and r > 1.0:
        val += alpha * (r - 1.0) * (r - 1.0)
    return float(max(val, 0.0))


def rho_variance_poly_grad(
    coeffs: tuple[float, float, float, float, float],
    rho: float,
    *,
    over1_penalty: float = 0.0,
) -> float:
    p4, p3, p2, p1, _ = coeffs
    r = float(rho)
    grad = ((4.0 * p4 * r + 3.0 * p3) * r + 2.0 * p2) * r + p1
    alpha = float(over1_penalty)
    if alpha > 0.0 and r > 1.0:
        grad += 2.0 * alpha * (r - 1.0)
    return float(grad)


def real_polynomial_roots(coeffs: list[float], *, tol: float = 1e-8) -> list[float]:
    """Real roots of a low-degree polynomial with leading zeros removed."""
    cleaned = list(coeffs)
    while cleaned and abs(cleaned[0]) <= tol:
        cleaned.pop(0)
    if len(cleaned) <= 1:
        return []
    roots = np.roots(np.asarray(cleaned, dtype=float))
    out: list[float] = []
    for root in roots:
        if abs(float(np.imag(root))) <= tol * max(1.0, abs(float(np.real(root)))):
            out.append(float(np.real(root)))
    return out


def exact_minimize_rho_variance(
    coeffs: tuple[float, float, float, float, float],
    *,
    rho_cap: float,
    over1_penalty: float,
    rho_init: float,
) -> tuple[float, float, float, int, int]:
    """Minimize the one-dimensional quartic surrogate variance over the rho cap."""
    p4, p3, p2, p1, _ = coeffs
    candidates = [float(rho_init), 0.0]
    cap_enabled = rho_cap is not None and float(rho_cap) >= 0.0
    lo = -float(rho_cap) if cap_enabled else -float("inf")
    hi = float(rho_cap) if cap_enabled else float("inf")
    if cap_enabled:
        candidates.extend([lo, hi])

    base_roots = real_polynomial_roots([4.0 * p4, 3.0 * p3, 2.0 * p2, p1])
    stationary = 0
    alpha = float(over1_penalty)
    for root in base_roots:
        if alpha <= 0.0 or root <= 1.0 + 1e-10:
            candidates.append(root)
            stationary += 1

    if alpha > 0.0:
        penalized_roots = real_polynomial_roots(
            [4.0 * p4, 3.0 * p3, 2.0 * (p2 + alpha), p1 - 2.0 * alpha]
        )
        for root in penalized_roots:
            if root >= 1.0 - 1e-10:
                candidates.append(root)
                stationary += 1
        candidates.append(1.0)
    best_rho = float(rho_init)
    best_obj = float("inf")
    unique: list[float] = []
    for cand in candidates:
        if not math.isfinite(cand):
            continue
        r = max(lo, min(hi, float(cand)))
        if any(abs(r - seen) <= 1e-10 * max(1.0, abs(r), abs(seen)) for seen in unique):
            continue
        unique.append(r)
        obj = rho_variance_poly_value(coeffs, r, over1_penalty=over1_penalty)
        if math.isfinite(obj) and obj < best_obj:
            best_rho = r
            best_obj = obj
    best_grad = rho_variance_poly_grad(coeffs, best_rho, over1_penalty=over1_penalty)
    return best_rho, best_obj, best_grad, len(unique), stationary


def global_weighted_polynomial_variance_coefficients(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    w: torch.Tensor,
) -> list[float]:
    """Variance polynomial coefficients for ell(r)=sum_k coeff[k] r^k."""
    coeffs = [c.detach().double() for c in coeffs_low_to_high]
    ww = w.detach().double()
    degree = len(coeffs) - 1
    means_local = [torch.sum(ww * c) for c in coeffs]
    second_local = []
    for power in range(2 * degree + 1):
        acc = torch.zeros((), device=ww.device, dtype=torch.float64)
        for i in range(degree + 1):
            j = power - i
            if 0 <= j <= degree:
                acc = acc + torch.sum(ww * coeffs[i] * coeffs[j])
        second_local.append(acc)
    packed = torch.stack(means_local + second_local)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    means = [float(x.item()) for x in packed[: degree + 1]]
    seconds = [float(x.item()) for x in packed[degree + 1 :]]

    var_coeffs: list[float] = []
    for power in range(2 * degree + 1):
        mean_sq_coeff = 0.0
        for i in range(degree + 1):
            j = power - i
            if 0 <= j <= degree:
                mean_sq_coeff += means[i] * means[j]
        var_coeffs.append(float(seconds[power] - mean_sq_coeff))
    return var_coeffs


def polynomial_value(coeffs_low_to_high: list[float] | tuple[float, ...], rho: float) -> float:
    val = 0.0
    r = float(rho)
    for coeff in reversed(coeffs_low_to_high):
        val = val * r + float(coeff)
    return float(val)


def polynomial_grad(coeffs_low_to_high: list[float] | tuple[float, ...], rho: float) -> float:
    if len(coeffs_low_to_high) <= 1:
        return 0.0
    deriv = [i * float(coeffs_low_to_high[i]) for i in range(1, len(coeffs_low_to_high))]
    return polynomial_value(deriv, rho)


def polynomial_variance_value(
    coeffs_low_to_high: list[float] | tuple[float, ...],
    rho: float,
    *,
    over1_penalty: float = 0.0,
) -> float:
    r = float(rho)
    val = polynomial_value(coeffs_low_to_high, r)
    alpha = float(over1_penalty)
    if alpha > 0.0 and r > 1.0:
        val += alpha * (r - 1.0) * (r - 1.0)
    return float(max(val, 0.0))


def polynomial_variance_grad(
    coeffs_low_to_high: list[float] | tuple[float, ...],
    rho: float,
    *,
    over1_penalty: float = 0.0,
) -> float:
    r = float(rho)
    grad = polynomial_grad(coeffs_low_to_high, r)
    alpha = float(over1_penalty)
    if alpha > 0.0 and r > 1.0:
        grad += 2.0 * alpha * (r - 1.0)
    return float(grad)


def exact_minimize_polynomial_variance(
    coeffs_low_to_high: list[float],
    *,
    rho_cap: float,
    over1_penalty: float,
    rho_init: float,
) -> tuple[float, float, float, int, int]:
    """Exact scalar minimization of a variance polynomial over the rho cap."""
    candidates = [float(rho_init), 0.0]
    cap_enabled = rho_cap is not None and float(rho_cap) >= 0.0
    lo = -float(rho_cap) if cap_enabled else -float("inf")
    hi = float(rho_cap) if cap_enabled else float("inf")
    if cap_enabled:
        candidates.extend([lo, hi])

    deriv_low = [i * float(coeffs_low_to_high[i]) for i in range(1, len(coeffs_low_to_high))]
    base_roots = real_polynomial_roots(list(reversed(deriv_low)))
    stationary = 0
    alpha = float(over1_penalty)
    for root in base_roots:
        if alpha <= 0.0 or root <= 1.0 + 1e-10:
            candidates.append(root)
            stationary += 1

    if alpha > 0.0:
        penalized_deriv_low = list(deriv_low)
        while len(penalized_deriv_low) < 2:
            penalized_deriv_low.append(0.0)
        penalized_deriv_low[0] -= 2.0 * alpha
        penalized_deriv_low[1] += 2.0 * alpha
        penalized_roots = real_polynomial_roots(list(reversed(penalized_deriv_low)))
        for root in penalized_roots:
            if root >= 1.0 - 1e-10:
                candidates.append(root)
                stationary += 1
        candidates.append(1.0)

    best_rho = float(rho_init)
    best_obj = float("inf")
    unique: list[float] = []
    for cand in candidates:
        if not math.isfinite(cand):
            continue
        r = max(lo, min(hi, float(cand)))
        if any(abs(r - seen) <= 1e-10 * max(1.0, abs(r), abs(seen)) for seen in unique):
            continue
        unique.append(r)
        obj = polynomial_variance_value(coeffs_low_to_high, r, over1_penalty=over1_penalty)
        if math.isfinite(obj) and obj < best_obj:
            best_rho = r
            best_obj = obj
    best_grad = polynomial_variance_grad(coeffs_low_to_high, best_rho, over1_penalty=over1_penalty)
    return best_rho, best_obj, best_grad, len(unique), stationary


def optimize_rho_global_polynomial(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    *,
    rho_init: float,
    rho_lr: float,
    rho_steps: int,
    rho_cap: float,
    optimizer: str,
    accept_only_improve: bool,
    line_search_halvings: int,
    guard_rho0: bool,
    over1_penalty: float,
) -> dict[str, float]:
    var_coeffs = global_weighted_polynomial_variance_coefficients(coeffs_low_to_high, w)
    start_obj = polynomial_variance_value(var_coeffs, rho_init, over1_penalty=over1_penalty)
    start_grad = polynomial_variance_grad(var_coeffs, rho_init, over1_penalty=over1_penalty)
    zero_obj = polynomial_variance_value(var_coeffs, 0.0, over1_penalty=over1_penalty)
    zero_grad = polynomial_variance_grad(var_coeffs, 0.0, over1_penalty=over1_penalty)

    rho = float(rho_init)
    cur_obj = float(start_obj)
    cur_grad = float(start_grad)
    best_rho = float(rho)
    best_obj = float(start_obj)
    best_grad = float(start_grad)
    selected_rho0 = 0.0
    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    accepted_steps = 0
    backtracks = 0
    line_search_failed = 0.0
    lr_last = 0.0
    exact_candidates = 0
    exact_stationary = 0

    if optimizer == "exact":
        best_rho, best_obj, best_grad, exact_candidates, exact_stationary = exact_minimize_polynomial_variance(
            var_coeffs,
            rho_cap=rho_cap,
            over1_penalty=over1_penalty,
            rho_init=rho_init,
        )
    else:
        for k in range(1, int(rho_steps) + 1):
            grad = cur_grad
            if not math.isfinite(grad):
                break
            if optimizer == "adam":
                m_next = beta1 * m + (1.0 - beta1) * grad
                v_next = beta2 * v + (1.0 - beta2) * grad * grad
                mhat = m_next / (1.0 - beta1**k)
                vhat = v_next / (1.0 - beta2**k)
                direction = mhat / (math.sqrt(vhat) + 1e-12)
            else:
                m_next = m
                v_next = v
                direction = grad

            lr_try = float(rho_lr)
            step_accepted = False
            for _ in range(max(0, int(line_search_halvings)) + 1):
                rho_candidate = rho - lr_try * direction
                if rho_cap is not None and rho_cap >= 0:
                    rho_candidate = max(-float(rho_cap), min(float(rho_cap), rho_candidate))
                cand_obj = polynomial_variance_value(
                    var_coeffs,
                    rho_candidate,
                    over1_penalty=over1_penalty,
                )
                cand_grad = polynomial_variance_grad(
                    var_coeffs,
                    rho_candidate,
                    over1_penalty=over1_penalty,
                )
                if math.isfinite(cand_obj) and cand_obj <= cur_obj + 1e-14:
                    rho = float(rho_candidate)
                    cur_obj = float(cand_obj)
                    cur_grad = float(cand_grad)
                    if cur_obj < best_obj:
                        best_rho = float(rho)
                        best_obj = float(cur_obj)
                        best_grad = float(cur_grad)
                    m = m_next
                    v = v_next
                    lr_last = lr_try
                    accepted_steps += 1
                    step_accepted = True
                    break
                lr_try *= 0.5
                backtracks += 1

            if not step_accepted:
                line_search_failed = 1.0
                break

    rho = best_rho
    final_obj = best_obj
    final_grad = best_grad
    if guard_rho0 and math.isfinite(zero_obj) and zero_obj <= final_obj + 1e-14:
        rho = 0.0
        final_obj = float(zero_obj)
        final_grad = float(zero_grad)
        selected_rho0 = 1.0

    accepted = 1.0
    if accept_only_improve and final_obj > start_obj:
        rho = float(rho_init)
        final_obj = start_obj
        final_grad = start_grad
        accepted = 0.0

    return {
        "rho": float(rho),
        "objective_start": float(start_obj),
        "objective_final": float(final_obj),
        "objective_rho0": float(zero_obj),
        "grad_start": float(start_grad),
        "grad_final": float(final_grad),
        "grad_rho0": float(zero_grad),
        "accepted": accepted,
        "selected_rho0": float(selected_rho0),
        "accepted_steps": float(accepted_steps),
        "backtracks": float(backtracks),
        "lr_last": float(lr_last),
        "line_search_failed": float(line_search_failed),
        "exact_candidates": float(exact_candidates),
        "exact_stationary": float(exact_stationary),
        "over1_penalty": float(over1_penalty),
        "surrogate_poly_degree": float(len(coeffs_low_to_high) - 1),
    }


def rho_total_ess_from_polynomial(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    carry_logw: torch.Tensor,
    rho: float,
    *,
    surrogate_logw_clip: float = 0.0,
    raw_clip_jvp: float = 0.0,
    rank_clip_jvp_topk: int = 0,
    adaptive_rank_clip_jvp: bool = False,
    adaptive_rank_clip_jvp_ess_thresholds: str = "0.97,0.93,0.88",
    adaptive_rank_clip_jvp_topks: str = "0,2,4,8",
    raw_clip_center: str = "median",
    raw_clip_count_eps: float = 1e-6,
    raw_clip_jvp_in_rho_objective: bool = False,
    surrogate_kwargs: dict[str, Any] | None = None,
    world_size: int = 1,
    global_start: int = 0,
    local_n: int | None = None,
    prepared_global_state: dict[str, Any] | None = None,
) -> float:
    """ESS/N of carry + current surrogate increment at a candidate rho."""
    if prepared_global_state is not None:
        prepared_carry = prepared_global_state["carry_logw"]
        carry_for_eval = carry_logw.detach().float().reshape(-1)
        if int(carry_for_eval.numel()) != int(prepared_carry.numel()):
            carry_for_eval = (
                torch.zeros_like(prepared_carry)
                if bool(torch.count_nonzero(carry_for_eval).item() == 0)
                else prepared_carry
            )
        return rho_ess_from_prepared_global_state(
            prepared_global_state,
            float(rho),
            carry_logw=carry_for_eval,
            surrogate_logw_clip=float(surrogate_logw_clip),
            raw_clip_jvp=float(raw_clip_jvp),
            rank_clip_jvp_topk=int(rank_clip_jvp_topk),
            adaptive_rank_clip_jvp=bool(adaptive_rank_clip_jvp),
            adaptive_rank_clip_jvp_ess_thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
            adaptive_rank_clip_jvp_topks=str(adaptive_rank_clip_jvp_topks),
            raw_clip_center=str(raw_clip_center),
            raw_clip_count_eps=float(raw_clip_count_eps),
        )
    if local_n is None:
        local_n = int(carry_logw.numel())
    if raw_clip_jvp_in_rho_objective and (
        float(raw_clip_jvp) > 0.0 or int(rank_clip_jvp_topk) > 0 or bool(adaptive_rank_clip_jvp)
    ):
        inc = evaluate_surrogate_polynomial_with_jvp_clip(
            coeffs_low_to_high,
            float(rho),
            surrogate_kwargs=surrogate_kwargs,
            jvp_clip=float(raw_clip_jvp),
            jvp_rank_topk=int(rank_clip_jvp_topk),
            adaptive_rank_clip_jvp=bool(adaptive_rank_clip_jvp),
            adaptive_rank_clip_jvp_ess_thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
            adaptive_rank_clip_jvp_topks=str(adaptive_rank_clip_jvp_topks),
            center_mode=str(raw_clip_center),
            count_eps=float(raw_clip_count_eps),
            world_size=world_size,
            global_start=global_start,
            local_n=int(local_n),
        )
    else:
        inc = evaluate_surrogate_polynomial(coeffs_low_to_high, float(rho))
    inc = clip_final_surrogate_logw(inc, float(surrogate_logw_clip))
    return distributed_ess_frac_from_local_logw(carry_logw.float() + inc.float())


def rho_ess_triplet_from_polynomial(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    carry_logw: torch.Tensor,
    *,
    rho_start: float,
    rho_final: float,
    prefix: str,
    surrogate_logw_clip: float = 0.0,
    raw_clip_jvp: float = 0.0,
    rank_clip_jvp_topk: int = 0,
    adaptive_rank_clip_jvp: bool = False,
    adaptive_rank_clip_jvp_ess_thresholds: str = "0.97,0.93,0.88",
    adaptive_rank_clip_jvp_topks: str = "0,2,4,8",
    raw_clip_center: str = "median",
    raw_clip_count_eps: float = 1e-6,
    raw_clip_jvp_in_rho_objective: bool = False,
    surrogate_kwargs: dict[str, Any] | None = None,
    world_size: int = 1,
    global_start: int = 0,
    local_n: int | None = None,
    prepared_global_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    """ESS/N at the optimizer start, selected rho, and rho=0."""
    common = {
        "surrogate_logw_clip": surrogate_logw_clip,
        "raw_clip_jvp": raw_clip_jvp,
        "rank_clip_jvp_topk": rank_clip_jvp_topk,
        "adaptive_rank_clip_jvp": adaptive_rank_clip_jvp,
        "adaptive_rank_clip_jvp_ess_thresholds": adaptive_rank_clip_jvp_ess_thresholds,
        "adaptive_rank_clip_jvp_topks": adaptive_rank_clip_jvp_topks,
        "raw_clip_center": raw_clip_center,
        "raw_clip_count_eps": raw_clip_count_eps,
        "raw_clip_jvp_in_rho_objective": raw_clip_jvp_in_rho_objective,
        "surrogate_kwargs": surrogate_kwargs,
        "world_size": world_size,
        "global_start": global_start,
        "local_n": local_n,
        "prepared_global_state": prepared_global_state,
    }
    return {
        f"{prefix}_start": rho_total_ess_from_polynomial(
            coeffs_low_to_high,
            carry_logw,
            float(rho_start),
            **common,
        ),
        f"{prefix}_final": rho_total_ess_from_polynomial(
            coeffs_low_to_high,
            carry_logw,
            float(rho_final),
            **common,
        ),
        f"{prefix}_rho0": rho_total_ess_from_polynomial(
            coeffs_low_to_high,
            carry_logw,
            0.0,
            **common,
        ),
    }


def optimize_rho_global_ess_polynomial(
    coeffs_low_to_high: tuple[torch.Tensor, ...],
    carry_logw: torch.Tensor,
    *,
    rho_init: float,
    rho_steps: int,
    rho_cap: float,
    guard_rho0: bool,
    accept_only_improve: bool,
    surrogate_logw_clip: float,
    raw_clip_jvp: float = 0.0,
    rank_clip_jvp_topk: int = 0,
    adaptive_rank_clip_jvp: bool = False,
    adaptive_rank_clip_jvp_ess_thresholds: str = "0.97,0.93,0.88",
    adaptive_rank_clip_jvp_topks: str = "0,2,4,8",
    raw_clip_center: str = "median",
    raw_clip_count_eps: float = 1e-6,
    raw_clip_jvp_in_rho_objective: bool = False,
    surrogate_kwargs: dict[str, Any] | None = None,
    world_size: int = 1,
    global_start: int = 0,
    local_n: int | None = None,
    ess_prefix: str = "rho_total_ess",
    prepared_global_state: dict[str, Any] | None = None,
    ess_search_mode: str = "exact",
    ess_coarse_candidates: int = 11,
) -> dict[str, float]:
    """Maximize ESS/N of carried residual plus the current surrogate log-weight."""
    search_mode = str(ess_search_mode)
    if search_mode == "exact":
        # Backward-compatible name: this is a dense surrogate-only search,
        # not a finite/true model evaluation.
        search_mode = "dense"
    if search_mode not in {"dense", "coarse"}:
        raise ValueError(f"Unknown ESS rho search mode {search_mode!r}")
    cap_enabled = rho_cap is not None and float(rho_cap) >= 0.0
    if cap_enabled:
        lo = -float(rho_cap)
        hi = float(rho_cap)
    else:
        radius = max(5.0, abs(float(rho_init)) + 1.0)
        lo = -radius
        hi = radius

    start_ess = rho_total_ess_from_polynomial(
        coeffs_low_to_high,
        carry_logw,
        float(rho_init),
        surrogate_logw_clip=surrogate_logw_clip,
        raw_clip_jvp=raw_clip_jvp,
        rank_clip_jvp_topk=rank_clip_jvp_topk,
        adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
        adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
        adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
        raw_clip_center=raw_clip_center,
        raw_clip_count_eps=raw_clip_count_eps,
        raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
        surrogate_kwargs=surrogate_kwargs,
        world_size=world_size,
        global_start=global_start,
        local_n=local_n,
        prepared_global_state=prepared_global_state,
    )
    zero_ess = rho_total_ess_from_polynomial(
        coeffs_low_to_high,
        carry_logw,
        0.0,
        surrogate_logw_clip=surrogate_logw_clip,
        raw_clip_jvp=raw_clip_jvp,
        rank_clip_jvp_topk=rank_clip_jvp_topk,
        adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
        adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
        adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
        raw_clip_center=raw_clip_center,
        raw_clip_count_eps=raw_clip_count_eps,
        raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
        surrogate_kwargs=surrogate_kwargs,
        world_size=world_size,
        global_start=global_start,
        local_n=local_n,
        prepared_global_state=prepared_global_state,
    )
    grid_size = (
        max(17, int(rho_steps) + 1)
        if search_mode == "dense"
        else max(3, int(ess_coarse_candidates))
    )
    candidates = np.linspace(lo, hi, grid_size, dtype=float).tolist()
    candidates.extend([float(rho_init), 0.0, lo, hi])

    best_rho = float(rho_init)
    best_ess = float(start_ess)
    unique: list[float] = []
    for cand in candidates:
        if not math.isfinite(float(cand)):
            continue
        r = max(lo, min(hi, float(cand)))
        if any(abs(r - seen) <= 1e-10 * max(1.0, abs(r), abs(seen)) for seen in unique):
            continue
        unique.append(r)
        ess = rho_total_ess_from_polynomial(
            coeffs_low_to_high,
            carry_logw,
            r,
            surrogate_logw_clip=surrogate_logw_clip,
            raw_clip_jvp=raw_clip_jvp,
            rank_clip_jvp_topk=rank_clip_jvp_topk,
            adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
            adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
            adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
            raw_clip_center=raw_clip_center,
            raw_clip_count_eps=raw_clip_count_eps,
            raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
            surrogate_kwargs=surrogate_kwargs,
            world_size=world_size,
            global_start=global_start,
            local_n=local_n,
            prepared_global_state=prepared_global_state,
        )
        if math.isfinite(ess) and ess > best_ess:
            best_rho = r
            best_ess = float(ess)

    sorted_grid = sorted(x for x in unique if lo <= x <= hi)
    if search_mode == "dense" and len(sorted_grid) >= 3:
        best_idx = min(range(len(sorted_grid)), key=lambda i: abs(sorted_grid[i] - best_rho))
        left = sorted_grid[max(0, best_idx - 1)]
        right = sorted_grid[min(len(sorted_grid) - 1, best_idx + 1)]
        if right > left:
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            x1 = right - gr * (right - left)
            x2 = left + gr * (right - left)
            f1 = rho_total_ess_from_polynomial(
                coeffs_low_to_high,
                carry_logw,
                x1,
                surrogate_logw_clip=surrogate_logw_clip,
                raw_clip_jvp=raw_clip_jvp,
                rank_clip_jvp_topk=rank_clip_jvp_topk,
                adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
                adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
                adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
                raw_clip_center=raw_clip_center,
                raw_clip_count_eps=raw_clip_count_eps,
                raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
                surrogate_kwargs=surrogate_kwargs,
                world_size=world_size,
                global_start=global_start,
                local_n=local_n,
                prepared_global_state=prepared_global_state,
            )
            f2 = rho_total_ess_from_polynomial(
                coeffs_low_to_high,
                carry_logw,
                x2,
                surrogate_logw_clip=surrogate_logw_clip,
                raw_clip_jvp=raw_clip_jvp,
                rank_clip_jvp_topk=rank_clip_jvp_topk,
                adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
                adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
                adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
                raw_clip_center=raw_clip_center,
                raw_clip_count_eps=raw_clip_count_eps,
                raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
                surrogate_kwargs=surrogate_kwargs,
                world_size=world_size,
                global_start=global_start,
                local_n=local_n,
                prepared_global_state=prepared_global_state,
            )
            for _ in range(24):
                if f1 < f2:
                    left = x1
                    x1 = x2
                    f1 = f2
                    x2 = left + gr * (right - left)
                    f2 = rho_total_ess_from_polynomial(
                        coeffs_low_to_high,
                        carry_logw,
                        x2,
                        surrogate_logw_clip=surrogate_logw_clip,
                        raw_clip_jvp=raw_clip_jvp,
                        rank_clip_jvp_topk=rank_clip_jvp_topk,
                        adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
                        adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
                        adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
                        raw_clip_center=raw_clip_center,
                        raw_clip_count_eps=raw_clip_count_eps,
                        raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
                        surrogate_kwargs=surrogate_kwargs,
                        world_size=world_size,
                        global_start=global_start,
                        local_n=local_n,
                        prepared_global_state=prepared_global_state,
                    )
                else:
                    right = x2
                    x2 = x1
                    f2 = f1
                    x1 = right - gr * (right - left)
                    f1 = rho_total_ess_from_polynomial(
                        coeffs_low_to_high,
                        carry_logw,
                        x1,
                        surrogate_logw_clip=surrogate_logw_clip,
                        raw_clip_jvp=raw_clip_jvp,
                        rank_clip_jvp_topk=rank_clip_jvp_topk,
                        adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
                        adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
                        adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
                        raw_clip_center=raw_clip_center,
                        raw_clip_count_eps=raw_clip_count_eps,
                        raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
                        surrogate_kwargs=surrogate_kwargs,
                        world_size=world_size,
                        global_start=global_start,
                        local_n=local_n,
                        prepared_global_state=prepared_global_state,
                    )
            mid = 0.5 * (left + right)
            mid_ess = rho_total_ess_from_polynomial(
                coeffs_low_to_high,
                carry_logw,
                mid,
                surrogate_logw_clip=surrogate_logw_clip,
                raw_clip_jvp=raw_clip_jvp,
                rank_clip_jvp_topk=rank_clip_jvp_topk,
                adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
                adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
                adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
                raw_clip_center=raw_clip_center,
                raw_clip_count_eps=raw_clip_count_eps,
                raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
                surrogate_kwargs=surrogate_kwargs,
                world_size=world_size,
                global_start=global_start,
                local_n=local_n,
                prepared_global_state=prepared_global_state,
            )
            for r, ess in ((x1, f1), (x2, f2), (mid, mid_ess)):
                if math.isfinite(ess) and ess > best_ess:
                    best_rho = float(r)
                    best_ess = float(ess)

    selected_rho0 = 0.0
    if guard_rho0 and math.isfinite(zero_ess) and zero_ess >= best_ess - 1e-14:
        best_rho = 0.0
        best_ess = float(zero_ess)
        selected_rho0 = 1.0

    accepted = 1.0
    if accept_only_improve and best_ess < start_ess - 1e-14:
        best_rho = float(rho_init)
        best_ess = float(start_ess)
        accepted = 0.0

    final_obj = 1.0 - float(best_ess)
    start_obj = 1.0 - float(start_ess)
    zero_obj = 1.0 - float(zero_ess)
    return {
        "rho": float(best_rho),
        "objective_start": float(start_obj),
        "objective_final": float(final_obj),
        "objective_rho0": float(zero_obj),
        "grad_start": float("nan"),
        "grad_final": float("nan"),
        "grad_rho0": float("nan"),
        "accepted": float(accepted),
        "selected_rho0": float(selected_rho0),
        "accepted_steps": 0.0,
        "backtracks": 0.0,
        "lr_last": 0.0,
        "line_search_failed": 0.0,
        "exact_candidates": float(len(unique)),
        "search_candidates": float(len(unique)),
        "exact_stationary": 0.0,
        "over1_penalty": 0.0,
        "surrogate_poly_degree": float(len(coeffs_low_to_high) - 1),
        f"{ess_prefix}_start": float(start_ess),
        f"{ess_prefix}_final": float(best_ess),
        f"{ess_prefix}_rho0": float(zero_ess),
        "raw_clip_jvp_in_rho_objective": float(
            bool(
                raw_clip_jvp_in_rho_objective
                and (float(raw_clip_jvp) > 0.0 or int(rank_clip_jvp_topk) > 0)
            )
        ),
        # Retain this legacy metric name for old plotting code.
        "rho_ess_search_exact": float(search_mode == "dense"),
        "rho_ess_search_coarse_candidates": float(
            int(ess_coarse_candidates) if search_mode == "coarse" else 0
        ),
    }


def optimize_rho_global(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
    *,
    rho_init: float,
    rho_lr: float,
    rho_steps: int,
    rho_cap: float,
    optimizer: str,
    accept_only_improve: bool,
    line_search_halvings: int,
    guard_rho0: bool,
    over1_penalty: float,
) -> dict[str, float]:
    start_obj, start_grad = global_weighted_poly_objective(
        rho_init, a, b, c, w, over1_penalty=over1_penalty
    )
    zero_obj, zero_grad = global_weighted_poly_objective(
        0.0, a, b, c, w, over1_penalty=over1_penalty
    )
    rho = float(rho_init)
    cur_obj = float(start_obj)
    cur_grad = float(start_grad)
    best_rho = float(rho)
    best_obj = float(start_obj)
    best_grad = float(start_grad)
    selected_rho0 = 0.0
    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    accepted_steps = 0
    backtracks = 0
    line_search_failed = 0.0
    lr_last = 0.0

    exact_candidates = 0
    exact_stationary = 0

    if optimizer == "exact":
        coeffs = global_weighted_poly_variance_coefficients(a, b, c, w)
        best_rho, best_obj, best_grad, exact_candidates, exact_stationary = exact_minimize_rho_variance(
            coeffs,
            rho_cap=rho_cap,
            over1_penalty=over1_penalty,
            rho_init=rho_init,
        )
    else:
        for k in range(1, int(rho_steps) + 1):
            grad = cur_grad
            if not math.isfinite(grad):
                break
            if optimizer == "adam":
                m_next = beta1 * m + (1.0 - beta1) * grad
                v_next = beta2 * v + (1.0 - beta2) * grad * grad
                mhat = m_next / (1.0 - beta1**k)
                vhat = v_next / (1.0 - beta2**k)
                direction = mhat / (math.sqrt(vhat) + 1e-12)
            else:
                m_next = m
                v_next = v
                direction = grad

            lr_try = float(rho_lr)
            step_accepted = False
            for _ in range(max(0, int(line_search_halvings)) + 1):
                rho_candidate = rho - lr_try * direction
                if rho_cap is not None and rho_cap >= 0:
                    rho_candidate = max(-float(rho_cap), min(float(rho_cap), rho_candidate))
                cand_obj, cand_grad = global_weighted_poly_objective(
                    rho_candidate, a, b, c, w, over1_penalty=over1_penalty
                )
                if math.isfinite(cand_obj) and cand_obj <= cur_obj + 1e-14:
                    rho = float(rho_candidate)
                    cur_obj = float(cand_obj)
                    cur_grad = float(cand_grad)
                    if cur_obj < best_obj:
                        best_rho = float(rho)
                        best_obj = float(cur_obj)
                        best_grad = float(cur_grad)
                    m = m_next
                    v = v_next
                    lr_last = lr_try
                    accepted_steps += 1
                    step_accepted = True
                    break
                lr_try *= 0.5
                backtracks += 1

            if not step_accepted:
                line_search_failed = 1.0
                break

    rho = best_rho
    final_obj = best_obj
    final_grad = best_grad
    if guard_rho0 and math.isfinite(zero_obj) and zero_obj <= final_obj + 1e-14:
        rho = 0.0
        final_obj = float(zero_obj)
        final_grad = float(zero_grad)
        selected_rho0 = 1.0

    accepted = 1.0
    if accept_only_improve and final_obj > start_obj:
        rho = float(rho_init)
        final_obj = start_obj
        final_grad = start_grad
        accepted = 0.0

    return {
        "rho": float(rho),
        "objective_start": float(start_obj),
        "objective_final": float(final_obj),
        "objective_rho0": float(zero_obj),
        "grad_start": float(start_grad),
        "grad_final": float(final_grad),
        "grad_rho0": float(zero_grad),
        "accepted": accepted,
        "selected_rho0": float(selected_rho0),
        "accepted_steps": float(accepted_steps),
        "backtracks": float(backtracks),
        "lr_last": float(lr_last),
        "line_search_failed": float(line_search_failed),
        "exact_candidates": float(exact_candidates),
        "exact_stationary": float(exact_stationary),
        "over1_penalty": float(over1_penalty),
        "surrogate_poly_degree": 2.0,
    }


def pa_gate_proposal_mean(
    rho_ratio: float,
    *,
    latents: torch.Tensor,
    mu_a: torch.Tensor,
    variance: torch.Tensor,
    beta_l: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    lhat_temp: float,
    gate_power: float,
    linear_baseline_kappa: float = 0.0,
    pa_residual_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proposal reverse mean and displacement h*a_r for the pA-gate target."""
    y = latents.float()
    nu_a = y - mu_a.float()
    beta = float(lhat_temp) * beta_l.float()
    baseline_lhat_coeff = float(linear_baseline_kappa) / max(
        abs(float(lhat_temp)), 1e-30
    )
    residual_gate_coeff = float(pa_residual_alpha) * (
        float(gate_power)
        * theta.float()[:, None, None, None]
        * float(eta)
        - baseline_lhat_coeff
    )
    psi = -residual_gate_coeff * beta
    q = residual_gate_coeff * beta + float(rho_ratio) * psi
    nu0 = nu_a
    disp = nu0 + variance.float() * q
    return y - disp, disp


def sample_proposal_and_lhat(
    rho_ratio: float,
    *,
    latents: torch.Tensor,
    lhat: torch.Tensor,
    mu_a: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    variance: torch.Tensor,
    beta_l: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    z: torch.Tensor,
    lhat_temp: float,
    gate_power: float,
    linear_baseline_kappa: float,
    pa_residual_alpha: float,
    lhat_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the rho proposal and update lhat using only endpoint A/B kernels.

    This is the cheap path used when finite Gaussian diagnostic monitoring is
    disabled.  It does not evaluate the forward K^{b_r} correction at x_new.
    """
    mu_rho, _ = pa_gate_proposal_mean(
        rho_ratio,
        latents=latents,
        mu_a=mu_a,
        variance=variance,
        beta_l=beta_l,
        theta=theta,
        eta=eta,
        lhat_temp=lhat_temp,
        gate_power=gate_power,
        linear_baseline_kappa=linear_baseline_kappa,
        pa_residual_alpha=pa_residual_alpha,
    )
    x_new = mu_rho + z.to(dtype=latents.dtype)
    logp_a = gaussian_log_prob_isotropic(x_new, gate_mu_a, variance)
    logp_b = gaussian_log_prob_isotropic(x_new, gate_mu_b, variance)
    kernel_lr = float(lhat_temp) * (logp_b - logp_a)
    lhat_new = lhat + kernel_lr
    if lhat_clip is not None and lhat_clip > 0:
        lhat_new = lhat_new.clamp(-float(lhat_clip), float(lhat_clip))
    return x_new, lhat_new, kernel_lr.float()


def hybrid_exact_gate_reverse_logw(
    rho_ratio: float,
    *,
    surrogate_kwargs: dict[str, Any],
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power_new: float,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    lhat_clip: float | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cheap hybrid weight with exact gate and reverse Gaussian factors.

    With the Brownian draw fixed, the proposal sample is affine in rho.  The
    equal-variance A/B Gaussian log-ratio is therefore affine as well (before
    the optional lhat clamp), making the exact finite gate factor cheap to
    evaluate for every rho candidate.  The exact reverse Gaussian ratio is
    likewise cheap and needs no additional diffusion-model call.

    Only the retained forward/JVP expansion remains approximate:

        hybrid = retained total - retained gate - retained reverse
                 + exact gate + exact reverse.
    """
    terms = local_surrogate_terms(float(rho_ratio), **surrogate_kwargs)
    retained = surrogate_decomposition_from_terms(terms)
    mu_rho, _ = pa_gate_proposal_mean(
        float(rho_ratio),
        latents=surrogate_kwargs["latents"],
        mu_a=surrogate_kwargs["mu_a"],
        variance=surrogate_kwargs["variance"],
        beta_l=surrogate_kwargs["beta_l"],
        theta=surrogate_kwargs["theta"],
        eta=float(eta),
        lhat_temp=float(surrogate_kwargs["lhat_temp"]),
        gate_power=float(surrogate_kwargs["gate_power"]),
        linear_baseline_kappa=float(
            surrogate_kwargs.get("linear_baseline_kappa", 0.0)
        ),
        pa_residual_alpha=float(
            surrogate_kwargs.get("pa_residual_alpha", 1.0)
        ),
    )
    x_new = mu_rho + surrogate_kwargs["z"].to(
        dtype=surrogate_kwargs["latents"].dtype
    )
    variance = surrogate_kwargs["variance"]
    logp_gate_a = gaussian_log_prob_isotropic(x_new, gate_mu_a, variance)
    logp_gate_b = gaussian_log_prob_isotropic(x_new, gate_mu_b, variance)
    kernel_lr = float(surrogate_kwargs["lhat_temp"]) * (
        logp_gate_b - logp_gate_a
    )
    lhat_new = lhat + kernel_lr
    if lhat_clip is not None and float(lhat_clip) > 0.0:
        lhat_new = lhat_new.clamp(-float(lhat_clip), float(lhat_clip))

    exact_gate = (
        float(gate_power_new)
        * log1m_theta_from_lhat(lhat_new, c_new, eta_new)
        - float(surrogate_kwargs["gate_power"])
        * log1m_theta_from_lhat(lhat, c, eta)
    )
    kappa = float(
        surrogate_kwargs.get("linear_baseline_kappa", 0.0)
    )
    kappa_new = float(
        surrogate_kwargs.get("linear_baseline_kappa_new", kappa)
    )
    exact_gate = exact_gate + centered_lhat_increment(
        lhat_new,
        lhat,
        kappa_new=kappa_new,
        kappa=kappa,
        lhat_temp=float(surrogate_kwargs["lhat_temp"]),
    )
    exact_gate = float(
        surrogate_kwargs.get("pa_residual_alpha", 1.0)
    ) * exact_gate
    exact_reverse = gaussian_log_prob_isotropic(
        x_new, surrogate_kwargs["mu_a"], variance
    ) - gaussian_log_prob_isotropic(x_new, mu_rho, variance)
    hybrid = (
        retained["total"].float()
        - retained["group_gate"].float()
        - retained["group_reverse_kernel_alpha"].float()
        + exact_gate.float()
        + exact_reverse.float()
    )

    decomp = dict(retained)
    decomp["gate_retained"] = retained["group_gate"].float()
    decomp["reverse_retained"] = retained[
        "group_reverse_kernel_alpha"
    ].float()
    decomp["gate_exact"] = exact_gate.float()
    decomp["reverse_exact"] = exact_reverse.float()
    decomp["group_gate"] = exact_gate.float()
    decomp["group_reverse_kernel_alpha"] = exact_reverse.float()
    decomp["group_kernel_alpha"] = (
        decomp["group_forward_kernel_alpha"].float() + exact_reverse.float()
    )
    decomp["total"] = hybrid.float()
    return hybrid.float(), decomp


def optimize_rho_hybrid_incremental_ess(
    *,
    surrogate_kwargs: dict[str, Any],
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power_new: float,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    lhat_clip: float,
    rho_init: float,
    rho_cap: float,
    rho_steps: int,
    ess_search_mode: str,
    ess_coarse_candidates: int,
    rho_lr: float = 0.05,
    rho_optimizer: str = "adam",
    line_search_halvings: int = 12,
    guard_rho0: bool,
    accept_only_improve: bool,
    min_ess_gain: float,
    surrogate_logw_clip: float,
    raw_clip_jvp: float,
    rank_clip_jvp_topk: int,
    adaptive_rank_clip_jvp: bool,
    adaptive_rank_clip_jvp_ess_thresholds: str,
    adaptive_rank_clip_jvp_topks: str,
    raw_clip_center: str,
    raw_clip_count_eps: float,
    raw_clip_jvp_in_rho_objective: bool,
    world_size: int,
    expected_particles: int,
    carry_logw: torch.Tensor | None = None,
    rho_objective: str = "incremental_ess",
) -> dict[str, Any]:
    """Dense, one-gather rho search on the cheap hybrid weight.

    The retained terms are low-degree polynomials in rho.  The exact reverse
    Gaussian ratio is quadratic, while the equal-variance A/B log-ratio
    entering the exact gate is affine.  Build those sufficient statistics
    once, then evaluate every rho candidate using only particle-sized tensors.
    This avoids repeating latent-sized work for a 100+ point rho grid.  The
    backward-compatible ``incremental_ess`` objective scores each increment
    alone; ``total_ess`` adds the carried global log weights before scoring.
    """
    if float(rho_cap) < 0.0:
        raise ValueError("Hybrid rho ESS search requires a finite nonnegative cap")
    objective = str(rho_objective)
    if objective not in {"incremental_ess", "total_ess"}:
        raise ValueError(
            "Hybrid rho ESS search requires rho_objective in "
            "{'incremental_ess', 'total_ess'}"
        )
    use_total_ess = objective == "total_ess"
    if use_total_ess and carry_logw is None:
        raise ValueError("Hybrid total_ess rho search requires carried log weights")
    mode = str(ess_search_mode)
    if mode == "gradient":
        return optimize_rho_hybrid_gradient_ess(
            surrogate_kwargs=surrogate_kwargs,
            lhat=lhat,
            c=c,
            eta=eta,
            c_new=c_new,
            eta_new=eta_new,
            gate_power_new=gate_power_new,
            gate_mu_a=gate_mu_a,
            gate_mu_b=gate_mu_b,
            lhat_clip=lhat_clip,
            rho_init=rho_init,
            rho_cap=rho_cap,
            rho_lr=rho_lr,
            rho_steps=rho_steps,
            rho_optimizer=rho_optimizer,
            line_search_halvings=line_search_halvings,
            guard_rho0=guard_rho0,
            accept_only_improve=accept_only_improve,
            min_ess_gain=min_ess_gain,
            surrogate_logw_clip=surrogate_logw_clip,
            raw_clip_jvp=raw_clip_jvp,
            rank_clip_jvp_topk=rank_clip_jvp_topk,
            adaptive_rank_clip_jvp=adaptive_rank_clip_jvp,
            adaptive_rank_clip_jvp_ess_thresholds=adaptive_rank_clip_jvp_ess_thresholds,
            adaptive_rank_clip_jvp_topks=adaptive_rank_clip_jvp_topks,
            raw_clip_center=raw_clip_center,
            raw_clip_jvp_in_rho_objective=raw_clip_jvp_in_rho_objective,
            world_size=world_size,
            expected_particles=expected_particles,
            carry_logw=carry_logw,
            rho_objective=rho_objective,
        )
    if mode not in {"dense", "coarse", "exact"}:
        raise ValueError(f"Unknown hybrid ESS rho search mode {mode!r}")
    n_candidates = (
        max(17, int(rho_steps) + 1)
        if mode in {"dense", "exact"}
        else max(3, int(ess_coarse_candidates))
    )
    bounded_init = max(
        -float(rho_cap), min(float(rho_cap), float(rho_init))
    )
    candidates = torch.linspace(
        -float(rho_cap),
        float(rho_cap),
        n_candidates,
        device=lhat.device,
        dtype=torch.float32,
    )
    # Include rho_init and rho=0 exactly, even for an even/coarse grid.
    candidates = torch.unique(
        torch.cat(
            [
                candidates,
                torch.tensor(
                    [bounded_init, 0.0],
                    device=lhat.device,
                    dtype=torch.float32,
                ),
            ]
        ),
        sorted=True,
    )

    degree = 3 if str(surrogate_kwargs["gamma_delta_mode"]) == "full" else 2
    nodes = (0.0, 1.0, -1.0) if degree == 2 else (0.0, 1.0, -1.0, 2.0)
    node_decomp = tuple(
        surrogate_decomposition_from_terms(
            local_surrogate_terms(rho_value, **surrogate_kwargs)
        )
        for rho_value in nodes
    )

    def component_coefficients(name: str) -> tuple[torch.Tensor, ...]:
        return _polynomial_coefficients_from_samples(
            tuple(decomp[name].float().reshape(-1) for decomp in node_decomp)
        )

    retained_total_coeffs = component_coefficients("total")
    retained_gate_coeffs = component_coefficients("group_gate")
    retained_reverse_coeffs = component_coefficients(
        "group_reverse_kernel_alpha"
    )
    jvp_coeffs = component_coefficients("proposal_hessian_jvp")

    cand_col = candidates[:, None]

    def evaluate_coefficients(
        coeffs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        out = torch.zeros(
            (int(candidates.numel()), int(coeffs[0].numel())),
            device=candidates.device,
            dtype=torch.float32,
        )
        power = torch.ones_like(cand_col)
        for coefficient in coeffs:
            out = out + power * coefficient[None, :]
            power = power * cand_col
        return out

    retained_total = evaluate_coefficients(retained_total_coeffs)
    retained_gate = evaluate_coefficients(retained_gate_coeffs)
    retained_reverse = evaluate_coefficients(retained_reverse_coeffs)
    local_jvp_matrix = evaluate_coefficients(jvp_coeffs)

    # Exact gate: x_new(rho)=x0+rho*dx and, for equal variances,
    # log K_B(x)-log K_A(x) is affine in x and hence in rho.
    mu0, _ = pa_gate_proposal_mean(
        0.0,
        latents=surrogate_kwargs["latents"],
        mu_a=surrogate_kwargs["mu_a"],
        variance=surrogate_kwargs["variance"],
        beta_l=surrogate_kwargs["beta_l"],
        theta=surrogate_kwargs["theta"],
        eta=float(eta),
        lhat_temp=float(surrogate_kwargs["lhat_temp"]),
        gate_power=float(surrogate_kwargs["gate_power"]),
        linear_baseline_kappa=float(
            surrogate_kwargs.get("linear_baseline_kappa", 0.0)
        ),
        pa_residual_alpha=float(
            surrogate_kwargs.get("pa_residual_alpha", 1.0)
        ),
    )
    x0 = mu0.float() + surrogate_kwargs["z"].float()
    variance = surrogate_kwargs["variance"].float()
    gate_lr0 = float(surrogate_kwargs["lhat_temp"]) * (
        gaussian_log_prob_isotropic(x0, gate_mu_b, variance)
        - gaussian_log_prob_isotropic(x0, gate_mu_a, variance)
    )
    q0 = float(surrogate_kwargs.get("pa_residual_alpha", 1.0)) * (
        float(surrogate_kwargs["gate_power"])
        * surrogate_kwargs["theta"].float()[:, None, None, None]
        * float(eta)
        * float(surrogate_kwargs["lhat_temp"])
        * surrogate_kwargs["beta_l"].float()
        - float(surrogate_kwargs.get("linear_baseline_kappa", 0.0))
        * surrogate_kwargs["beta_l"].float()
    )
    psi = -q0
    dx = -_broadcast_particle_scalar(variance, q0) * psi
    gate_mean_delta = (gate_mu_b.float() - gate_mu_a.float()).flatten(1)
    gate_lr1 = float(surrogate_kwargs["lhat_temp"]) * (
        torch.sum(dx.flatten(1) * gate_mean_delta, dim=1) / variance
    )
    lhat_candidates = (
        lhat.float()[None, :]
        + gate_lr0[None, :]
        + cand_col * gate_lr1[None, :]
    )
    if float(lhat_clip) > 0.0:
        lhat_candidates = lhat_candidates.clamp(
            -float(lhat_clip), float(lhat_clip)
        )
    exact_gate = (
        float(gate_power_new)
        * log1m_theta_from_lhat(lhat_candidates, c_new, eta_new)
        - float(surrogate_kwargs["gate_power"])
        * log1m_theta_from_lhat(lhat.float(), c, eta)[None, :]
    )
    kappa = float(
        surrogate_kwargs.get("linear_baseline_kappa", 0.0)
    )
    kappa_new = float(
        surrogate_kwargs.get("linear_baseline_kappa_new", kappa)
    )
    exact_gate = exact_gate + centered_lhat_increment(
        lhat_candidates,
        lhat.float()[None, :],
        kappa_new=kappa_new,
        kappa=kappa,
        lhat_temp=float(surrogate_kwargs["lhat_temp"]),
    )
    exact_gate = float(
        surrogate_kwargs.get("pa_residual_alpha", 1.0)
    ) * exact_gate

    # Exact reverse ratio:
    # log L^{nu_A}(x)-log L^{a_rho}(x)
    #   = z^T q(rho) - 0.5 * variance * ||q(rho)||^2.
    zf = surrogate_kwargs["z"].float().flatten(1)
    q0f = q0.flatten(1)
    psif = psi.flatten(1)
    reverse_c0 = (
        torch.sum(zf * q0f, dim=1)
        - 0.5 * variance * torch.sum(q0f * q0f, dim=1)
    )
    reverse_c1 = (
        torch.sum(zf * psif, dim=1)
        - variance * torch.sum(q0f * psif, dim=1)
    )
    reverse_c2 = -0.5 * variance * torch.sum(psif * psif, dim=1)
    exact_reverse = (
        reverse_c0[None, :]
        + cand_col * reverse_c1[None, :]
        + cand_col.square() * reverse_c2[None, :]
    )
    local_weight_matrix = (
        retained_total
        - retained_gate
        - retained_reverse
        + exact_gate
        + exact_reverse
    )

    rows_to_gather = [local_weight_matrix, local_jvp_matrix]
    if use_total_ess:
        assert carry_logw is not None
        local_carry = carry_logw.detach().float().reshape(-1)
        if int(local_carry.numel()) != int(local_weight_matrix.shape[1]):
            raise ValueError(
                "Hybrid total_ess carried weights must match the local "
                f"population: expected {int(local_weight_matrix.shape[1])}, "
                f"got {int(local_carry.numel())}"
            )
        rows_to_gather.append(local_carry[None, :])
    gathered = all_gather_particle_rows(
        torch.cat(rows_to_gather, dim=0),
        int(world_size),
    ).float()
    if int(gathered.shape[1]) != int(expected_particles):
        raise RuntimeError(
            "Hybrid rho ESS search must use the complete global population: "
            f"expected {expected_particles}, got {int(gathered.shape[1])}"
        )
    n_rho = int(candidates.numel())
    global_weights = gathered[:n_rho]
    global_jvp = gathered[n_rho : 2 * n_rho]
    global_carry = gathered[2 * n_rho] if use_total_ess else None
    inc_ess_values: list[float] = []
    total_ess_values: list[float] = []
    for index in range(n_rho):
        inc = global_weights[index]
        if bool(raw_clip_jvp_in_rho_objective) and (
            float(raw_clip_jvp) > 0.0
            or int(rank_clip_jvp_topk) > 0
            or bool(adaptive_rank_clip_jvp)
        ):
            jvp = global_jvp[index]
            effective_topk = adaptive_rank_clip_topk_from_ess(
                global_ess_frac_from_logw(inc),
                enabled=bool(adaptive_rank_clip_jvp),
                fallback_topk=int(rank_clip_jvp_topk),
                thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
                topks=str(adaptive_rank_clip_jvp_topks),
            )
            if int(effective_topk) > 0:
                _, residual, _, _, _ = _global_centered_rank_clip(
                    jvp,
                    int(effective_topk),
                    center_mode=str(raw_clip_center),
                    count_eps=float(raw_clip_count_eps),
                )
            else:
                _, residual, _, _ = _global_centered_clip(
                    jvp,
                    float(raw_clip_jvp),
                    center_mode=str(raw_clip_center),
                    count_eps=float(raw_clip_count_eps),
                )
            inc = inc - residual
        inc = clip_final_surrogate_logw(inc, float(surrogate_logw_clip))
        inc_ess_values.append(global_ess_frac_from_logw(inc))
        if global_carry is not None:
            total_logw = torch.nan_to_num(
                global_carry + inc,
                nan=0.0,
                posinf=80.0,
                neginf=-80.0,
            )
            total_ess_values.append(global_ess_frac_from_logw(total_logw))

    ess_values = total_ess_values if use_total_ess else inc_ess_values

    candidate_values = [float(value) for value in candidates.tolist()]
    zero_index = min(
        range(n_rho), key=lambda index: abs(candidate_values[index])
    )
    start_index = min(
        range(n_rho),
        key=lambda index: abs(candidate_values[index] - bounded_init),
    )
    best_index = max(range(n_rho), key=lambda index: ess_values[index])
    zero_ess = float(ess_values[zero_index])
    start_ess = float(ess_values[start_index])
    best_ess = float(ess_values[best_index])
    selected_rho0 = 0.0
    min_gain = max(0.0, float(min_ess_gain))
    if bool(guard_rho0) and best_ess <= zero_ess + min_gain:
        best_index = zero_index
        best_ess = zero_ess
        selected_rho0 = 1.0
    accepted = 1.0
    if bool(accept_only_improve) and best_ess + 1e-14 < start_ess:
        best_index = start_index
        best_ess = start_ess
        accepted = 0.0
        selected_rho0 = float(abs(candidate_values[start_index]) <= 1e-12)

    inc_start_ess = float(inc_ess_values[start_index])
    inc_final_ess = float(inc_ess_values[best_index])
    inc_zero_ess = float(inc_ess_values[zero_index])
    if use_total_ess:
        total_start_ess = float(total_ess_values[start_index])
        total_final_ess = float(total_ess_values[best_index])
        total_zero_ess = float(total_ess_values[zero_index])
    else:
        total_start_ess = float("nan")
        total_final_ess = float("nan")
        total_zero_ess = float("nan")

    return {
        "rho": candidate_values[best_index],
        "objective_start": 1.0 - start_ess,
        "objective_final": 1.0 - best_ess,
        "objective_rho0": 1.0 - zero_ess,
        "rho_inc_ess_start": inc_start_ess,
        "rho_inc_ess_final": inc_final_ess,
        "rho_inc_ess_rho0": inc_zero_ess,
        "rho_total_ess_start": total_start_ess,
        "rho_total_ess_final": total_final_ess,
        "rho_total_ess_rho0": total_zero_ess,
        "accepted": accepted,
        "selected_rho0": selected_rho0,
        "accepted_steps": 1.0,
        "backtracks": 0.0,
        "lr_last": 0.0,
        "line_search_failed": 0.0,
        "exact_candidates": float(n_rho),
        "search_candidates": float(n_rho),
        "exact_stationary": 0.0,
        "hybrid_exact_gate_reverse": 1.0,
        "hybrid_ess_objective": objective,
        "hybrid_min_ess_gain": min_gain,
        "surrogate_poly_degree": float("nan"),
        "raw_clip_jvp_in_rho_objective": float(
            bool(raw_clip_jvp_in_rho_objective)
        ),
        "rho_optimizer_effective": "dense" if mode in {"dense", "exact"} else "coarse",
        "rho_optimizer_iterations": 0.0,
        "rho_objective_evaluations": float(n_rho),
    }


def optimize_rho_hybrid_gradient_ess(
    *,
    surrogate_kwargs: dict[str, Any],
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power_new: float,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    lhat_clip: float,
    rho_init: float,
    rho_cap: float,
    rho_lr: float,
    rho_steps: int,
    rho_optimizer: str,
    line_search_halvings: int,
    guard_rho0: bool,
    accept_only_improve: bool,
    min_ess_gain: float,
    surrogate_logw_clip: float,
    raw_clip_jvp: float,
    rank_clip_jvp_topk: int,
    adaptive_rank_clip_jvp: bool,
    adaptive_rank_clip_jvp_ess_thresholds: str,
    adaptive_rank_clip_jvp_topks: str,
    raw_clip_center: str,
    raw_clip_jvp_in_rho_objective: bool,
    world_size: int,
    expected_particles: int,
    carry_logw: torch.Tensor | None = None,
    rho_objective: str = "incremental_ess",
) -> dict[str, Any]:
    """Projected gradient search on the same cheap hybrid ESS objective.

    All latent-sized retained/exact terms are reduced to per-particle
    polynomial/affine sufficient statistics before this function's sole
    collective.  Adam/GD iterations then operate independently but identically
    on the gathered global population and require no diffusion-model calls or
    collectives.  Fixed-rank JVP clipping is differentiated piecewise while
    retaining exactly the same winsorized objective used by dense search.
    """
    if float(rho_cap) < 0.0:
        raise ValueError("Hybrid rho gradient search requires a finite nonnegative cap")
    if float(rho_lr) <= 0.0:
        raise ValueError("Hybrid rho gradient search requires rho_lr > 0")
    optimizer = str(rho_optimizer)
    if optimizer not in {"adam", "gd"}:
        raise ValueError(
            "Hybrid rho gradient search requires --rho-optimizer adam or gd"
        )
    objective = str(rho_objective)
    if objective not in {"incremental_ess", "total_ess"}:
        raise ValueError(
            "Hybrid rho gradient search requires rho_objective in "
            "{'incremental_ess', 'total_ess'}"
        )
    use_total_ess = objective == "total_ess"
    if use_total_ess and carry_logw is None:
        raise ValueError("Hybrid total_ess rho search requires carried log weights")

    degree = 3 if str(surrogate_kwargs["gamma_delta_mode"]) == "full" else 2
    nodes = (0.0, 1.0, -1.0) if degree == 2 else (0.0, 1.0, -1.0, 2.0)
    node_decomp = tuple(
        surrogate_decomposition_from_terms(
            local_surrogate_terms(rho_value, **surrogate_kwargs)
        )
        for rho_value in nodes
    )

    def component_coefficients(name: str) -> tuple[torch.Tensor, ...]:
        return tuple(
            value.detach().float().reshape(-1)
            for value in _polynomial_coefficients_from_samples(
                tuple(decomp[name].float().reshape(-1) for decomp in node_decomp)
            )
        )

    retained_total = component_coefficients("total")
    retained_gate = component_coefficients("group_gate")
    retained_reverse = component_coefficients("group_reverse_kernel_alpha")
    retained_base = tuple(
        total - gate - reverse
        for total, gate, reverse in zip(
            retained_total, retained_gate, retained_reverse
        )
    )
    jvp_coeffs = component_coefficients("proposal_hessian_jvp")

    # Exact gate sufficient statistics: lhat_new(rho)=lhat+lr0+rho*lr1.
    mu0, _ = pa_gate_proposal_mean(
        0.0,
        latents=surrogate_kwargs["latents"],
        mu_a=surrogate_kwargs["mu_a"],
        variance=surrogate_kwargs["variance"],
        beta_l=surrogate_kwargs["beta_l"],
        theta=surrogate_kwargs["theta"],
        eta=float(eta),
        lhat_temp=float(surrogate_kwargs["lhat_temp"]),
        gate_power=float(surrogate_kwargs["gate_power"]),
        linear_baseline_kappa=float(
            surrogate_kwargs.get("linear_baseline_kappa", 0.0)
        ),
        pa_residual_alpha=float(surrogate_kwargs.get("pa_residual_alpha", 1.0)),
    )
    x0 = mu0.float() + surrogate_kwargs["z"].float()
    variance = surrogate_kwargs["variance"].float()
    gate_lr0 = float(surrogate_kwargs["lhat_temp"]) * (
        gaussian_log_prob_isotropic(x0, gate_mu_b, variance)
        - gaussian_log_prob_isotropic(x0, gate_mu_a, variance)
    )
    q0 = float(surrogate_kwargs.get("pa_residual_alpha", 1.0)) * (
        float(surrogate_kwargs["gate_power"])
        * surrogate_kwargs["theta"].float()[:, None, None, None]
        * float(eta)
        * float(surrogate_kwargs["lhat_temp"])
        * surrogate_kwargs["beta_l"].float()
        - float(surrogate_kwargs.get("linear_baseline_kappa", 0.0))
        * surrogate_kwargs["beta_l"].float()
    )
    psi = -q0
    dx = -_broadcast_particle_scalar(variance, q0) * psi
    gate_mean_delta = (gate_mu_b.float() - gate_mu_a.float()).flatten(1)
    gate_lr1 = float(surrogate_kwargs["lhat_temp"]) * (
        torch.sum(dx.flatten(1) * gate_mean_delta, dim=1) / variance
    )

    # Exact reverse ratio is quadratic in rho.
    zf = surrogate_kwargs["z"].float().flatten(1)
    q0f = q0.flatten(1)
    psif = psi.flatten(1)
    reverse_c0 = (
        torch.sum(zf * q0f, dim=1)
        - 0.5 * variance * torch.sum(q0f * q0f, dim=1)
    )
    reverse_c1 = (
        torch.sum(zf * psif, dim=1)
        - variance * torch.sum(q0f * psif, dim=1)
    )
    reverse_c2 = -0.5 * variance * torch.sum(psif * psif, dim=1)

    local_n = int(retained_base[0].numel())
    local_lhat = lhat.detach().float().reshape(-1)
    local_rows = [
        *retained_base,
        *jvp_coeffs,
        gate_lr0.detach().float().reshape(-1),
        gate_lr1.detach().float().reshape(-1),
        local_lhat,
        reverse_c0.detach().float().reshape(-1),
        reverse_c1.detach().float().reshape(-1),
        reverse_c2.detach().float().reshape(-1),
    ]
    if any(int(row.numel()) != local_n for row in local_rows):
        raise ValueError("Hybrid rho gradient sufficient statistics disagree on local N")
    if use_total_ess:
        assert carry_logw is not None
        local_carry = carry_logw.detach().float().reshape(-1)
        if int(local_carry.numel()) != local_n:
            raise ValueError(
                "Hybrid total_ess carried weights must match the local population: "
                f"expected {local_n}, got {int(local_carry.numel())}"
            )
        local_rows.append(local_carry)
    gathered = all_gather_particle_rows(
        torch.stack(local_rows, dim=0), int(world_size)
    ).detach().float()
    if int(gathered.shape[1]) != int(expected_particles):
        raise RuntimeError(
            "Hybrid rho gradient search must use the complete global population: "
            f"expected {expected_particles}, got {int(gathered.shape[1])}"
        )

    index = 0
    n_coeffs = degree + 1
    global_base = tuple(gathered[index + offset] for offset in range(n_coeffs))
    index += n_coeffs
    global_jvp = tuple(gathered[index + offset] for offset in range(n_coeffs))
    index += n_coeffs
    global_gate_lr0 = gathered[index]
    global_gate_lr1 = gathered[index + 1]
    global_lhat = gathered[index + 2]
    global_reverse = (gathered[index + 3], gathered[index + 4], gathered[index + 5])
    index += 6
    global_carry = gathered[index] if use_total_ess else None

    kappa = float(surrogate_kwargs.get("linear_baseline_kappa", 0.0))
    kappa_new = float(
        surrogate_kwargs.get("linear_baseline_kappa_new", kappa)
    )
    residual_alpha = float(surrogate_kwargs.get("pa_residual_alpha", 1.0))
    use_jvp_clip = bool(
        raw_clip_jvp_in_rho_objective
        and (
            float(raw_clip_jvp) > 0.0
            or int(rank_clip_jvp_topk) > 0
            or bool(adaptive_rank_clip_jvp)
        )
    )

    def evaluate_polynomial(
        coefficients: tuple[torch.Tensor, ...], rho_value: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros_like(coefficients[0])
        for coefficient in reversed(coefficients):
            result = result * rho_value + coefficient
        return result

    def evaluate_tensors(
        rho_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inc = evaluate_polynomial(global_base, rho_value)
        lhat_new = global_lhat + global_gate_lr0 + rho_value * global_gate_lr1
        if float(lhat_clip) > 0.0:
            lhat_new = lhat_new.clamp(-float(lhat_clip), float(lhat_clip))
        exact_gate = (
            float(gate_power_new)
            * log1m_theta_from_lhat(lhat_new, c_new, eta_new)
            - float(surrogate_kwargs["gate_power"])
            * log1m_theta_from_lhat(global_lhat, c, eta)
        )
        exact_gate = exact_gate + centered_lhat_increment(
            lhat_new,
            global_lhat,
            kappa_new=kappa_new,
            kappa=kappa,
            lhat_temp=float(surrogate_kwargs["lhat_temp"]),
        )
        inc = residual_alpha * exact_gate + inc
        inc = (
            inc
            + global_reverse[0]
            + rho_value * global_reverse[1]
            + rho_value.square() * global_reverse[2]
        )
        if use_jvp_clip:
            jvp = evaluate_polynomial(global_jvp, rho_value)
            effective_topk = adaptive_rank_clip_topk_from_ess(
                float(_rho_gradient_ess_frac(inc).detach().item()),
                enabled=bool(adaptive_rank_clip_jvp),
                fallback_topk=int(rank_clip_jvp_topk),
                thresholds=str(adaptive_rank_clip_jvp_ess_thresholds),
                topks=str(adaptive_rank_clip_jvp_topks),
            )
            residual = _rho_gradient_centered_clip_residual(
                jvp,
                clip=float(raw_clip_jvp),
                rank_topk=int(effective_topk),
                center_mode=str(raw_clip_center),
            )
            inc = inc - residual
        inc = clip_final_surrogate_logw(inc, float(surrogate_logw_clip))
        inc_ess = _rho_gradient_ess_frac(inc)
        if global_carry is None:
            total_ess = torch.tensor(
                float("nan"), device=inc.device, dtype=torch.float32
            )
            target_ess = inc_ess
        else:
            total_ess = _rho_gradient_ess_frac(global_carry + inc)
            target_ess = total_ess
        return target_ess, inc_ess, total_ess

    objective_evaluations = 0

    def evaluate_float(rho_value: float, *, with_grad: bool) -> dict[str, float]:
        nonlocal objective_evaluations
        # The denoising loop intentionally runs under no_grad; locally re-enable
        # scalar autograd only for rho. Gathered sufficient statistics remain
        # detached, so this cannot retain a model graph.
        with torch.enable_grad():
            rho_tensor = torch.tensor(
                float(rho_value),
                device=gathered.device,
                dtype=torch.float32,
                requires_grad=bool(with_grad),
            )
            target_ess, inc_ess, total_ess = evaluate_tensors(rho_tensor)
            grad = float("nan")
            if with_grad:
                loss = 1.0 - target_ess
                grad_tensor = torch.autograd.grad(loss, rho_tensor)[0]
                grad = float(grad_tensor.detach().item())
        objective_evaluations += 1
        return {
            "rho": float(rho_value),
            "ess": float(target_ess.detach().item()),
            "inc_ess": float(inc_ess.detach().item()),
            "total_ess": float(total_ess.detach().item()),
            "grad": grad,
        }

    bounded_init = max(-float(rho_cap), min(float(rho_cap), float(rho_init)))
    start = evaluate_float(bounded_init, with_grad=True)
    zero = start if abs(bounded_init) <= 1e-15 else evaluate_float(0.0, with_grad=True)
    current = dict(start)
    best = dict(start)
    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    accepted_steps = 0
    iterations_used = 0
    backtracks = 0
    lr_last = 0.0
    line_search_failed = 0.0

    for iteration in range(1, max(0, int(rho_steps)) + 1):
        iterations_used = iteration
        grad = float(current["grad"])
        if not math.isfinite(grad):
            line_search_failed = 1.0
            break
        if optimizer == "adam":
            m_next = beta1 * m + (1.0 - beta1) * grad
            v_next = beta2 * v + (1.0 - beta2) * grad * grad
            mhat = m_next / (1.0 - beta1**iteration)
            vhat = v_next / (1.0 - beta2**iteration)
            direction = mhat / (math.sqrt(vhat) + 1e-12)
        else:
            m_next = m
            v_next = v
            direction = grad

        lr_try = float(rho_lr)
        step_accepted = False
        for _ in range(max(0, int(line_search_halvings)) + 1):
            candidate_rho = max(
                -float(rho_cap),
                min(float(rho_cap), float(current["rho"]) - lr_try * direction),
            )
            candidate = evaluate_float(candidate_rho, with_grad=True)
            if math.isfinite(candidate["ess"]) and (
                candidate["ess"] + 1e-14 >= current["ess"]
            ):
                current = candidate
                if current["ess"] > best["ess"]:
                    best = dict(current)
                m = m_next
                v = v_next
                lr_last = lr_try
                accepted_steps += 1
                step_accepted = True
                break
            lr_try *= 0.5
            backtracks += 1
        if not step_accepted:
            line_search_failed = 1.0
            break

    final = dict(best)
    selected_rho0 = 0.0
    min_gain = max(0.0, float(min_ess_gain))
    if bool(guard_rho0) and final["ess"] <= zero["ess"] + min_gain:
        final = dict(zero)
        selected_rho0 = 1.0
    accepted = 1.0
    if bool(accept_only_improve) and final["ess"] + 1e-14 < start["ess"]:
        final = dict(start)
        accepted = 0.0
        selected_rho0 = float(abs(final["rho"]) <= 1e-12)

    return {
        "rho": final["rho"],
        "objective_start": 1.0 - start["ess"],
        "objective_final": 1.0 - final["ess"],
        "objective_rho0": 1.0 - zero["ess"],
        "rho_inc_ess_start": start["inc_ess"],
        "rho_inc_ess_final": final["inc_ess"],
        "rho_inc_ess_rho0": zero["inc_ess"],
        "rho_total_ess_start": start["total_ess"],
        "rho_total_ess_final": final["total_ess"],
        "rho_total_ess_rho0": zero["total_ess"],
        "grad_start": start["grad"],
        "grad_final": final["grad"],
        "grad_rho0": zero["grad"],
        "accepted": accepted,
        "selected_rho0": selected_rho0,
        "accepted_steps": float(accepted_steps),
        "backtracks": float(backtracks),
        "lr_last": float(lr_last),
        "line_search_failed": float(line_search_failed),
        "exact_candidates": 0.0,
        "search_candidates": float(objective_evaluations),
        "exact_stationary": 0.0,
        "hybrid_exact_gate_reverse": 1.0,
        "hybrid_ess_objective": objective,
        "hybrid_min_ess_gain": min_gain,
        "surrogate_poly_degree": float(degree),
        "raw_clip_jvp_in_rho_objective": float(
            bool(raw_clip_jvp_in_rho_objective)
        ),
        "rho_optimizer_effective": optimizer,
        "rho_optimizer_iterations": float(iterations_used),
        "rho_objective_evaluations": float(objective_evaluations),
    }


def finite_weight_diagnostics_for_rho(
    rho_ratio: float,
    *,
    pipe,
    scheduler,
    timestep: int,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    mu_a: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    forward_displacement: torch.Tensor,
    beta_l: torch.Tensor,
    theta: torch.Tensor,
    z: torch.Tensor,
    lhat_temp: float,
    gate_power: float,
    gate_power_new: float,
    lhat_clip: float | None,
    theta_eps: float,
    linear_baseline_kappa: float = 0.0,
    linear_baseline_kappa_new: float | None = None,
    pa_residual_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Finite Gaussian weight plus exact component decomposition.

    The exact diagnostic log-weight used here is

        theta + reverse_kernel + forward_kernel,

    where theta is the exact finite gate-ratio term, reverse_kernel is
    log L^nu_A - log L^a_rho, and forward_kernel is log K^b_rho - log K^mu.
    Returning the pieces lets the monitor localize surrogate/true mismatches
    without changing the sampler's propagation rule.
    """
    mu_rho, _ = pa_gate_proposal_mean(
        rho_ratio,
        latents=latents,
        mu_a=mu_a,
        variance=variance,
        beta_l=beta_l,
        theta=theta,
        eta=eta,
        lhat_temp=lhat_temp,
        gate_power=gate_power,
        linear_baseline_kappa=linear_baseline_kappa,
        pa_residual_alpha=pa_residual_alpha,
    )
    x_new = mu_rho + z.to(dtype=latents.dtype)

    logp_base_a = gaussian_log_prob_isotropic(x_new, mu_a, variance)
    logp_gate_a = gaussian_log_prob_isotropic(x_new, gate_mu_a, variance)
    logp_gate_b = gaussian_log_prob_isotropic(x_new, gate_mu_b, variance)
    logp_rho = gaussian_log_prob_isotropic(x_new, mu_rho, variance)
    kernel_lr = float(lhat_temp) * (logp_gate_b - logp_gate_a)
    lhat_new = lhat + kernel_lr
    if lhat_clip is not None and lhat_clip > 0:
        lhat_new = lhat_new.clamp(-float(lhat_clip), float(lhat_clip))

    theta_term = (
        float(gate_power_new)
        * log1m_theta_from_lhat(lhat_new, c_new, eta_new)
        - float(gate_power) * log1m_theta_from_lhat(lhat, c, eta)
    )
    if linear_baseline_kappa_new is None:
        linear_baseline_kappa_new = float(linear_baseline_kappa)
    theta_term = theta_term + centered_lhat_increment(
        lhat_new,
        lhat,
        kappa_new=float(linear_baseline_kappa_new),
        kappa=float(linear_baseline_kappa),
        lhat_temp=float(lhat_temp),
    )
    theta_term = float(pa_residual_alpha) * theta_term
    mean_fwd_mu, var_fwd = forward_kernel_mean_variance(scheduler, timestep, x_new)
    mean_fwd_rho = mean_fwd_mu.float()
    prev_t = int(scheduler.previous_timestep(int(timestep)))
    if prev_t >= 0 and abs(float(rho_ratio)) > 1e-12:
        _, _, _, beta_x = ab_outputs_and_beta(
            pipe,
            x_new.to(dtype=latents.dtype),
            prev_t,
            prompt_embeds,
            linear_baseline_kappa=float(linear_baseline_kappa_new),
        )
        theta_x = theta_from_lhat(lhat_new, c_new, eta_new, theta_eps).float()
        residual_guidance_x = float(pa_residual_alpha) * (
            float(gate_power_new)
            * theta_x
            * float(eta_new)
            * float(lhat_temp)
            - float(linear_baseline_kappa_new)
        )
        psi_x = (
            -residual_guidance_x[:, None, None, None]
            * beta_x.float()
        )
        mean_fwd_rho = mean_fwd_mu.float() + float(rho_ratio) * var_fwd * psi_x
    logk_mu = gaussian_log_prob_isotropic(latents, mean_fwd_mu, var_fwd)
    logk_rho = gaussian_log_prob_isotropic(latents, mean_fwd_rho, var_fwd)
    reverse_kernel = logp_base_a - logp_rho
    forward_kernel = logk_rho - logk_mu
    logw = (
        theta_term
        + reverse_kernel
        + forward_kernel
    )
    finite_components = {
        "theta": theta_term.float(),
        "reverse_kernel": reverse_kernel.float(),
        "forward_kernel": forward_kernel.float(),
        "total": logw.float(),
    }

    return x_new, lhat_new, logw.float(), kernel_lr.float(), finite_components


def finite_weight_for_rho(
    rho_ratio: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_new, lhat_new, logw, kernel_lr, _ = finite_weight_diagnostics_for_rho(
        rho_ratio,
        **kwargs,
    )
    return x_new, lhat_new, logw, kernel_lr


def nan_terms(names: tuple[str, ...] | list[str], like: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: torch.full_like(like.float(), float("nan")) for name in names}


def local_expansion_math_components(
    rho_ratio: float,
    *,
    latents: torch.Tensor,
    mu_a: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    forward_displacement: torch.Tensor,
    beta_l: torch.Tensor,
    theta: torch.Tensor,
    eta: float,
    c: float,
    eta_new: float,
    c_new: float,
    lhat: torch.Tensor,
    lhat_new: torch.Tensor,
    z: torch.Tensor,
    lhat_temp: float,
    gate_power: float,
    h_alpha_s: torch.Tensor | None,
    finite_components: dict[str, torch.Tensor] | None,
    surrogate_decomp: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Stage the local expansion against the finite pieces.

    These tensors are diagnostics only.  They intentionally keep some pieces
    exact while approximating one block at a time:

    * gate_* compares exact gate ratio against S/Taylor approximations.
    * forward_* compares exact forward kernel against local alpha/Hessian.
    * total_* ladders isolate which replacement introduces most residual.
    """
    if finite_components is None:
        return nan_terms(MATH_ALL_NAMES, lhat)

    y = latents.float().flatten(1)
    zf = z.float().flatten(1)
    mu_a_f = mu_a.float().flatten(1)
    gate_mu_a_f = gate_mu_a.float().flatten(1)
    gate_mu_b_f = gate_mu_b.float().flatten(1)
    beta = float(lhat_temp) * beta_l.float().flatten(1)
    hmu = forward_displacement.float().flatten(1)
    theta_f = theta.float()
    eta_f = float(eta)
    eta_new_f = float(eta_new)
    gate_pow = float(gate_power)
    rho = torch.tensor(float(rho_ratio), device=latents.device, dtype=torch.float32)
    var_f = _broadcast_particle_scalar(variance.float(), y)
    fvar = forward_variance.float()

    nu_a = y - mu_a_f
    gate_nu_a = y - gate_mu_a_f
    gate_nu_b = y - gate_mu_b_f
    alpha_l = (gate_nu_a.pow(2).sum(dim=1) - gate_nu_b.pow(2).sum(dim=1)) / (2.0 * variance.float())
    if h_alpha_s is None:
        h_alpha_s_f = (
            eta_f * float(lhat_temp) * alpha_l
            + math.log(max(float(c_new), 1e-30) / max(float(c), 1e-30))
            + (eta_new_f - eta_f) * lhat.float()
        )
    else:
        h_alpha_s_f = h_alpha_s.float()

    psi = -gate_pow * theta_f[:, None] * eta_f * beta
    q = gate_pow * theta_f[:, None] * eta_f * beta + rho * psi
    delta = -nu_a - var_f * q + zf

    beta_dot_delta = torch.sum(beta * delta, dim=1)
    beta_dot_z = torch.sum(beta * zf, dim=1)
    s_exact = (
        math.log(max(float(c_new), 1e-30) / max(float(c), 1e-30))
        + eta_new_f * lhat_new.float()
        - eta_f * lhat.float()
    )
    s_local = h_alpha_s_f + eta_f * beta_dot_delta

    gate_taylor2_exact_s = -gate_pow * (
        theta_f * s_exact + 0.5 * theta_f * (1.0 - theta_f) * s_exact.pow(2)
    )
    gate_taylor2_local_s = -gate_pow * (
        theta_f * s_local + 0.5 * theta_f * (1.0 - theta_f) * s_local.pow(2)
    )
    gate_time = -gate_pow * theta_f * h_alpha_s_f
    gate_linear_delta = -gate_pow * theta_f * eta_f * beta_dot_delta
    gate_linear_brownian = -gate_pow * theta_f * eta_f * beta_dot_z
    gate_curvature_delta = (
        -0.5
        * gate_pow
        * theta_f
        * (1.0 - theta_f)
        * (eta_f * beta_dot_delta).pow(2)
    )
    gate_curvature_brownian = (
        -0.5
        * gate_pow
        * theta_f
        * (1.0 - theta_f)
        * (eta_f * beta_dot_z).pow(2)
    )
    gate_retained_delta = gate_time + gate_linear_delta + gate_curvature_delta
    gate_retained_brownian = gate_time + gate_linear_brownian + gate_curvature_brownian

    reverse_linear_delta = torch.sum(q * delta, dim=1)
    reverse_linear_brownian = torch.sum(q * zf, dim=1)
    reverse_retained_alpha = torch.sum(nu_a * q, dim=1) + 0.5 * variance.float() * torch.sum(q * q, dim=1)
    reverse_raw_delta = reverse_retained_alpha + reverse_linear_delta

    forward_alpha = (
        -rho * torch.sum(hmu * psi, dim=1)
        - 0.5 * (rho**2) * fvar * torch.sum(psi * psi, dim=1)
    )
    if surrogate_decomp is None or surrogate_decomp.get("proposal_hessian") is None:
        proposal_hessian = torch.full_like(forward_alpha, float("nan"))
    else:
        proposal_hessian = surrogate_decomp["proposal_hessian"].float()
    forward_alpha_hessian = forward_alpha + proposal_hessian

    finite_reverse = finite_components["reverse_kernel"].float()
    finite_forward = finite_components["forward_kernel"].float()

    out = {
        "s_exact": s_exact.float(),
        "s_local": s_local.float(),
        "s_local_error": (s_local - s_exact).float(),
        "gate_taylor2_exact_s": gate_taylor2_exact_s.float(),
        "gate_taylor2_local_s": gate_taylor2_local_s.float(),
        "gate_retained_delta": gate_retained_delta.float(),
        "gate_retained_brownian": gate_retained_brownian.float(),
        "gate_linear_delta": gate_linear_delta.float(),
        "gate_linear_brownian": gate_linear_brownian.float(),
        "reverse_linear_delta": reverse_linear_delta.float(),
        "reverse_linear_brownian": reverse_linear_brownian.float(),
        "linear_gate_reverse_delta": (gate_linear_delta + reverse_linear_delta).float(),
        "linear_gate_reverse_brownian": (gate_linear_brownian + reverse_linear_brownian).float(),
        "reverse_retained_alpha": reverse_retained_alpha.float(),
        "reverse_raw_delta": reverse_raw_delta.float(),
        "forward_alpha": forward_alpha.float(),
        "forward_alpha_hessian": forward_alpha_hessian.float(),
        "total_gate_taylor2_exact_s": (
            gate_taylor2_exact_s + finite_reverse + finite_forward
        ).float(),
        "total_gate_taylor2_local_s": (
            gate_taylor2_local_s + finite_reverse + finite_forward
        ).float(),
        "total_gate_retained_delta": (
            gate_retained_delta + finite_reverse + finite_forward
        ).float(),
        "total_gate_retained_brownian": (
            gate_retained_brownian + finite_reverse + finite_forward
        ).float(),
        "total_forward_alpha": (
            gate_retained_delta + finite_reverse + forward_alpha
        ).float(),
        "total_forward_alpha_hessian": (
            gate_retained_delta + finite_reverse + forward_alpha_hessian
        ).float(),
    }
    return out


def weighted_variance_np(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(torch.var(x, unbiased=False).item())


def decomposition_stats(
    prefix: str,
    terms: dict[str, torch.Tensor] | None,
    total: torch.Tensor,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Per-term variance/covariance diagnostics for a log-weight decomposition.

    For total L = sum_j T_j, Var(L) = sum_j Cov(T_j, L).  The covariance share
    is therefore the most useful signed attribution of an ESS drop.
    """
    out: dict[str, Any] = {}
    total = total.detach().float()
    total_mask = torch.isfinite(total)
    total_clean = total[total_mask]
    if int(total_clean.numel()) >= 2:
        total_centered = total_clean - total_clean.mean()
        total_var = torch.mean(total_centered * total_centered)
        total_std = torch.sqrt(total_var)
    else:
        total_centered = torch.empty(0, device=total.device, dtype=torch.float32)
        total_var = torch.tensor(float("nan"), device=total.device)
        total_std = torch.tensor(float("nan"), device=total.device)

    out[f"{prefix}_total_var"] = float(total_var.item())
    out[f"{prefix}_total_std"] = float(total_std.item())

    best_name = "none"
    best_abs_cov = -1.0
    best_share = float("nan")
    term_names = SURROGATE_DECOMP_NAMES if names is None else names
    for name in term_names:
        x = None if terms is None else terms.get(name)
        if x is None:
            out[f"{prefix}_{name}_mean"] = float("nan")
            out[f"{prefix}_{name}_std"] = float("nan")
            out[f"{prefix}_{name}_cov_total"] = float("nan")
            out[f"{prefix}_{name}_var_share"] = float("nan")
            out[f"{prefix}_{name}_corr_total"] = float("nan")
            continue
        x = x.detach().float()
        mask = total_mask & torch.isfinite(x)
        if int(mask.sum().item()) < 2:
            out[f"{prefix}_{name}_mean"] = float("nan")
            out[f"{prefix}_{name}_std"] = float("nan")
            out[f"{prefix}_{name}_cov_total"] = float("nan")
            out[f"{prefix}_{name}_var_share"] = float("nan")
            out[f"{prefix}_{name}_corr_total"] = float("nan")
            continue
        xv = x[mask]
        tv = total[mask]
        xc = xv - xv.mean()
        tc = tv - tv.mean()
        var_x = torch.mean(xc * xc)
        std_x = torch.sqrt(var_x)
        var_t = torch.mean(tc * tc)
        std_t = torch.sqrt(var_t)
        cov = torch.mean(xc * tc)
        if float(var_t.item()) > 1e-30:
            share = float((cov / var_t).item())
        else:
            share = float("nan")
        if float((std_x * std_t).item()) > 1e-30:
            corr = float((cov / (std_x * std_t)).item())
        else:
            corr = float("nan")
        cov_f = float(cov.item())
        out[f"{prefix}_{name}_mean"] = float(xv.mean().item())
        out[f"{prefix}_{name}_std"] = float(std_x.item())
        out[f"{prefix}_{name}_cov_total"] = cov_f
        out[f"{prefix}_{name}_var_share"] = share
        out[f"{prefix}_{name}_corr_total"] = corr
        if math.isfinite(cov_f) and abs(cov_f) > best_abs_cov and name != "total":
            best_name = name
            best_abs_cov = abs(cov_f)
            best_share = share

    out[f"{prefix}_top_abs_cov_term"] = best_name
    out[f"{prefix}_top_abs_cov_share"] = best_share
    return out


def _nan_tail_stats(prefix: str, ks: tuple[int, ...]) -> dict[str, float]:
    out = {
        f"{prefix}_mean": float("nan"),
        f"{prefix}_std": float("nan"),
        f"{prefix}_abs_mean": float("nan"),
        f"{prefix}_abs_max": float("nan"),
        f"{prefix}_z_abs_max": float("nan"),
        f"{prefix}_kurtosis": float("nan"),
    }
    for k in ks:
        out[f"{prefix}_top{k}_energy_frac"] = float("nan")
        out[f"{prefix}_top{k}_abs_cov_frac"] = float("nan")
        out[f"{prefix}_top{k}_signed_cov_frac"] = float("nan")
        out[f"{prefix}_trim_top{k}_std"] = float("nan")
        out[f"{prefix}_trim_top{k}_std_ratio"] = float("nan")
        out[f"{prefix}_trim_top{k}_cov_share"] = float("nan")
    return out


def scalar_tail_stats(
    prefix: str,
    x: torch.Tensor | None,
    total: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, float]:
    """Tail diagnostics for deciding whether one scalar term is outlier-driven."""
    if x is None:
        return _nan_tail_stats(prefix, ks)
    x = x.detach().float().reshape(-1)
    total = total.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(total)
    if int(mask.sum().item()) < 2:
        return _nan_tail_stats(prefix, ks)

    xv = x[mask]
    tv = total[mask]
    xc = xv - xv.mean()
    tc = tv - tv.mean()
    var_x = torch.mean(xc * xc)
    std_x = torch.sqrt(var_x)
    var_t = torch.mean(tc * tc)
    cov = torch.mean(xc * tc)
    eps = 1e-30
    z = xc / std_x.clamp_min(eps)
    energy = xc * xc
    cov_contrib = xc * tc
    energy_sum = energy.sum()
    abs_cov_sum = cov_contrib.abs().sum()
    signed_cov_sum = cov_contrib.sum()
    order = torch.argsort(xc.abs(), descending=True)

    out = {
        f"{prefix}_mean": float(xv.mean().item()),
        f"{prefix}_std": float(std_x.item()),
        f"{prefix}_abs_mean": float(xv.abs().mean().item()),
        f"{prefix}_abs_max": float(xv.abs().max().item()),
        f"{prefix}_z_abs_max": float(z.abs().max().item()),
        f"{prefix}_kurtosis": float(torch.mean(z.pow(4)).item()) if float(std_x.item()) > eps else float("nan"),
    }
    n = int(xv.numel())
    for k in ks:
        kk = min(int(k), n)
        top = order[:kk]
        keep = torch.ones(n, device=xv.device, dtype=torch.bool)
        keep[top] = False
        out[f"{prefix}_top{k}_energy_frac"] = (
            float((energy[top].sum() / energy_sum).item())
            if float(energy_sum.item()) > eps
            else float("nan")
        )
        out[f"{prefix}_top{k}_abs_cov_frac"] = (
            float((cov_contrib[top].abs().sum() / abs_cov_sum).item())
            if float(abs_cov_sum.item()) > eps
            else float("nan")
        )
        out[f"{prefix}_top{k}_signed_cov_frac"] = (
            float((cov_contrib[top].sum() / signed_cov_sum).item())
            if abs(float(signed_cov_sum.item())) > eps
            else float("nan")
        )
        if int(keep.sum().item()) >= 2:
            xk = xv[keep]
            tk = tv[keep]
            xkc = xk - xk.mean()
            tkc = tk - tk.mean()
            std_k = torch.sqrt(torch.mean(xkc * xkc))
            var_tk = torch.mean(tkc * tkc)
            cov_k = torch.mean(xkc * tkc)
            out[f"{prefix}_trim_top{k}_std"] = float(std_k.item())
            out[f"{prefix}_trim_top{k}_std_ratio"] = (
                float((std_k / std_x).item()) if float(std_x.item()) > eps else float("nan")
            )
            out[f"{prefix}_trim_top{k}_cov_share"] = (
                float((cov_k / var_tk).item()) if float(var_tk.item()) > eps else float("nan")
            )
        else:
            out[f"{prefix}_trim_top{k}_std"] = float("nan")
            out[f"{prefix}_trim_top{k}_std_ratio"] = float("nan")
            out[f"{prefix}_trim_top{k}_cov_share"] = float("nan")
    return out


def topk_overlap_stats(
    prefix: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, float]:
    """Overlap between the largest centered deviations of two scalar terms."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"{prefix}_top{k}_overlap_count"] = float("nan")
        out[f"{prefix}_top{k}_overlap_frac"] = float("nan")
        out[f"{prefix}_top{k}_overlap_jaccard"] = float("nan")
    if x is None or y is None:
        return out
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return out
    xv = x[mask]
    yv = y[mask]
    x_order = torch.argsort((xv - xv.mean()).abs(), descending=True)
    y_order = torch.argsort((yv - yv.mean()).abs(), descending=True)
    n = int(xv.numel())
    for k in ks:
        kk = min(int(k), n)
        x_top = x_order[:kk]
        y_top = y_order[:kk]
        x_mark = torch.zeros(n, device=xv.device, dtype=torch.bool)
        y_mark = torch.zeros(n, device=xv.device, dtype=torch.bool)
        x_mark[x_top] = True
        y_mark[y_top] = True
        overlap = int((x_mark & y_mark).sum().item())
        union = int((x_mark | y_mark).sum().item())
        out[f"{prefix}_top{k}_overlap_count"] = float(overlap)
        out[f"{prefix}_top{k}_overlap_frac"] = float(overlap) / float(max(kk, 1))
        out[f"{prefix}_top{k}_overlap_jaccard"] = (
            float(overlap) / float(union) if union > 0 else float("nan")
        )
    return out


def scalar_relation_stats(prefix: str, x: torch.Tensor | None, anchor: torch.Tensor | None) -> dict[str, float]:
    if x is None or anchor is None:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_corr_jvp": float("nan"),
            f"{prefix}_abs_corr_jvp_abs": float("nan"),
        }
    x = x.detach().float().reshape(-1)
    anchor = anchor.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(anchor)
    if int(mask.sum().item()) < 2:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_corr_jvp": float("nan"),
            f"{prefix}_abs_corr_jvp_abs": float("nan"),
        }
    xv = x[mask]
    av = anchor[mask]
    return {
        f"{prefix}_mean": float(xv.mean().item()),
        f"{prefix}_std": float(xv.std(unbiased=False).item()),
        f"{prefix}_abs_mean": float(xv.abs().mean().item()),
        f"{prefix}_abs_max": float(xv.abs().max().item()),
        f"{prefix}_corr_jvp": finite_corr(xv, av),
        f"{prefix}_abs_corr_jvp_abs": finite_corr(xv.abs(), av.abs()),
    }


def jvp_tail_diagnostics(
    terms: dict[str, torch.Tensor] | None,
    total: torch.Tensor,
    drift_components: dict[str, torch.Tensor],
) -> dict[str, float]:
    if terms is None:
        return scalar_tail_stats("jvp", None, total)
    jvp = terms.get("proposal_hessian_jvp")
    out = scalar_tail_stats("jvp", jvp, total)
    jvp_core = terms.get("diag_gamma_jbeta_inner")
    out.update(scalar_tail_stats("jvp_core", jvp_core, total))
    out.update(scalar_relation_stats("jvp_core", jvp_core, jvp))
    for src, prefix in (
        ("diag_jvp_scale", "jvp_scale"),
        ("diag_gamma_jbeta_cos", "jvp_cos"),
        ("diag_gamma_delta_norm", "jvp_gamma_delta_norm"),
        ("diag_jbeta_gamma_delta_norm", "jvp_jbeta_delta_norm"),
        ("diag_beta_dot_gamma_delta", "jvp_beta_dot_gamma_delta"),
        ("diag_q_norm", "jvp_q_norm"),
        ("diag_nu0_norm", "jvp_nu0_norm"),
        ("diag_psi_norm", "jvp_psi_norm"),
    ):
        out.update(scalar_relation_stats(prefix, terms.get(src), jvp))
    brownian = drift_components.get("brownian")
    if brownian is not None:
        bf = brownian.detach().float().flatten(1)
        brownian_norm = torch.linalg.vector_norm(bf, dim=1) / math.sqrt(max(int(bf.shape[1]), 1))
    else:
        brownian_norm = None
    out.update(scalar_relation_stats("jvp_brownian_norm_per_sqrt_dim", brownian_norm, jvp))
    return out


def kernel_tail_diagnostics(
    terms: dict[str, torch.Tensor] | None,
    finite_components: dict[str, torch.Tensor] | None,
    surrogate_total: torch.Tensor,
    true_total: torch.Tensor,
) -> dict[str, float]:
    """Tail diagnostics for Gaussian kernel terms and their overlap with JVP tails."""
    jvp = None if terms is None else terms.get("proposal_hessian_jvp")
    surrogate_reverse = None
    surrogate_forward = None
    if terms is not None:
        surrogate_reverse = terms.get("group_reverse_kernel_alpha")
        if surrogate_reverse is None and "reverse_linear" in terms and "reverse_quadratic" in terms:
            surrogate_reverse = terms["reverse_linear"] + terms["reverse_quadratic"]
        surrogate_forward = terms.get("group_forward_kernel_alpha")
        if surrogate_forward is None and "forward_linear" in terms and "forward_quadratic" in terms:
            surrogate_forward = terms["forward_linear"] + terms["forward_quadratic"]

    out: dict[str, float] = {}
    out.update(scalar_tail_stats("surrogate_reverse_kernel", surrogate_reverse, surrogate_total))
    out.update(scalar_tail_stats("surrogate_forward_kernel", surrogate_forward, surrogate_total))
    out.update(topk_overlap_stats("surrogate_reverse_kernel_jvp", surrogate_reverse, jvp))
    out.update(topk_overlap_stats("surrogate_forward_kernel_jvp", surrogate_forward, jvp))

    finite_reverse = None if finite_components is None else finite_components.get("reverse_kernel")
    finite_forward = None if finite_components is None else finite_components.get("forward_kernel")
    out.update(scalar_tail_stats("finite_reverse_kernel_tail", finite_reverse, true_total))
    out.update(scalar_tail_stats("finite_forward_kernel_tail", finite_forward, true_total))
    out.update(topk_overlap_stats("finite_reverse_kernel_jvp", finite_reverse, jvp))
    out.update(topk_overlap_stats("finite_forward_kernel_jvp", finite_forward, jvp))
    out.update(topk_overlap_stats("finite_reverse_surrogate_reverse_kernel", finite_reverse, surrogate_reverse))
    out.update(topk_overlap_stats("finite_forward_surrogate_forward_kernel", finite_forward, surrogate_forward))
    return out


def vector_summary_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    mask = torch.isfinite(x)
    if int(mask.sum().item()) < 1:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_var": float("nan"),
        }
    xv = x[mask]
    var = torch.var(xv, unbiased=False)
    return {
        f"{prefix}_mean": float(xv.mean().item()),
        f"{prefix}_std": float(torch.sqrt(var).item()),
        f"{prefix}_var": float(var.item()),
    }


def comparison_stats(prefix: str, approx: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Diagnostics for approx-vs-target beyond correlation."""
    approx = approx.detach().float()
    target = target.detach().float()
    mask = torch.isfinite(approx) & torch.isfinite(target)
    if int(mask.sum().item()) < 2:
        return {
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_resid_std": float("nan"),
            f"{prefix}_affine_slope": float("nan"),
            f"{prefix}_affine_resid_std": float("nan"),
            f"{prefix}_affine_resid_rel_std": float("nan"),
        }
    x = approx[mask]
    y = target[mask]
    resid = x - y
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = torch.mean(x_c * x_c)
    var_y = torch.mean(y_c * y_c)
    if float(var_x.item()) <= 1e-30:
        slope = torch.tensor(float("nan"), device=x.device)
        affine_resid = torch.full_like(y, float("nan"))
    else:
        slope = torch.mean(x_c * y_c) / var_x
        intercept = y.mean() - slope * x.mean()
        affine_resid = y - (slope * x + intercept)
    target_std = torch.sqrt(var_y).clamp_min(1e-30)
    affine_resid_std = torch.std(affine_resid, unbiased=False)
    return {
        f"{prefix}_rmse": float(torch.sqrt(torch.mean(resid * resid)).item()),
        f"{prefix}_resid_std": float(torch.std(resid, unbiased=False).item()),
        f"{prefix}_affine_slope": float(slope.item()),
        f"{prefix}_affine_resid_std": float(affine_resid_std.item()),
        f"{prefix}_affine_resid_rel_std": float((affine_resid_std / target_std).item()),
    }


def _nan_agreement_tail_stats(prefix: str, ks: tuple[int, ...]) -> dict[str, float]:
    out = {
        f"{prefix}_bad_score": float("nan"),
        f"{prefix}_raw_resid_std": float("nan"),
        f"{prefix}_raw_resid_abs_max": float("nan"),
        f"{prefix}_affine_resid_std": float("nan"),
        f"{prefix}_affine_resid_abs_max": float("nan"),
        f"{prefix}_affine_resid_z_abs_max": float("nan"),
        f"{prefix}_affine_resid_kurtosis": float("nan"),
    }
    for k in ks:
        out[f"{prefix}_bad_top{k}_raw_resid_energy_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_affine_resid_energy_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_approx_var_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_target_var_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_abs_cross_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_approx_weight_mass_frac"] = float("nan")
        out[f"{prefix}_bad_top{k}_target_weight_mass_frac"] = float("nan")
        out[f"{prefix}_trim_bad_top{k}_corr"] = float("nan")
        out[f"{prefix}_trim_bad_top{k}_affine_resid_rel_std"] = float("nan")
        out[f"{prefix}_trim_bad_top{k}_approx_ess_frac"] = float("nan")
        out[f"{prefix}_trim_bad_top{k}_target_ess_frac"] = float("nan")
    return out


def _centered_energy_frac(x: torch.Tensor, top: torch.Tensor, eps: float = 1e-30) -> float:
    xc = x - x.mean()
    energy = xc * xc
    denom = float(energy.sum().item())
    return float((energy[top].sum() / energy.sum()).item()) if denom > eps else float("nan")


def _softmax_mass_frac(logw: torch.Tensor, top: torch.Tensor, eps: float = 1e-30) -> float:
    if int(logw.numel()) < 1:
        return float("nan")
    w = torch.softmax(logw - logw.max(), dim=0)
    denom = float(w.sum().item())
    return float(w[top].sum().item() / denom) if denom > eps else float("nan")


def _affine_resid_rel_std(approx: torch.Tensor, target: torch.Tensor) -> float:
    if int(approx.numel()) < 2:
        return float("nan")
    out = comparison_stats("_tmp", approx, target)
    return float(out["_tmp_affine_resid_rel_std"])


def agreement_tail_stats(
    prefix: str,
    approx: torch.Tensor,
    target: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, float]:
    """Approx-vs-target tail diagnostics.

    The bad set is ranked by the affine residual, i.e. particles where the
    finite target cannot be predicted even after allowing a global scale/shift
    of the surrogate.  The top-k mass/variance columns show whether those bad
    particles also dominate the actual weighting problem.
    """
    approx = approx.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    mask = torch.isfinite(approx) & torch.isfinite(target)
    if int(mask.sum().item()) < 2:
        return _nan_agreement_tail_stats(prefix, ks)

    x = approx[mask]
    y = target[mask]
    raw_resid = x - y
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = torch.mean(x_c * x_c)
    if float(var_x.item()) > 1e-30:
        slope = torch.mean(x_c * y_c) / var_x
        intercept = y.mean() - slope * x.mean()
        affine_resid = y - (slope * x + intercept)
        bad_score = affine_resid.abs()
        bad_score_name = "affine_resid"
    else:
        affine_resid = torch.full_like(y, float("nan"))
        bad_score = raw_resid.abs()
        bad_score_name = "raw_resid"

    raw_c = raw_resid - raw_resid.mean()
    raw_energy = raw_c * raw_c
    raw_energy_sum = raw_energy.sum()
    finite_aff = torch.isfinite(affine_resid)
    if bool(finite_aff.all().item()):
        aff_c = affine_resid - affine_resid.mean()
        aff_std = torch.sqrt(torch.mean(aff_c * aff_c))
        aff_energy = aff_c * aff_c
        aff_energy_sum = aff_energy.sum()
        aff_z = aff_c / aff_std.clamp_min(1e-30)
        aff_abs_max = float(affine_resid.abs().max().item())
        aff_z_abs_max = float(aff_z.abs().max().item())
        aff_kurtosis = (
            float(torch.mean(aff_z.pow(4)).item())
            if float(aff_std.item()) > 1e-30
            else float("nan")
        )
        aff_std_f = float(aff_std.item())
    else:
        aff_energy = torch.full_like(raw_energy, float("nan"))
        aff_energy_sum = torch.tensor(float("nan"), device=x.device)
        aff_abs_max = float("nan")
        aff_z_abs_max = float("nan")
        aff_kurtosis = float("nan")
        aff_std_f = float("nan")

    cross_abs = (x_c * y_c).abs()
    cross_abs_sum = cross_abs.sum()
    order = torch.argsort(bad_score, descending=True)
    n = int(x.numel())
    eps = 1e-30
    out = {
        f"{prefix}_bad_score": bad_score_name,
        f"{prefix}_raw_resid_std": float(torch.std(raw_resid, unbiased=False).item()),
        f"{prefix}_raw_resid_abs_max": float(raw_resid.abs().max().item()),
        f"{prefix}_affine_resid_std": aff_std_f,
        f"{prefix}_affine_resid_abs_max": aff_abs_max,
        f"{prefix}_affine_resid_z_abs_max": aff_z_abs_max,
        f"{prefix}_affine_resid_kurtosis": aff_kurtosis,
    }
    for k in ks:
        kk = min(int(k), n)
        top = order[:kk]
        keep = torch.ones(n, device=x.device, dtype=torch.bool)
        keep[top] = False
        raw_denom = float(raw_energy_sum.item())
        out[f"{prefix}_bad_top{k}_raw_resid_energy_frac"] = (
            float((raw_energy[top].sum() / raw_energy_sum).item())
            if raw_denom > eps
            else float("nan")
        )
        aff_denom = float(aff_energy_sum.item())
        out[f"{prefix}_bad_top{k}_affine_resid_energy_frac"] = (
            float((aff_energy[top].sum() / aff_energy_sum).item())
            if math.isfinite(aff_denom) and aff_denom > eps
            else float("nan")
        )
        out[f"{prefix}_bad_top{k}_approx_var_frac"] = _centered_energy_frac(x, top)
        out[f"{prefix}_bad_top{k}_target_var_frac"] = _centered_energy_frac(y, top)
        cross_denom = float(cross_abs_sum.item())
        out[f"{prefix}_bad_top{k}_abs_cross_frac"] = (
            float((cross_abs[top].sum() / cross_abs_sum).item())
            if cross_denom > eps
            else float("nan")
        )
        out[f"{prefix}_bad_top{k}_approx_weight_mass_frac"] = _softmax_mass_frac(x, top)
        out[f"{prefix}_bad_top{k}_target_weight_mass_frac"] = _softmax_mass_frac(y, top)
        if int(keep.sum().item()) >= 2:
            xk = x[keep]
            yk = y[keep]
            out[f"{prefix}_trim_bad_top{k}_corr"] = finite_corr(xk, yk)
            out[f"{prefix}_trim_bad_top{k}_affine_resid_rel_std"] = _affine_resid_rel_std(xk, yk)
            out[f"{prefix}_trim_bad_top{k}_approx_ess_frac"] = global_ess_frac_from_logw(xk)
            out[f"{prefix}_trim_bad_top{k}_target_ess_frac"] = global_ess_frac_from_logw(yk)
        else:
            out[f"{prefix}_trim_bad_top{k}_corr"] = float("nan")
            out[f"{prefix}_trim_bad_top{k}_affine_resid_rel_std"] = float("nan")
            out[f"{prefix}_trim_bad_top{k}_approx_ess_frac"] = float("nan")
            out[f"{prefix}_trim_bad_top{k}_target_ess_frac"] = float("nan")
    return out


def approximation_pair_stats(
    prefix: str,
    approx: torch.Tensor,
    target: torch.Tensor,
    *,
    include_ess: bool = False,
) -> dict[str, float]:
    out = vector_summary_stats(prefix, approx)
    out[f"{prefix}_corr"] = finite_corr(approx, target)
    out.update(comparison_stats(prefix, approx, target))
    if include_ess:
        mask = torch.isfinite(approx)
        if int(mask.sum().item()) == int(approx.numel()) and int(approx.numel()) >= 1:
            out[f"{prefix}_ess_frac"] = global_ess_frac_from_logw(approx)
        else:
            out[f"{prefix}_ess_frac"] = float("nan")
    return out


def expansion_math_stats(
    prefix: str,
    math_terms: dict[str, torch.Tensor] | None,
    finite_terms: dict[str, torch.Tensor] | None,
    true_total: torch.Tensor,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if math_terms is None or finite_terms is None:
        return out
    for name in MATH_STANDALONE_NAMES:
        x = math_terms.get(name)
        if x is not None:
            out.update(vector_summary_stats(f"{prefix}_{name}", x))
    for approx_name, target_name in MATH_COMPONENT_TARGETS:
        approx = math_terms.get(approx_name)
        target = finite_terms.get(target_name)
        if approx is not None and target is not None:
            out.update(
                approximation_pair_stats(
                    f"{prefix}_{approx_name}_vs_{target_name}",
                    approx,
                    target,
                )
            )
    for name in MATH_TOTAL_NAMES:
        approx = math_terms.get(name)
        if approx is not None:
            out.update(
                approximation_pair_stats(
                    f"{prefix}_{name}_vs_true",
                    approx,
                    true_total,
                    include_ess=True,
                )
            )
    return out


def make_rho_monitor_plot(
    rows: list[dict[str, Any]],
    out_dir: Path,
    resample_ess: float,
    no_resample_last_steps: int = 0,
    mask_final_inc_ess: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(5, 2, figsize=(15, 18), dpi=140)

    def plot_values(key: str, *, incremental: bool = False) -> np.ndarray:
        values = np.asarray(
            [float(r.get(key, float("nan"))) for r in rows],
            dtype=float,
        )
        final_k = max(0, int(no_resample_last_steps))
        if (
            incremental
            and mask_final_inc_ess
            and final_k > 0
            and values.size > 0
        ):
            values[max(0, values.size - final_k):] = 1.0
        return values

    axes[0, 0].plot(step, [r["rho_used"] for r in rows], label="rho ratio")
    axes[0, 0].plot(step, [r["rho_objective_start"] for r in rows], label="rho obj start")
    axes[0, 0].plot(step, [r["rho_objective_final"] for r in rows], label="rho obj final")
    if any(np.isfinite(float(r.get("rho_total_ess_final", np.nan))) for r in rows):
        axes[0, 0].plot(step, [r.get("rho_total_ess_start", float("nan")) for r in rows], ls=":", label="surrogate total ESS start")
        axes[0, 0].plot(step, [r.get("rho_total_ess_final", float("nan")) for r in rows], ls="--", label="surrogate total ESS final")
        axes[0, 0].plot(step, [r.get("rho_total_ess_rho0", float("nan")) for r in rows], ls="-.", label="surrogate total ESS rho=0")
    axes[0, 0].set_title("Rho Optimization")
    axes[0, 0].set_xlabel("reverse step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(step, [r["guidance_mean"] for r in rows], label="gate_power*theta*eta*lhat_temp")
    axes[0, 1].plot(
        step,
        [r["linear_baseline_kappa"] for r in rows],
        label="linear baseline kappa",
    )
    axes[0, 1].plot(
        step,
        [r["residual_guidance_mean"] for r in rows],
        label="target guidance - kappa",
    )
    axes[0, 1].plot(
        step,
        [r["proposal_guidance_mean"] for r in rows],
        label="kappa + (1-rho)*(target-kappa)",
    )
    axes[0, 1].set_title("Gate vs Proposal Guidance")
    axes[0, 1].set_xlabel("reverse step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        step,
        plot_values("inc_ess_frac", incremental=True),
        label="incremental",
    )
    axes[1, 0].plot(step, [r.get("cess_ess_frac", float("nan")) for r in rows], label="conditional")
    axes[1, 0].plot(step, [r.get("cess_ess_true_frac", float("nan")) for r in rows], lw=1.4, label="finite true conditional")
    axes[1, 0].plot(
        step,
        plot_values("inc_ess_surrogate_frac", incremental=True),
        lw=1.2,
        label="local surrogate",
    )
    axes[1, 0].plot(
        step,
        plot_values("inc_ess_true_frac", incremental=True),
        lw=1.0,
        alpha=0.55,
        label="finite true inc rho",
    )
    axes[1, 0].plot(step, [r.get("cum_ess_true_frac", float("nan")) for r in rows], lw=1.7, label="finite true full total rho")
    axes[1, 0].plot(step, [r.get("cum_ess_true_active_frac", float("nan")) for r in rows], lw=1.5, label="finite true active total")
    axes[1, 0].plot(step, [r.get("cum_ess_true_rho0_frac", float("nan")) for r in rows], lw=1.5, ls=":", label="finite true full total rho=0")
    axes[1, 0].plot(step, [r.get("surrogate_total_ess_frac", float("nan")) for r in rows], alpha=0.65, label="surrogate full total rho")
    axes[1, 0].plot(step, [r.get("surrogate_total_ess_rho0_frac", float("nan")) for r in rows], alpha=0.65, ls=":", label="surrogate full total rho=0")
    axes[1, 0].plot(step, [r["cum_ess_frac"] for r in rows], label="active cumulative before resample")
    axes[1, 0].plot(step, [r.get("full_cum_ess_frac", float("nan")) for r in rows], label="full cumulative with residual", alpha=0.65)
    axes[1, 0].plot(step, [r["cum_ess_after_resample"] for r in rows], ls="--", label="after resample")
    if resample_ess <= 1.0:
        axes[1, 0].axhline(resample_ess, color="tab:red", ls=":", label="threshold")
    rearm_ess = max(float(r.get("resample_ess_rearm", 0.0)) for r in rows)
    if rearm_ess > 0.0:
        axes[1, 0].axhline(
            rearm_ess,
            color="tab:orange",
            ls="--",
            label="hysteresis rearm",
        )
    rs = [r for r in rows if int(r["resampled"]) == 1]
    if rs:
        axes[1, 0].scatter(
            [r["step"] for r in rs],
            [r.get("resample_ess_used", r["cum_ess_frac"]) for r in rs],
            s=18,
            c="black",
            label="resample",
        )
    blocked_rs = [r for r in rows if int(r.get("resample_hysteresis_blocked", 0)) == 1]
    if blocked_rs:
        axes[1, 0].scatter(
            [r["step"] for r in blocked_rs],
            [r.get("resample_ess_used", r["cum_ess_frac"]) for r in blocked_rs],
            s=28,
            marker="x",
            c="tab:orange",
            label="suppressed by hysteresis",
        )
    axes[1, 0].set_ylim(0.0, 1.01)
    final_k = max(0, int(no_resample_last_steps))
    if mask_final_inc_ess and final_k > 0 and step.size > 0:
        mask_start_index = max(0, step.size - final_k)
        axes[1, 0].axvspan(
            step[mask_start_index] - 0.5,
            step[-1] + 0.5,
            color="0.75",
            alpha=0.18,
            label="incESS display mask",
        )
        axes[1, 0].set_title(
            f"ESS/N (incremental curves displayed as 1 in final {final_k}; CSV unchanged)"
        )
    else:
        axes[1, 0].set_title("ESS/N")
    axes[1, 0].set_xlabel("reverse step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(step, [r["std_logw_used"] for r in rows], label="used for SMC")
    axes[1, 1].plot(step, [r["std_logw_true"] for r in rows], label="finite Gaussian diagnostic")
    axes[1, 1].plot(step, [r.get("std_logw_true_active", float("nan")) for r in rows], label="finite diagnostic active")
    axes[1, 1].plot(step, [r["std_logw_true_rho0"] for r in rows], ls=":", label="finite diagnostic rho=0")
    axes[1, 1].plot(step, [r["std_logw_surrogate"] for r in rows], label="local surrogate")
    axes[1, 1].set_title("Log-Weight Spread")
    axes[1, 1].set_xlabel("reverse step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(step, [r["true_var_rho0"] for r in rows], label="finite var rho=0")
    axes[2, 0].plot(step, [r["true_var_rho"] for r in rows], label="finite var rho")
    axes[2, 0].plot(step, [r.get("cum_true_var_rho0", float("nan")) for r in rows], ls=":", label="finite total var rho=0")
    axes[2, 0].plot(step, [r.get("cum_true_var_rho", float("nan")) for r in rows], ls="--", label="finite total var rho")
    axes[2, 0].plot(step, [r["used_var_rho"] for r in rows], label="used var rho")
    axes[2, 0].plot(step, [r["surrogate_var_rho"] for r in rows], label="surrogate var rho")
    axes[2, 0].set_title("Finite vs Surrogate Variance")
    axes[2, 0].set_xlabel("reverse step")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(step, [r["corr_surrogate_true"] for r in rows], label="corr(local, finite)")
    axes[2, 1].axhline(0.0, color="black", lw=0.8)
    axes[2, 1].set_ylim(-1.05, 1.05)
    axes[2, 1].set_title("Surrogate/True Correlation")
    axes[2, 1].set_xlabel("reverse step")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    axes[3, 0].plot(step, [r["mean_lhat"] for r in rows], label="mean lhat")
    axes[3, 0].fill_between(
        step,
        [r["mean_lhat"] - r["std_lhat"] for r in rows],
        [r["mean_lhat"] + r["std_lhat"] for r in rows],
        alpha=0.2,
    )
    axes[3, 0].set_title("lhat Mean +/- Std")
    axes[3, 0].set_xlabel("reverse step")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend()

    axes[3, 1].plot(step, [r["mean_theta"] for r in rows], label="mean theta")
    axes[3, 1].plot(step, [r["min_theta"] for r in rows], ls="--", label="min")
    axes[3, 1].plot(step, [r["max_theta"] for r in rows], ls="--", label="max")
    axes[3, 1].set_title("Theta")
    axes[3, 1].set_xlabel("reverse step")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend()

    beta_plot_keys = [
        ("mean_gl2_per_dim", "E||T beta||^2/d"),
        ("std_gl2_per_dim", "Std(||T beta||^2)/d"),
        ("beta_coord_var_mean", "mean coord Var(T beta)"),
        ("beta_mean_norm2_per_dim", "||E[T beta]||^2/d"),
        ("beta_pair_dist_mean_per_sqrt_dim", "pair dist/sqrt(d)"),
    ]
    for key, label in beta_plot_keys:
        if key in rows[0]:
            vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
            vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-30), np.nan)
            axes[4, 0].semilogy(step, vals, label=label)
    if rs:
        yvals = [max(float(r.get("beta_coord_var_mean", r.get("mean_gl2_per_dim", 1e-30))), 1e-30) for r in rs]
        axes[4, 0].scatter([r["step"] for r in rs], yvals, s=18, c="black", label="resample")
    axes[4, 0].set_title("Tempered Beta_l Spread Across Particles")
    axes[4, 0].set_xlabel("reverse step")
    axes[4, 0].grid(True, alpha=0.3)
    axes[4, 0].legend(fontsize=7)

    axes[4, 1].plot(step, [r["c_t"] for r in rows], label="c")
    axes[4, 1].plot(step, [r["eta_t"] for r in rows], label="eta")
    axes[4, 1].plot(step, [r["lhat_temp"] for r in rows], label="lhat_temp")
    axes[4, 1].plot(step, [r["gate_power"] for r in rows], label="gate_power")
    axes[4, 1].set_title("Parameters")
    axes[4, 1].set_xlabel("reverse step")
    axes[4, 1].grid(True, alpha=0.3)
    axes[4, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_rho_monitor.png", bbox_inches="tight")
    plt.close(fig)


def make_genealogy_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "unique_initial_ancestors" not in rows[0]:
        return
    step = np.asarray([int(r["step"]) for r in rows], dtype=int)

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=140, sharex=True)
    axes[0].plot(step, vals("unique_initial_ancestors"), lw=1.8, label="unique roots")
    axes[0].plot(step, vals("initial_ancestor_effective_roots"), lw=1.5, label="ESS roots")
    axes[0].set_ylabel("particles")
    axes[0].set_title("Initial Ancestors Still Represented")
    axes[0].legend(fontsize=8)

    axes[1].plot(step, vals("max_initial_ancestor_copies"), lw=1.8, color="tab:orange", label="max copies")
    axes[1].plot(
        step,
        vals("mean_initial_ancestor_copies_nonzero"),
        lw=1.5,
        color="tab:green",
        label="mean copies among live roots",
    )
    axes[1].set_ylabel("copies")
    axes[1].set_title("Ancestor Concentration")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, vals("initial_ancestor_entropy_frac"), lw=1.8, label="root entropy / log N")
    axes[2].plot(step, vals("cum_ess_frac"), lw=1.4, label="active cumulative ESS/N")
    axes[2].plot(step, vals("full_cum_ess_frac"), lw=1.4, alpha=0.75, label="full cumulative ESS/N")
    resampled = vals("resampled")
    rs = np.isfinite(resampled) & (resampled > 0)
    if bool(rs.any()):
        axes[2].scatter(step[rs], np.full(int(rs.sum()), 0.02), s=16, c="black", label="resample")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_ylabel("fraction")
    axes[2].set_xlabel("reverse step")
    axes[2].set_title("Genealogy vs Weight Degeneracy")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_genealogy.png", bbox_inches="tight")
    plt.close(fig)


def make_gate_schedule_diagnostic_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot the gate in its natural logit coordinate and per-step velocity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: float(r["step"]))
    if not rows:
        return

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    step = vals("step")
    c_t = vals("c_t")
    eta_t = vals("eta_t")
    theta = vals("mean_theta")
    theta_min = vals("min_theta")
    theta_max = vals("max_theta")
    eps = 1e-8
    theta_safe = np.clip(theta, eps, 1.0 - eps)
    theta_logit = np.log(theta_safe) - np.log1p(-theta_safe)
    log_c = np.log(np.clip(c_t, eps, None))

    delta_logit = np.full_like(theta_logit, np.nan)
    delta_log_c = np.full_like(log_c, np.nan)
    delta_logit[1:] = np.diff(theta_logit)
    delta_log_c[1:] = np.diff(log_c)
    data_velocity = delta_logit - delta_log_c
    sensitivity = vals("gate_power") * eta_t * vals("lhat_temp") * theta
    resampled = vals("resampled") > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140, constrained_layout=True)

    ax = axes[0, 0]
    eta_ax = ax.twinx()
    line_c = ax.plot(step, log_c, label="log(c_t)")
    line_kappa = ax.plot(
        step,
        vals("linear_baseline_kappa"),
        color="tab:green",
        alpha=0.8,
        label="kappa_t",
    )
    line_eta = eta_ax.plot(step, eta_t, color="tab:orange", label="eta_t")
    ax.set_ylabel("log(c_t)")
    eta_ax.set_ylabel("eta_t")
    ax.set_title("Deterministic Gate Schedules")
    schedule_lines = line_c + line_kappa + line_eta
    ax.legend(
        schedule_lines,
        [x.get_label() for x in schedule_lines],
        fontsize=8,
    )

    ax = axes[0, 1]
    logit_ax = ax.twinx()
    line_theta = ax.plot(step, theta, label="mean theta")
    band = ax.fill_between(step, theta_min, theta_max, alpha=0.18, label="particle min/max")
    line_logit = logit_ax.plot(
        step,
        theta_logit,
        color="tab:red",
        alpha=0.8,
        label="logit(mean theta)",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("theta")
    logit_ax.set_ylabel("logit(mean theta)")
    ax.set_title("Gate Level")
    ax.legend(
        line_theta + [band] + line_logit,
        [x.get_label() for x in line_theta] + ["particle min/max"] + [x.get_label() for x in line_logit],
        fontsize=8,
    )

    ax = axes[1, 0]
    ax.plot(step, delta_logit, label="delta logit(mean theta)")
    ax.plot(step, delta_log_c, label="delta log(c_t)")
    ax.plot(step, data_velocity, label="remainder: eta*lhat + population")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("change per reverse step")
    ax.set_title("Gate-Logit Velocity")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ess_ax = ax.twinx()
    line_sens = ax.plot(
        step,
        sensitivity,
        label="gate_power * eta * lhat_temp * theta",
    )
    line_inc = ess_ax.plot(step, vals("inc_ess_surrogate_frac"), color="tab:green", label="incESS/N")
    line_cess = ess_ax.plot(step, vals("cess_ess_frac"), color="tab:purple", label="CESS/N")
    scatter = None
    if bool(resampled.any()):
        scatter = ess_ax.scatter(
            step[resampled],
            vals("cess_ess_frac")[resampled],
            color="black",
            s=18,
            label="resample",
            zorder=4,
        )
    ax.set_ylabel("local gate sensitivity")
    ess_ax.set_ylabel("ESS/N")
    ess_ax.set_ylim(-0.02, 1.02)
    ax.set_title("Sensitivity and Local Degeneracy")
    handles = line_sens + line_inc + line_cess + ([scatter] if scatter is not None else [])
    ax.legend(handles, [x.get_label() for x in handles], fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("reverse step")
        ax.grid(True, alpha=0.25)

    fig.savefig(out_dir / "smc_pa_gate_gate_schedule_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_rho_surrogate_optimization_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Dedicated before/after view of the surrogate quantity optimized over rho."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows:
        return
    step = np.asarray([float(r["step"]) for r in rows], dtype=float)

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    ess_start = vals("rho_total_ess_start")
    ess_final = vals("rho_total_ess_final")
    ess_rho0 = vals("rho_total_ess_rho0")
    use_ess = bool(np.isfinite(ess_final).any())

    if use_ess:
        before = ess_start
        after = ess_final
        rho0 = ess_rho0
        gain = after - before
        ylabel = "ESS / N"
        main_title = "Surrogate Total ESS Before/After Rho Optimization"
        gain_title = "Surrogate ESS Gain From Rho"
        gain_ylabel = "delta ESS / N"
        better_note = "higher is better"
        ylim_bounds = (0.0, 1.005)
    else:
        before = vals("rho_objective_start")
        after = vals("rho_objective_final")
        rho0 = vals("rho_objective_rho0")
        gain = before - after
        ylabel = "objective"
        main_title = "Surrogate Objective Before/After Rho Optimization"
        gain_title = "Surrogate Objective Reduction From Rho"
        gain_ylabel = "objective start - final"
        better_note = "lower is better"
        ylim_bounds = (0.0, None)

    finite_pair = np.isfinite(before) & np.isfinite(after)
    if not bool(finite_pair.any()):
        return

    abs_gain = np.abs(gain)
    finite_gain = abs_gain[np.isfinite(abs_gain)]
    gain_max = float(np.max(finite_gain)) if finite_gain.size else 0.0
    active = np.isfinite(abs_gain) & (abs_gain > max(1e-5, 0.05 * gain_max))
    if bool(active.any()):
        idx = np.where(active)[0]
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(idx) - 1]])
        best_start = int(idx[starts[0]])
        best_end = int(idx[ends[0]])
        best_score = -float("inf")
        for s, e in zip(starts, ends):
            block = idx[s : e + 1]
            score = float(np.nansum(abs_gain[block]))
            if score > best_score:
                best_score = score
                best_start = int(block[0])
                best_end = int(block[-1])
        lo_i = max(0, best_start - 10)
        hi_i = min(len(step) - 1, best_end + 10)
    else:
        lo_i, hi_i = 0, len(step) - 1

    def set_zoom_ylim(ax: plt.Axes, arrays: list[np.ndarray], *, min_span: float) -> None:
        packed = np.concatenate([a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()])
        if packed.size == 0:
            return
        lo = float(np.min(packed))
        hi = float(np.max(packed))
        span = max(hi - lo, min_span)
        mid = 0.5 * (lo + hi)
        lo = mid - 0.55 * span
        hi = mid + 0.55 * span
        lower, upper = ylim_bounds
        if lower is not None:
            lo = max(float(lower), lo)
        if upper is not None:
            hi = min(float(upper), hi)
        if hi > lo:
            ax.set_ylim(lo, hi)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=160, sharex=False)

    axes[0].plot(step, before, lw=1.8, label="before rho opt")
    axes[0].plot(step, after, lw=1.8, label="after rho opt")
    if np.isfinite(rho0).any():
        axes[0].plot(step, rho0, lw=1.2, ls=":", label="rho=0")
    axes[0].set_title(f"{main_title} ({better_note})")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="best")
    set_zoom_ylim(axes[0], [before, after, rho0], min_span=0.05 if use_ess else 1e-4)

    win = slice(lo_i, hi_i + 1)
    axes[1].plot(step[win], before[win], lw=2.0, label="before")
    axes[1].plot(step[win], after[win], lw=2.0, label="after")
    if np.isfinite(rho0[win]).any():
        axes[1].plot(step[win], rho0[win], lw=1.2, ls=":", label="rho=0")
    axes[1].set_title("Zoom: Rho-Active Window")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="best")
    set_zoom_ylim(axes[1], [before[win], after[win], rho0[win]], min_span=0.05 if use_ess else 1e-4)

    axes[2].plot(step, gain, color="tab:green", lw=1.8, label="after - before" if use_ess else "before - after")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title(gain_title)
    axes[2].set_xlabel("reverse step")
    axes[2].set_ylabel(gain_ylabel)
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(loc="best")
    finite_g = gain[np.isfinite(gain)]
    if finite_g.size:
        lo = min(0.0, float(np.min(finite_g)))
        hi = max(0.0, float(np.max(finite_g)))
        span = max(hi - lo, 1e-5)
        axes[2].set_ylim(lo - 0.06 * span, hi + 0.08 * span)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_surrogate_rho_optimization.png", bbox_inches="tight")
    plt.close(fig)


def make_rho_inc_ess_optimization_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Dedicated before/after view of rho's effect on current incremental ESS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows:
        return
    step = np.asarray([float(r["step"]) for r in rows], dtype=float)

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    before = vals("rho_inc_ess_start")
    after = vals("rho_inc_ess_final")
    rho0 = vals("rho_inc_ess_rho0")
    if not bool((np.isfinite(before) & np.isfinite(after)).any()):
        return

    gain = after - before
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=160, sharex=False)

    axes[0].plot(step, before, lw=1.8, label="before rho opt")
    axes[0].plot(step, after, lw=1.8, label="after rho opt")
    if np.isfinite(rho0).any():
        axes[0].plot(step, rho0, lw=1.2, ls=":", label="rho=0")
    axes[0].set_ylim(0.0, 1.005)
    axes[0].set_title("Surrogate Incremental ESS Before/After Rho Optimization")
    axes[0].set_ylabel("inc ESS / N")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="best")

    abs_gain = np.abs(gain)
    finite_gain = abs_gain[np.isfinite(abs_gain)]
    gain_max = float(np.max(finite_gain)) if finite_gain.size else 0.0
    active = np.isfinite(abs_gain) & (abs_gain > max(1e-5, 0.05 * gain_max))
    if bool(active.any()):
        idx = np.where(active)[0]
        lo_i = max(0, int(idx[0]) - 10)
        hi_i = min(len(step) - 1, int(idx[-1]) + 10)
    else:
        lo_i, hi_i = 0, len(step) - 1
    win = slice(lo_i, hi_i + 1)
    axes[1].plot(step[win], before[win], lw=2.0, label="before")
    axes[1].plot(step[win], after[win], lw=2.0, label="after")
    if np.isfinite(rho0[win]).any():
        axes[1].plot(step[win], rho0[win], lw=1.2, ls=":", label="rho=0")
    finite_zoom = np.concatenate(
        [a[np.isfinite(a)] for a in (before[win], after[win], rho0[win]) if np.isfinite(a).any()]
    )
    if finite_zoom.size:
        lo = max(0.0, float(np.min(finite_zoom)) - 0.03)
        hi = min(1.005, float(np.max(finite_zoom)) + 0.03)
        if hi > lo:
            axes[1].set_ylim(lo, hi)
    axes[1].set_title("Zoom: Rho-Active Window")
    axes[1].set_ylabel("inc ESS / N")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="best")

    axes[2].plot(step, gain, color="tab:green", lw=1.8, label="after - before")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("Surrogate Incremental ESS Gain From Rho")
    axes[2].set_xlabel("reverse step")
    axes[2].set_ylabel("delta inc ESS / N")
    axes[2].grid(True, alpha=0.28)
    axes[2].legend(loc="best")
    finite_g = gain[np.isfinite(gain)]
    if finite_g.size:
        lo = min(0.0, float(np.min(finite_g)))
        hi = max(0.0, float(np.max(finite_g)))
        span = max(hi - lo, 1e-5)
        axes[2].set_ylim(lo - 0.06 * span, hi + 0.08 * span)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_surrogate_rho_inc_ess_optimization.png", bbox_inches="tight")
    plt.close(fig)


def make_inc_ess_agreement_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Finite-vs-surrogate comparison restricted to current incremental weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows:
        return
    step = np.asarray([float(r["step"]) for r in rows], dtype=float)

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    surrogate = vals("inc_ess_surrogate_frac")
    surrogate_unclipped = vals("inc_ess_surrogate_unclipped_frac")
    surrogate_no_jvp = vals("inc_ess_surrogate_no_jvp_frac")
    finite_true = vals("inc_ess_true_frac")
    used = vals("inc_ess_frac")
    finite_rho0 = vals("inc_ess_true_rho0_frac")
    if not bool((np.isfinite(surrogate) & np.isfinite(finite_true)).any()):
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, surrogate, lw=1.8, label="rank-clipped surrogate inc ESS")
    if np.isfinite(surrogate_unclipped).any():
        axes[0].plot(step, surrogate_unclipped, lw=1.2, ls="--", label="unclipped surrogate inc ESS")
    if np.isfinite(surrogate_no_jvp).any():
        axes[0].plot(step, surrogate_no_jvp, lw=1.2, ls="-.", label="surrogate no-JVP inc ESS")
    axes[0].plot(step, finite_true, lw=1.6, label="finite true inc ESS")
    axes[0].plot(step, used, lw=1.2, ls="--", label="used inc ESS")
    if np.isfinite(finite_rho0).any():
        axes[0].plot(step, finite_rho0, lw=1.1, ls=":", label="finite true inc rho=0")
    resampled = vals("resampled")
    rs = np.isfinite(resampled) & (resampled > 0)
    if bool(rs.any()):
        axes[0].scatter(step[rs], used[rs], s=16, c="black", label="resample")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_title("Incremental ESS: Finite vs Surrogate")
    axes[0].legend(fontsize=8)

    diff = surrogate - finite_true
    axes[1].plot(step, diff, lw=1.6, label="rank-clipped surrogate - finite")
    if np.isfinite(surrogate_unclipped).any():
        axes[1].plot(step, surrogate_unclipped - finite_true, lw=1.2, ls="--", label="unclipped surrogate - finite")
    if np.isfinite(surrogate_no_jvp).any():
        axes[1].plot(step, surrogate_no_jvp - finite_true, lw=1.2, ls="-.", label="no-JVP surrogate - finite")
    axes[1].plot(step, vals("inc_ess_true_gain_vs_rho0"), lw=1.2, label="finite rho gain vs rho=0")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Incremental ESS Gap")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, vals("true_var_rho"), lw=1.5, label="finite true inc var")
    axes[2].plot(step, vals("surrogate_unclipped_var_rho"), lw=1.2, ls="--", label="unclipped surrogate var")
    axes[2].plot(step, vals("surrogate_var_rho"), lw=1.4, label="rank-clipped surrogate var")
    axes[2].plot(step, vals("surrogate_no_jvp_var_rho"), lw=1.2, ls="-.", label="surrogate var with JVP=0")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].set_title("Incremental Variance: True, Surrogate, No-JVP")
    axes[2].legend(fontsize=8)

    mask = np.isfinite(surrogate) & np.isfinite(finite_true)
    axes[3].scatter(finite_true[mask], surrogate[mask], s=12, alpha=0.7, label="rank-clipped")
    mask_unclipped = np.isfinite(surrogate_unclipped) & np.isfinite(finite_true)
    if bool(mask_unclipped.any()):
        axes[3].scatter(
            finite_true[mask_unclipped],
            surrogate_unclipped[mask_unclipped],
            s=10,
            alpha=0.45,
            label="unclipped",
        )
    axes[3].plot([0.0, 1.0], [0.0, 1.0], color="black", lw=0.8, ls=":")
    axes[3].set_xlim(0.0, 1.01)
    axes[3].set_ylim(0.0, 1.01)
    axes[3].set_xlabel("finite true inc ESS / N")
    axes[3].set_ylabel("surrogate inc ESS / N")
    axes[3].set_title("Inc ESS Scatter")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        if ax is not axes[3]:
            ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_inc_ess_agreement.png", bbox_inches="tight")
    plt.close(fig)


def make_gate_temper_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows])

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, vals("gate_temper_alpha"), label="alpha")
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Gate Temper Alpha")
    axes[0].legend()

    axes[1].plot(step, vals("gate_temper_base_ess"), label="base active-cum ESS")
    axes[1].plot(step, vals("gate_temper_active_ess"), label="tempered active-cum ESS")
    axes[1].plot(step, vals("gate_temper_full_ess"), label="full active-cum ESS")
    axes[1].plot(step, vals("gate_temper_carry_ess"), label="carry ESS", ls="--")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("Gate/Carry ESS")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, vals("gate_temper_term_std"), label="gate term std")
    axes[2].plot(step, vals("gate_temper_residual_std"), label="residual std")
    axes[2].plot(step, vals("gate_temper_carry_std"), label="carry std")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].set_title("Residual Accumulation")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, vals("gate_temper_residual_abs_mean"), label="mean |residual|")
    axes[3].plot(step, vals("gate_temper_residual_abs_max"), label="max |residual|")
    axes[3].plot(step, vals("gate_temper_residual_nonzero_frac"), label="nonzero frac")
    axes[3].set_yscale("symlog", linthresh=1e-4)
    axes[3].set_title("Residual Size")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, vals("surrogate_logw_preclip_std"), label="surrogate preclip std")
    axes[4].plot(step, vals("std_logw_surrogate"), label="surrogate final std")
    axes[4].plot(step, vals("surrogate_logw_clip_frac"), label="clip fraction")
    axes[4].set_yscale("symlog", linthresh=1e-4)
    axes[4].set_title("Final Surrogate Clipping")
    axes[4].legend(fontsize=8)

    resampled = vals("resampled")
    axes[5].plot(step, np.cumsum(np.nan_to_num(resampled)), label="cumulative resamples")
    axes[5].plot(step, vals("cum_ess_after_resample"), label="cum ESS after resample")
    axes[5].plot(step, vals("post_step_ess_frac"), label="remaining ESS")
    axes[5].plot(step, vals("post_step_logw_std"), label="remaining logw std")
    axes[5].plot(step, vals("post_step_logw_abs_max"), label="remaining |logw|max")
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].set_title("Resampling and Final Residual")
    axes[5].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_temper_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_jvp_temper_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows])

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    if not np.isfinite(vals("jvp_temper_alpha")).any():
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, vals("jvp_temper_alpha"), label="alpha")
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("JVP Temper Alpha")
    axes[0].legend()

    axes[1].plot(step, vals("jvp_temper_base_ess"), label="base active-cum ESS")
    axes[1].plot(step, vals("jvp_temper_active_ess"), label="tempered active-cum ESS")
    axes[1].plot(step, vals("jvp_temper_full_ess"), label="full active-cum ESS")
    axes[1].plot(step, vals("jvp_temper_carry_ess"), label="carry ESS", ls="--")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("JVP/Carry ESS")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, vals("raw_clip_jvp_term_std"), label="raw JVP term std")
    axes[2].plot(step, vals("raw_clip_jvp_clipped_term_std"), label="raw-clipped JVP std")
    axes[2].plot(step, vals("jvp_temper_term_std"), label="tempered term std")
    axes[2].plot(step, vals("jvp_temper_residual_std"), label="residual std")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].set_title("JVP Term Stabilization")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, vals("jvp_temper_residual_abs_mean"), label="mean |residual|")
    axes[3].plot(step, vals("jvp_temper_residual_abs_max"), label="max |residual|")
    axes[3].plot(step, vals("jvp_temper_residual_nonzero_frac"), label="nonzero frac")
    axes[3].set_yscale("symlog", linthresh=1e-4)
    axes[3].set_title("Residual Size")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, vals("raw_clip_jvp_frac"), label="raw clip fraction")
    axes[4].plot(step, vals("raw_clip_jvp_residual_std"), label="raw clip removed std")
    axes[4].plot(step, vals("jvp_temper_residual_std"), label="tempered residual std")
    axes[4].set_yscale("symlog", linthresh=1e-4)
    axes[4].set_title("Clip vs Temper")
    axes[4].legend(fontsize=8)

    axes[5].plot(step, vals("std_logw_used"), label="used std")
    axes[5].plot(step, vals("std_logw_surrogate"), label="surrogate std")
    axes[5].plot(step, vals("cum_ess_frac"), label="active cum ESS")
    axes[5].plot(step, vals("full_cum_ess_frac"), label="full cum ESS")
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].set_title("Final Active/Full Weight")
    axes[5].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_jvp_temper_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_active_weight_temper_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows])

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)

    if not np.isfinite(vals("active_weight_temper_alpha")).any():
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, vals("active_weight_temper_alpha"), label="alpha")
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Active-Weight Temper Alpha")
    axes[0].legend()

    axes[1].plot(step, vals("active_weight_temper_base_ess"), label="base active-cum ESS")
    axes[1].plot(step, vals("active_weight_temper_active_ess"), label="tempered active-cum ESS")
    axes[1].plot(step, vals("active_weight_temper_full_ess"), label="full active-cum ESS")
    axes[1].plot(step, vals("active_weight_temper_carry_ess"), label="carry ESS", ls="--")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("Active-Weight/Carry ESS")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, vals("active_weight_temper_term_std"), label="full current term std")
    axes[2].plot(step, vals("active_weight_temper_residual_std"), label="residual std")
    axes[2].plot(step, vals("std_logw_used"), label="used std after temper")
    axes[2].plot(step, vals("std_logw_surrogate"), label="full surrogate std", alpha=0.8)
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].set_title("Full Weight Stabilization")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, vals("active_weight_temper_residual_abs_mean"), label="mean |residual|")
    axes[3].plot(step, vals("active_weight_temper_residual_abs_max"), label="max |residual|")
    axes[3].plot(step, vals("active_weight_temper_residual_nonzero_frac"), label="nonzero frac")
    axes[3].set_yscale("symlog", linthresh=1e-4)
    axes[3].set_title("Residual Size")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, vals("raw_clip_jvp_frac"), label="JVP raw clip fraction")
    axes[4].plot(step, vals("raw_clip_jvp_residual_std"), label="JVP raw clip removed std")
    axes[4].plot(step, vals("active_weight_temper_residual_std"), label="full temper residual std")
    axes[4].set_yscale("symlog", linthresh=1e-4)
    axes[4].set_title("Clip vs Full Temper")
    axes[4].legend(fontsize=8)

    axes[5].plot(step, vals("cum_ess_frac"), label="active cum ESS")
    axes[5].plot(step, vals("full_cum_ess_frac"), label="full cum ESS")
    axes[5].plot(step, vals("surrogate_total_ess_frac"), label="surrogate full total ESS", alpha=0.8)
    axes[5].plot(step, vals("surrogate_active_total_ess_frac"), label="surrogate active total ESS", alpha=0.8)
    axes[5].set_ylim(0.0, 1.01)
    axes[5].set_title("Final Active/Full ESS")
    axes[5].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_active_weight_temper_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_beta_variance_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "beta_coord_var_mean" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])
    rs = [r for r in rows if int(r.get("resampled", 0)) == 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)

    def mark_resamples(ax: plt.Axes) -> None:
        for r in rs:
            ax.axvline(r["step"], color="black", lw=0.6, alpha=0.18)

    spread_keys = [
        ("beta_coord_var_mean", "mean coordinate variance"),
        ("std_gl2_per_dim", "std squared norm / d"),
        ("beta_pair_dist_mean_per_sqrt_dim", "mean pair distance / sqrt(d)"),
    ]
    for key, label in spread_keys:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-30), np.nan)
        axes[0, 0].semilogy(step, vals, label=label)
    mark_resamples(axes[0, 0])
    axes[0, 0].set_title("Cross-Particle Beta_l Heterogeneity")
    axes[0, 0].set_xlabel("reverse step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    size_keys = [
        ("mean_gl2_per_dim", "E||T beta||^2/d"),
        ("beta_mean_norm2_per_dim", "||E[T beta]||^2/d"),
    ]
    for key, label in size_keys:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-30), np.nan)
        axes[0, 1].semilogy(step, vals, label=label)
    mark_resamples(axes[0, 1])
    axes[0, 1].set_title("Mean Beta_l Size vs Common Direction")
    axes[0, 1].set_xlabel("reverse step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(step, [r["std_logw_used"] for r in rows], label="used logw std")
    axes[1, 0].plot(step, [r["inc_ess_frac"] for r in rows], label="incESS/N")
    axes[1, 0].plot(step, [r["cum_ess_frac"] for r in rows], label="cumESS/N")
    mark_resamples(axes[1, 0])
    axes[1, 0].set_title("Collapse Diagnostics")
    axes[1, 0].set_xlabel("reverse step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(step, [r["mean_lhat"] for r in rows], label="mean lhat")
    axes[1, 1].plot(step, [r["std_lhat"] for r in rows], label="std lhat")
    axes[1, 1].plot(step, [r["mean_theta"] for r in rows], label="mean theta")
    mark_resamples(axes[1, 1])
    axes[1, 1].set_title("Gate State")
    axes[1, 1].set_xlabel("reverse step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_beta_variance.png", bbox_inches="tight")
    plt.close(fig)


def make_drift_magnitude_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "drift_base_norm_mean_per_sqrt_dim" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])
    rs = [r for r in rows if int(r.get("resampled", 0)) == 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)

    def mark_resamples(ax: plt.Axes) -> None:
        for r in rs:
            ax.axvline(r["step"], color="black", lw=0.6, alpha=0.18)

    norm_keys = [
        ("drift_base_norm_mean_per_sqrt_dim", "base velocity"),
        ("drift_guidance_rho0_norm_mean_per_sqrt_dim", "guidance rho=0"),
        ("drift_guidance_rho_correction_norm_mean_per_sqrt_dim", "rho correction"),
        ("drift_guidance_proposal_norm_mean_per_sqrt_dim", "proposal guidance"),
        ("drift_brownian_norm_mean_per_sqrt_dim", "Brownian"),
        ("drift_deterministic_total_norm_mean_per_sqrt_dim", "deterministic total"),
        ("drift_sample_total_norm_mean_per_sqrt_dim", "sample total"),
    ]
    for key, label in norm_keys:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-30), np.nan)
        axes[0, 0].semilogy(step, vals, label=label)
    mark_resamples(axes[0, 0])
    axes[0, 0].set_title("Proposal Step Component Magnitudes")
    axes[0, 0].set_ylabel("mean ||component|| / sqrt(dim)")
    axes[0, 0].set_xlabel("reverse step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    ratio_keys = [
        ("drift_guidance_proposal_over_base", "proposal guidance / base"),
        ("drift_guidance_proposal_over_brownian", "proposal guidance / Brownian"),
        ("drift_rho_correction_over_guidance_rho0", "rho correction / guidance rho=0"),
        ("drift_deterministic_over_brownian", "deterministic / Brownian"),
    ]
    for key, label in ratio_keys:
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = np.where(np.isfinite(vals), np.maximum(vals, 1e-30), np.nan)
        axes[0, 1].semilogy(step, vals, label=label)
    mark_resamples(axes[0, 1])
    axes[0, 1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0, 1].set_title("Relative Drift Scales")
    axes[0, 1].set_xlabel("reverse step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(step, [r["rho_used"] for r in rows], label="rho ratio")
    axes[1, 0].plot(step, [r["guidance_mean"] for r in rows], label="target guidance coefficient")
    axes[1, 0].plot(step, [r["proposal_guidance_mean"] for r in rows], label="proposal guidance coefficient")
    if "proposal_pred_noise_guidance_mean" in rows[0]:
        axes[1, 0].plot(
            step,
            [r["proposal_pred_noise_guidance_mean"] for r in rows],
            label="proposal eps-equivalent guidance",
        )
    if "proposal_raw_pred_noise_guidance_mean" in rows[0]:
        axes[1, 0].plot(
            step,
            [r["proposal_raw_pred_noise_guidance_mean"] for r in rows],
            label="raw A/B eps-equivalent guidance",
        )
    mark_resamples(axes[1, 0])
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_title("Rho and Guidance Coefficients")
    axes[1, 0].set_xlabel("reverse step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(step, [r["std_logw_used"] for r in rows], label="used logw std")
    axes[1, 1].plot(step, [r["inc_ess_frac"] for r in rows], label="incESS/N")
    axes[1, 1].plot(step, [r["cess_ess_frac"] for r in rows], label="cessESS/N")
    axes[1, 1].plot(step, [r["cum_ess_frac"] for r in rows], label="cumESS/N")
    mark_resamples(axes[1, 1])
    axes[1, 1].set_title("ESS Context")
    axes[1, 1].set_xlabel("reverse step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_drift_magnitudes.png", bbox_inches="tight")
    plt.close(fig)


def make_decomposition_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "decomp_total_std" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)

    std_names = [
        ("group_kernel_alpha", "kernel alpha group"),
        ("group_reverse_kernel_alpha", "reverse kernel group"),
        ("group_forward_kernel_alpha", "forward kernel group"),
        ("group_gate", "gate group"),
        ("group_quadratic", "quadratic group"),
        ("gamma_quadratic", "Gamma quadratic"),
        ("gamma_rank1", "Gamma rank-1"),
        ("proposal_hessian", "proposal Hessian"),
        ("proposal_hessian_jvp", "proposal Hessian JVP"),
        ("gate_curvature", "gate curvature"),
    ]
    axes[0, 0].plot(step, [r["decomp_total_std"] for r in rows], color="black", lw=2, label="surrogate total")
    for name, label in std_names:
        key = f"decomp_{name}_std"
        if key in rows[0]:
            axes[0, 0].plot(step, [r[key] for r in rows], label=label)
    axes[0, 0].set_title("Retained-Order Term Std")
    axes[0, 0].set_xlabel("reverse step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    share_names = [
        ("group_kernel_alpha", "kernel alpha group"),
        ("group_reverse_kernel_alpha", "reverse kernel group"),
        ("group_forward_kernel_alpha", "forward kernel group"),
        ("group_gate", "gate group"),
        ("forward_linear", "forward linear"),
        ("reverse_linear", "reverse linear"),
        ("reverse_quadratic", "reverse quadratic"),
        ("gate_time", "gate time"),
        ("gate_curvature", "gate curvature"),
        ("proposal_hessian", "proposal Hessian"),
        ("proposal_hessian_rank1", "proposal Hessian rank-1"),
        ("proposal_hessian_jvp", "proposal Hessian JVP"),
        ("gamma_rank1", "Gamma rank-1"),
        ("gamma_quadratic", "Gamma quadratic"),
    ]
    for name, label in share_names:
        key = f"decomp_{name}_var_share"
        if key in rows[0]:
            axes[0, 1].plot(step, [r[key] for r in rows], label=label)
    axes[0, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0, 1].set_title("Cov(term, total) / Var(total)")
    axes[0, 1].set_xlabel("reverse step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    fine_names = [
        ("forward_linear", "forward linear"),
        ("forward_quadratic", "forward quadratic"),
        ("reverse_linear", "reverse linear"),
        ("reverse_quadratic", "reverse quadratic"),
        ("gate_time", "gate time"),
        ("gate_curvature", "gate curvature"),
        ("proposal_hessian", "proposal Hessian"),
        ("proposal_hessian_rank1", "proposal Hessian rank-1"),
        ("proposal_hessian_jvp", "proposal Hessian JVP"),
        ("gamma_rank1", "Gamma rank-1"),
    ]
    for name, label in fine_names:
        key = f"decomp_{name}_std"
        if key in rows[0]:
            axes[1, 0].plot(step, [r[key] for r in rows], label=label)
    axes[1, 0].set_title("Fine Term Std")
    axes[1, 0].set_xlabel("reverse step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    top_labels = [str(r.get("decomp_top_abs_cov_term", "none")) for r in rows]
    unique = sorted({x for x in top_labels if x != "none"})
    y_by_label = {label: i for i, label in enumerate(unique)}
    ys = [y_by_label.get(label, np.nan) for label in top_labels]
    axes[1, 1].scatter(step, ys, s=14)
    axes[1, 1].set_yticks(list(y_by_label.values()))
    axes[1, 1].set_yticklabels(list(y_by_label.keys()))
    axes[1, 1].set_title("Largest |Cov(term,total)| Contributor")
    axes[1, 1].set_xlabel("reverse step")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_term_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def make_rho0_decomposition_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "decomp_rho0_total_std" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)

    term_names = [
        ("group_kernel_alpha", "kernel alpha group"),
        ("group_reverse_kernel_alpha", "reverse kernel group"),
        ("group_forward_kernel_alpha", "forward kernel group"),
        ("group_gate", "gate group"),
        ("group_quadratic", "quadratic group"),
        ("proposal_hessian", "proposal Hessian"),
        ("proposal_hessian_jvp", "proposal Hessian JVP"),
        ("gate_time", "gate time"),
        ("gate_curvature", "gate curvature"),
    ]
    axes[0, 0].plot(
        step,
        [r.get("decomp_rho0_total_std", np.nan) for r in rows],
        color="black",
        lw=2,
        label="surrogate rho=0 total",
    )
    for name, label in term_names:
        key = f"decomp_rho0_{name}_std"
        if key in rows[0]:
            axes[0, 0].plot(step, [r.get(key, np.nan) for r in rows], label=label)
    axes[0, 0].set_title("Before-Rho Retained-Order Term Std")
    axes[0, 0].set_xlabel("reverse step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    for name, label in term_names:
        key = f"decomp_rho0_{name}_var_share"
        if key in rows[0]:
            axes[0, 1].plot(step, [r.get(key, np.nan) for r in rows], label=label)
    axes[0, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0, 1].set_title("Before-Rho Cov(term,total) / Var(total)")
    axes[0, 1].set_xlabel("reverse step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(
        step,
        [r.get("finite_rho0_total_std", np.nan) for r in rows],
        color="black",
        lw=2,
        label="finite rho=0 total",
    )
    for name, label in (
        ("theta", "exact gate/theta"),
        ("reverse_kernel", "reverse kernel"),
        ("forward_kernel", "forward kernel"),
    ):
        std_key = f"finite_rho0_{name}_std"
        share_key = f"finite_rho0_{name}_var_share"
        if std_key in rows[0]:
            axes[1, 0].plot(step, [r.get(std_key, np.nan) for r in rows], label=f"{label} std")
        if share_key in rows[0]:
            axes[1, 1].plot(step, [r.get(share_key, np.nan) for r in rows], label=label)
    axes[1, 0].set_title("Before-Rho Exact Finite Term Std")
    axes[1, 0].set_xlabel("reverse step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[1, 1].set_title("Before-Rho Exact Finite Variance Share")
    axes[1, 1].set_xlabel("reverse step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_term_decomposition_rho0.png", bbox_inches="tight")
    plt.close(fig)


def make_jvp_diagnostic_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "jvp_std" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])

    def values(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(16, 15), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, values("decomp_proposal_hessian_jvp_var_share"), label="JVP share")
    axes[0].plot(step, values("decomp_reverse_linear_var_share"), label="reverse linear")
    axes[0].plot(step, values("decomp_reverse_quadratic_var_share"), label="reverse quadratic")
    axes[0].plot(step, values("decomp_group_gate_var_share"), label="gate group")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_title("Signed Variance Shares")
    axes[0].legend(fontsize=8)

    axes[1].plot(step, values("jvp_std"), label="JVP std")
    axes[1].plot(step, values("jvp_trim_top1_std"), label="trim top 1")
    axes[1].plot(step, values("jvp_trim_top4_std"), label="trim top 4")
    axes[1].plot(step, values("std_logw_surrogate"), label="total surrogate std")
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_title("JVP Std After Removing Extremes")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, values("jvp_top1_energy_frac"), label="top 1")
    axes[2].plot(step, values("jvp_top2_energy_frac"), label="top 2")
    axes[2].plot(step, values("jvp_top4_energy_frac"), label="top 4")
    axes[2].plot(step, values("jvp_top8_energy_frac"), label="top 8")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("JVP Variance Energy in Largest Deviations")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, values("jvp_top1_abs_cov_frac"), label="top 1")
    axes[3].plot(step, values("jvp_top2_abs_cov_frac"), label="top 2")
    axes[3].plot(step, values("jvp_top4_abs_cov_frac"), label="top 4")
    axes[3].plot(step, values("jvp_top8_abs_cov_frac"), label="top 8")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_title("JVP |Cov Contribution| in Largest Deviations")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, values("jvp_z_abs_max"), label="max |z-score|")
    axes[4].plot(step, values("jvp_kurtosis"), label="kurtosis")
    axes[4].set_yscale("symlog", linthresh=1e-4)
    axes[4].set_title("JVP Tail Shape")
    axes[4].legend(fontsize=8)

    axes[5].plot(step, values("jvp_core_std"), label="<Delta, Jbeta Delta> std")
    axes[5].plot(step, values("jvp_scale_std"), label="JVP scale std")
    axes[5].plot(step, values("jvp_gamma_delta_norm_std"), label="Delta norm std")
    axes[5].plot(step, values("jvp_jbeta_delta_norm_std"), label="Jbeta Delta norm std")
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].set_title("JVP Ingredients")
    axes[5].legend(fontsize=8)

    axes[6].plot(step, values("jvp_brownian_norm_per_sqrt_dim_abs_corr_jvp_abs"), label="|JVP| vs Brownian norm")
    axes[6].plot(step, values("jvp_gamma_delta_norm_abs_corr_jvp_abs"), label="|JVP| vs Delta norm")
    axes[6].plot(step, values("jvp_jbeta_delta_norm_abs_corr_jvp_abs"), label="|JVP| vs Jbeta Delta norm")
    axes[6].plot(step, values("jvp_core_abs_corr_jvp_abs"), label="|JVP| vs JVP core")
    axes[6].set_ylim(-1.05, 1.05)
    axes[6].set_title("Particlewise Correlations with |JVP|")
    axes[6].legend(fontsize=7)

    axes[7].plot(step, values("rho_used"), label="rho")
    axes[7].plot(step, 1.0 - values("rho_used"), label="1-rho")
    axes[7].plot(step, values("proposal_guidance_mean"), label="proposal guidance")
    axes[7].plot(step, values("cum_ess_frac"), label="active ESS")
    axes[7].axhline(0.0, color="black", lw=0.8)
    axes[7].set_title("Rho / ESS Context")
    axes[7].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_jvp_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_kernel_tail_diagnostic_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "surrogate_reverse_kernel_top4_energy_frac" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])

    def values(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, values("surrogate_reverse_kernel_std"), label="surrogate reverse std")
    axes[0].plot(step, values("surrogate_reverse_kernel_trim_top4_std"), label="surrogate reverse trim top 4")
    axes[0].plot(step, values("finite_reverse_kernel_tail_std"), label="finite reverse std")
    axes[0].plot(step, values("finite_reverse_kernel_tail_trim_top4_std"), label="finite reverse trim top 4")
    axes[0].plot(step, values("jvp_std"), label="JVP std", alpha=0.75)
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_title("Reverse-Kernel Std After Removing Extremes")
    axes[0].legend(fontsize=8)

    axes[1].plot(step, values("surrogate_reverse_kernel_top4_energy_frac"), label="surrogate reverse")
    axes[1].plot(step, values("finite_reverse_kernel_tail_top4_energy_frac"), label="finite reverse")
    axes[1].plot(step, values("surrogate_forward_kernel_top4_energy_frac"), label="surrogate forward")
    axes[1].plot(step, values("jvp_top4_energy_frac"), label="JVP")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Top-4 Centered Energy Fraction")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, values("surrogate_reverse_kernel_trim_top4_std_ratio"), label="surrogate reverse")
    axes[2].plot(step, values("finite_reverse_kernel_tail_trim_top4_std_ratio"), label="finite reverse")
    axes[2].plot(step, values("surrogate_forward_kernel_trim_top4_std_ratio"), label="surrogate forward")
    axes[2].plot(step, values("jvp_trim_top4_std_ratio"), label="JVP")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].set_title("Std Ratio After Removing Top 4")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, values("surrogate_reverse_kernel_jvp_top4_overlap_frac"), label="surrogate reverse vs JVP")
    axes[3].plot(step, values("finite_reverse_kernel_jvp_top4_overlap_frac"), label="finite reverse vs JVP")
    axes[3].plot(step, values("surrogate_forward_kernel_jvp_top4_overlap_frac"), label="surrogate forward vs JVP")
    axes[3].plot(step, values("finite_forward_kernel_jvp_top4_overlap_frac"), label="finite forward vs JVP")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_title("Top-4 Particle Overlap With JVP")
    axes[3].legend(fontsize=7)

    axes[4].plot(step, values("finite_reverse_surrogate_reverse_kernel_top4_overlap_frac"), label="finite vs surrogate reverse")
    axes[4].plot(step, values("finite_forward_surrogate_forward_kernel_top4_overlap_frac"), label="finite vs surrogate forward")
    axes[4].set_ylim(-0.02, 1.02)
    axes[4].set_title("Finite/Surrogate Kernel Tail Overlap")
    axes[4].legend(fontsize=8)

    axes[5].plot(step, values("decomp_group_reverse_kernel_alpha_var_share"), label="surrogate reverse share")
    axes[5].plot(step, values("decomp_group_forward_kernel_alpha_var_share"), label="surrogate forward share")
    axes[5].plot(step, values("finite_reverse_kernel_var_share"), label="finite reverse share")
    axes[5].plot(step, values("finite_forward_kernel_var_share"), label="finite forward share")
    axes[5].plot(step, values("decomp_proposal_hessian_jvp_var_share"), label="JVP share")
    axes[5].axhline(0.0, color="black", lw=0.8)
    axes[5].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[5].set_title("Covariance Share Context")
    axes[5].legend(fontsize=7)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_kernel_tail_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_raw_clip_diagnostic_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "raw_clip_any_frac" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])

    def values(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, values("raw_clip_jvp_frac"), label="JVP")
    axes[0].plot(step, values("raw_clip_reverse_kernel_frac"), label="reverse kernel")
    axes[0].plot(step, values("raw_clip_any_frac"), label="any term", lw=2)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Raw Clip Fraction")
    axes[0].legend(fontsize=8)

    axes[1].plot(step, values("raw_clip_surrogate_orig_std"), label="original surrogate")
    axes[1].plot(step, values("raw_clip_surrogate_clipped_std"), label="raw clipped surrogate")
    axes[1].plot(step, values("std_logw_surrogate"), label="post final clip surrogate", ls="--")
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_title("Surrogate Std Before/After Raw Clip")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, values("raw_clip_surrogate_orig_ess"), label="original")
    axes[2].plot(step, values("raw_clip_surrogate_clipped_ess"), label="raw clipped")
    axes[2].plot(step, values("inc_ess_surrogate_frac"), label="used surrogate", ls="--")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Incremental ESS Before/After Raw Clip")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, values("raw_clip_surrogate_corr"), label="corr(original, clipped)")
    axes[3].axhline(0.0, color="black", lw=0.8)
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_title("Ranking Preservation")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, values("raw_clip_jvp_orig_mass"), label="JVP original mass")
    axes[4].plot(step, values("raw_clip_jvp_clipped_mass"), label="JVP clipped mass")
    axes[4].plot(step, values("raw_clip_reverse_kernel_orig_mass"), label="reverse original mass")
    axes[4].plot(step, values("raw_clip_reverse_kernel_clipped_mass"), label="reverse clipped mass")
    axes[4].set_ylim(-0.02, 1.02)
    axes[4].set_title("Mass on Clipped Particles")
    axes[4].legend(fontsize=7)

    axes[5].plot(step, values("raw_clip_jvp_residual_std"), label="JVP residual std")
    axes[5].plot(step, values("raw_clip_reverse_kernel_residual_std"), label="reverse residual std")
    axes[5].plot(step, values("raw_clip_total_residual_std"), label="total residual std", lw=2)
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].set_title("Removed Log-Weight Residual")
    axes[5].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_raw_clip_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_surrogate_agreement_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "surrogate_true_bad_top1_affine_resid_energy_frac" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])

    def values(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(16, 15), dpi=140)
    axes = axes.ravel()

    axes[0].plot(step, values("corr_surrogate_true"), label="corr(surrogate,true)")
    axes[0].plot(step, values("surrogate_true_trim_bad_top1_corr"), label="trim bad top 1")
    axes[0].plot(step, values("surrogate_true_trim_bad_top4_corr"), label="trim bad top 4")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title("Surrogate/Finiteness Correlation")
    axes[0].legend(fontsize=8)

    axes[1].plot(step, values("surrogate_true_affine_resid_rel_std"), label="affine residual / true std")
    axes[1].plot(step, values("surrogate_true_trim_bad_top1_affine_resid_rel_std"), label="trim bad top 1")
    axes[1].plot(step, values("surrogate_true_trim_bad_top4_affine_resid_rel_std"), label="trim bad top 4")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_title("Agreement Residual After Scale/Shift")
    axes[1].legend(fontsize=8)

    axes[2].plot(step, values("surrogate_true_bad_top1_affine_resid_energy_frac"), label="top 1")
    axes[2].plot(step, values("surrogate_true_bad_top2_affine_resid_energy_frac"), label="top 2")
    axes[2].plot(step, values("surrogate_true_bad_top4_affine_resid_energy_frac"), label="top 4")
    axes[2].plot(step, values("surrogate_true_bad_top8_affine_resid_energy_frac"), label="top 8")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Disagreement Energy in Worst Particles")
    axes[2].legend(fontsize=8)

    axes[3].plot(step, values("surrogate_true_bad_top4_approx_var_frac"), label="surrogate variance frac")
    axes[3].plot(step, values("surrogate_true_bad_top4_target_var_frac"), label="true variance frac")
    axes[3].plot(step, values("surrogate_true_bad_top4_abs_cross_frac"), label="|cross| frac")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_title("Worst 4 Disagreement Points: Variance Role")
    axes[3].legend(fontsize=8)

    axes[4].plot(step, values("surrogate_true_bad_top4_approx_weight_mass_frac"), label="surrogate mass")
    axes[4].plot(step, values("surrogate_true_bad_top4_target_weight_mass_frac"), label="true mass")
    axes[4].set_ylim(-0.02, 1.02)
    axes[4].set_title("Worst 4 Disagreement Points: Weight Mass")
    axes[4].legend(fontsize=8)

    axes[5].plot(step, values("std_logw_surrogate"), label="surrogate std")
    axes[5].plot(step, values("std_logw_true"), label="finite true std")
    axes[5].plot(step, values("surrogate_true_affine_resid_std"), label="affine residual std")
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].set_title("Std Scale")
    axes[5].legend(fontsize=8)

    axes[6].plot(step, values("math_forward_alpha_hessian_vs_forward_kernel_corr"), label="forward alpha+Hessian corr")
    axes[6].plot(step, values("math_forward_alpha_hessian_vs_forward_kernel_affine_resid_rel_std"), label="forward rel residual")
    axes[6].plot(step, values("math_total_forward_alpha_hessian_vs_true_corr"), label="total with forward surrogate corr")
    axes[6].axhline(0.0, color="black", lw=0.8)
    axes[6].set_title("Forward-Kernel Surrogate Check")
    axes[6].legend(fontsize=7)

    axes[7].plot(step, values("decomp_proposal_hessian_jvp_var_share"), label="JVP variance share")
    axes[7].plot(step, values("drift_guidance_proposal_over_brownian"), label="guidance / Brownian")
    axes[7].plot(step, values("rho_used"), label="rho")
    axes[7].plot(step, values("true_active_total_ess_frac"), label="true active ESS")
    axes[7].axhline(0.0, color="black", lw=0.8)
    axes[7].set_title("Context")
    axes[7].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("reverse step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_surrogate_agreement.png", bbox_inches="tight")
    plt.close(fig)


def make_math_diagnostic_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    if not rows or "math_total_forward_alpha_hessian_vs_true_resid_std" not in rows[0]:
        return
    step = np.array([r["step"] for r in rows])

    def values(key: str) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(16, 16), dpi=140)
    axes = axes.ravel()

    total_stages = [
        ("surrogate_true_resid_std", "surrogate"),
        ("math_total_gate_taylor2_exact_s_vs_true_resid_std", "gate Taylor, exact S"),
        ("math_total_gate_taylor2_local_s_vs_true_resid_std", "gate Taylor, local S"),
        ("math_total_gate_retained_delta_vs_true_resid_std", "gate retained, full delta"),
        ("math_total_gate_retained_brownian_vs_true_resid_std", "gate retained, Brownian"),
        ("math_total_forward_alpha_vs_true_resid_std", "forward alpha"),
        ("math_total_forward_alpha_hessian_vs_true_resid_std", "forward alpha+Hessian"),
    ]
    for key, label in total_stages:
        axes[0].plot(step, values(key), label=label)
    axes[0].set_title("Total Residual Std vs Finite True")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].legend(fontsize=8)

    corr_stages = [
        ("corr_surrogate_true", "surrogate"),
        ("math_total_gate_taylor2_exact_s_vs_true_corr", "gate Taylor, exact S"),
        ("math_total_gate_taylor2_local_s_vs_true_corr", "gate Taylor, local S"),
        ("math_total_forward_alpha_hessian_vs_true_corr", "forward alpha+Hessian"),
    ]
    for key, label in corr_stages:
        axes[1].plot(step, values(key), label=label)
    axes[1].set_title("Total Corr with Finite True")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(fontsize=8)

    ess_stages = [
        ("inc_ess_true_frac", "finite true"),
        ("inc_ess_surrogate_frac", "surrogate"),
        ("math_total_gate_taylor2_exact_s_vs_true_ess_frac", "gate Taylor, exact S"),
        ("math_total_forward_alpha_hessian_vs_true_ess_frac", "forward alpha+Hessian"),
    ]
    for key, label in ess_stages:
        axes[2].plot(step, values(key), label=label)
    axes[2].set_title("Incremental ESS by Staged Log Weight")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=8)

    finite_shares = [
        ("finite_theta_var_share", "exact gate"),
        ("finite_reverse_kernel_var_share", "exact reverse kernel"),
        ("finite_forward_kernel_var_share", "exact forward kernel"),
    ]
    for key, label in finite_shares:
        axes[3].plot(step, values(key), label=label)
    axes[3].axhline(0.0, color="black", lw=0.8)
    axes[3].set_title("Finite Exact Covariance Shares")
    axes[3].legend(fontsize=8)

    gate_terms = [
        ("math_gate_taylor2_exact_s_vs_theta_resid_std", "Taylor using exact S"),
        ("math_gate_taylor2_local_s_vs_theta_resid_std", "Taylor using local S"),
        ("math_gate_retained_delta_vs_theta_resid_std", "retained full delta"),
        ("math_gate_retained_brownian_vs_theta_resid_std", "retained Brownian"),
    ]
    for key, label in gate_terms:
        axes[4].plot(step, values(key), label=label)
    axes[4].set_title("Gate Term Residual Std")
    axes[4].set_yscale("symlog", linthresh=1e-4)
    axes[4].legend(fontsize=8)

    forward_terms = [
        ("math_forward_alpha_vs_forward_kernel_resid_std", "forward alpha"),
        ("math_forward_alpha_hessian_vs_forward_kernel_resid_std", "forward alpha+Hessian"),
    ]
    for key, label in forward_terms:
        axes[5].plot(step, values(key), label=label)
    axes[5].set_title("Forward-Kernel Residual Std")
    axes[5].set_yscale("symlog", linthresh=1e-4)
    axes[5].legend(fontsize=8)

    axes[6].plot(step, values("math_s_local_error_std"), label="S local - S exact")
    axes[6].plot(step, values("math_s_local_vs_s_exact_resid_std"), label="S comparison resid")
    axes[6].set_title("S Expansion Error")
    axes[6].set_yscale("symlog", linthresh=1e-4)
    axes[6].legend(fontsize=8)

    axes[7].plot(step, values("math_linear_gate_reverse_delta_std"), label="gate+reverse linear, full delta")
    axes[7].plot(step, values("math_linear_gate_reverse_brownian_std"), label="gate+reverse linear, Brownian")
    axes[7].set_title("Uncanceled Linear Diagnostics")
    axes[7].set_yscale("symlog", linthresh=1e-4)
    axes[7].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "smc_pa_gate_math_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def make_row(
    *,
    args: argparse.Namespace,
    step: int,
    timestep: int,
    c: float,
    eta: float,
    gate_power: float,
    gate_power_new: float,
    linear_baseline_kappa: float,
    linear_baseline_kappa_new: float,
    rho: float,
    theta: torch.Tensor,
    lhat: torch.Tensor,
    lhat_new: torch.Tensor,
    beta_l: torch.Tensor,
    logw_used: torch.Tensor,
    logw_true: torch.Tensor,
    logw_true_active: torch.Tensor,
    logw_surrogate_raw: torch.Tensor,
    logw_surrogate: torch.Tensor,
    logw_surrogate_unclipped: torch.Tensor | None,
    logw_active_total: torch.Tensor,
    logw_total_new: torch.Tensor,
    logw_true_rho0: torch.Tensor,
    logw_true_active_rho0: torch.Tensor,
    logw_surrogate_rho0: torch.Tensor,
    logw_total_prev: torch.Tensor,
    surrogate_decomp: dict[str, torch.Tensor] | None,
    surrogate_decomp_rho0: dict[str, torch.Tensor] | None,
    finite_components: dict[str, torch.Tensor] | None,
    finite_components_rho0: dict[str, torch.Tensor] | None,
    math_components: dict[str, torch.Tensor] | None,
    theta_term_exact: torch.Tensor,
    kernel_lr: torch.Tensor,
    drift_components: dict[str, torch.Tensor],
    opt: dict[str, float],
    do_resample: bool,
    cess_ess: float,
    resample_ess_used: float,
    resample_weight_ess: float,
    resample_allowed: bool,
    skipped_weight: bool,
    ancestor_idx: torch.Tensor | None = None,
    logw_total_after_resample: torch.Tensor | None = None,
    gate_temper_base: torch.Tensor | None = None,
    gate_temper_term: torch.Tensor | None = None,
    gate_temper_residual: torch.Tensor | None = None,
    gate_temper_carry: torch.Tensor | None = None,
    jvp_temper_base: torch.Tensor | None = None,
    jvp_temper_term: torch.Tensor | None = None,
    jvp_temper_residual: torch.Tensor | None = None,
    jvp_temper_carry: torch.Tensor | None = None,
    active_weight_temper_base: torch.Tensor | None = None,
    active_weight_temper_term: torch.Tensor | None = None,
    active_weight_temper_residual: torch.Tensor | None = None,
    active_weight_temper_carry: torch.Tensor | None = None,
    origin_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    theta_f = theta.float()
    beta_flat = beta_l.float().flatten(1)
    dim = max(int(beta_flat.shape[1]), 1)
    true_var = weighted_variance_np(logw_true)
    true_active_var = weighted_variance_np(logw_true_active)
    true0_var = weighted_variance_np(logw_true_rho0)
    true_active0_var = weighted_variance_np(logw_true_active_rho0)
    total_prev_f = logw_total_prev.float()
    active_total_f = logw_active_total.float()
    cum_true = total_prev_f + logw_true.float()
    cum_true_active = total_prev_f + logw_true_active.float()
    cum_true0 = total_prev_f + logw_true_rho0.float()
    cum_true_active0 = total_prev_f + logw_true_active_rho0.float()
    surrogate_total = total_prev_f + logw_surrogate.float()
    surrogate_total0 = total_prev_f + logw_surrogate_rho0.float()
    cum_true_var = weighted_variance_np(cum_true)
    cum_true_active_var = weighted_variance_np(cum_true_active)
    cum_true0_var = weighted_variance_np(cum_true0)
    cum_true_active0_var = weighted_variance_np(cum_true_active0)
    surrogate_total_var = weighted_variance_np(surrogate_total)
    surrogate_total0_var = weighted_variance_np(surrogate_total0)
    surr_var = weighted_variance_np(logw_surrogate)
    used_var = weighted_variance_np(logw_used)
    surrogate_unclipped_f = (
        logw_surrogate_unclipped.float()
        if logw_surrogate_unclipped is not None
        else logw_surrogate_raw.float()
    )
    jvp_component = None
    if surrogate_decomp is not None:
        jvp_component = surrogate_decomp.get("proposal_hessian_jvp")
    if jvp_component is not None and bool(torch.isfinite(jvp_component.float()).any().item()):
        jvp_component_f = torch.nan_to_num(jvp_component.float(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        jvp_component_f = torch.zeros_like(surrogate_unclipped_f)
    surrogate_no_jvp_f = surrogate_unclipped_f - jvp_component_f
    counterfactual_components: dict[str, torch.Tensor] = {}
    for label, key in (
        ("gate", "group_gate"),
        ("reverse", "group_reverse_kernel_alpha"),
        ("forward", "group_forward_kernel_alpha"),
    ):
        component = None if surrogate_decomp is None else surrogate_decomp.get(key)
        if component is not None and bool(
            torch.isfinite(component.float()).any().item()
        ):
            component_f = torch.nan_to_num(
                component.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            component_f = torch.zeros_like(surrogate_unclipped_f)
        counterfactual_components[label] = surrogate_unclipped_f - component_f
    raw_jvp_component = (
        None
        if surrogate_decomp is None
        else surrogate_decomp.get("proposal_hessian_jvp_raw")
    )
    raw_jvp_component_f = torch.zeros_like(surrogate_unclipped_f)
    alpha_eff = float(
        getattr(args, "jvp_shrink_alpha_effective", 1.0)
    )
    if raw_jvp_component is not None and bool(
        torch.isfinite(raw_jvp_component.float()).any().item()
    ):
        raw_jvp_component_f = torch.nan_to_num(
            raw_jvp_component.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    # Diagnostic only: undo the fixed JVP shrink while retaining the same
    # exact-gate/reverse hybrid and all other retained-order terms.  This is
    # the pre-JVP-tempering asymptotic log-weight requested by the finite
    # fact-check; it never affects rho selection, ESS, or resampling.
    surrogate_pre_jvp_shrink_f = (
        surrogate_unclipped_f
        + (1.0 - alpha_eff) * raw_jvp_component_f
    )
    jvp_fit_cov = float("nan")
    jvp_fit_var = float("nan")
    jvp_fit_alpha = float("nan")
    if raw_jvp_component is not None:
        raw_jvp_f = raw_jvp_component.float()
        fit_mask = (
            torch.isfinite(raw_jvp_f)
            & torch.isfinite(logw_true.float())
            & torch.isfinite(surrogate_unclipped_f)
        )
        if int(fit_mask.sum().item()) >= 4:
            x_fit = raw_jvp_f[fit_mask]
            # Remove the JVP approximation currently in the surrogate, then
            # regress the finite diagnostic's missing centered contribution on
            # the raw fixed-noise JVP.  This is an optional calibration
            # diagnostic; it never changes alpha during sampling.
            baseline_fit = (
                surrogate_unclipped_f[fit_mask] - alpha_eff * x_fit
            )
            y_fit = logw_true.float()[fit_mask] - baseline_fit
            xc_fit = x_fit - x_fit.mean()
            yc_fit = y_fit - y_fit.mean()
            var_fit_t = torch.mean(xc_fit * xc_fit)
            cov_fit_t = torch.mean(xc_fit * yc_fit)
            jvp_fit_var = float(var_fit_t.item())
            jvp_fit_cov = float(cov_fit_t.item())
            if jvp_fit_var > 1e-20:
                jvp_fit_alpha = float(
                    (cov_fit_t / var_fit_t).clamp(0.0, 1.0).item()
                )
    surr_unclipped_var = weighted_variance_np(surrogate_unclipped_f)
    surr_no_jvp_var = weighted_variance_np(surrogate_no_jvp_f)
    ess_true = global_ess_frac_from_logw(logw_true)
    ess_true_active = global_ess_frac_from_logw(logw_true_active)
    ess_true0 = global_ess_frac_from_logw(logw_true_rho0)
    ess_true_active0 = global_ess_frac_from_logw(logw_true_active_rho0)
    cum_ess_true = global_ess_frac_from_logw(cum_true)
    cum_ess_true_active = global_ess_frac_from_logw(cum_true_active)
    cum_ess_true0 = global_ess_frac_from_logw(cum_true0)
    cum_ess_true_active0 = global_ess_frac_from_logw(cum_true_active0)
    cess_true = conditional_ess_frac_from_logw(total_prev_f, logw_true.float())
    cess_true_active = conditional_ess_frac_from_logw(total_prev_f, logw_true_active.float())
    cess_true0 = conditional_ess_frac_from_logw(total_prev_f, logw_true_rho0.float())
    cess_true_active0 = conditional_ess_frac_from_logw(total_prev_f, logw_true_active_rho0.float())
    gate_base_f = gate_temper_base.float() if gate_temper_base is not None else torch.full_like(logw_used.float(), float("nan"))
    gate_term_f = gate_temper_term.float() if gate_temper_term is not None else torch.full_like(logw_used.float(), float("nan"))
    gate_resid_f = gate_temper_residual.float() if gate_temper_residual is not None else torch.zeros_like(logw_used.float())
    gate_carry_f = gate_temper_carry.float() if gate_temper_carry is not None else torch.zeros_like(logw_used.float())
    gate_resid_nonzero_frac = float((gate_resid_f.abs() > 1e-12).float().mean().item()) if gate_resid_f.numel() else float("nan")
    jvp_base_f = jvp_temper_base.float() if jvp_temper_base is not None else torch.full_like(logw_used.float(), float("nan"))
    jvp_term_f = jvp_temper_term.float() if jvp_temper_term is not None else torch.full_like(logw_used.float(), float("nan"))
    jvp_resid_f = jvp_temper_residual.float() if jvp_temper_residual is not None else torch.zeros_like(logw_used.float())
    jvp_carry_f = jvp_temper_carry.float() if jvp_temper_carry is not None else torch.zeros_like(logw_used.float())
    jvp_resid_nonzero_frac = float((jvp_resid_f.abs() > 1e-12).float().mean().item()) if jvp_resid_f.numel() else float("nan")
    active_weight_base_f = active_weight_temper_base.float() if active_weight_temper_base is not None else torch.full_like(logw_used.float(), float("nan"))
    active_weight_term_f = active_weight_temper_term.float() if active_weight_temper_term is not None else torch.full_like(logw_used.float(), float("nan"))
    active_weight_resid_f = active_weight_temper_residual.float() if active_weight_temper_residual is not None else torch.zeros_like(logw_used.float())
    active_weight_carry_f = active_weight_temper_carry.float() if active_weight_temper_carry is not None else torch.zeros_like(logw_used.float())
    active_weight_resid_nonzero_frac = (
        float((active_weight_resid_f.abs() > 1e-12).float().mean().item())
        if active_weight_resid_f.numel()
        else float("nan")
    )
    surrogate_raw_f = logw_surrogate_raw.float()
    surrogate_clip = float(getattr(args, "surrogate_logw_clip", 0.0))
    surrogate_clip_frac_global = (
        float((torch.isfinite(surrogate_raw_f) & (surrogate_raw_f.abs() > surrogate_clip)).float().mean().item())
        if surrogate_clip > 0.0 and surrogate_raw_f.numel()
        else 0.0
    )
    if do_resample and logw_total_after_resample is not None:
        post_step_logw = logw_total_after_resample.float()
        cum_ess_after_resample = global_ess_frac_from_logw(logw_total_after_resample)
    elif do_resample:
        post_step_logw = torch.zeros_like(logw_total_new.float())
        cum_ess_after_resample = 1.0
    else:
        post_step_logw = logw_total_new.float()
        cum_ess_after_resample = global_ess_frac_from_logw(logw_total_new)

    cfg_scale = float(getattr(args, "cfg_scale", 1.0))
    base_lambda_neg = float(getattr(args, "base_lambda_neg", 0.0))
    gate_ratio_mode = str(getattr(args, "gate_ratio_mode", "cfg-real"))
    raw_ab_beta_scale = cfg_scale if gate_ratio_mode == "cfg-real" else 1.0
    pred_noise_factor = float(opt.get("pred_noise_guidance_factor", float("nan")))
    raw_target_guidance = (
        float(gate_power)
        * theta_f
        * float(eta)
        * float(args.lhat_temp)
    )
    baseline_kappa = float(linear_baseline_kappa)
    baseline_kappa_new = float(linear_baseline_kappa_new)
    pa_residual_alpha = float(getattr(args, "pa_residual_alpha", 1.0))
    residual_guidance = pa_residual_alpha * (
        raw_target_guidance - baseline_kappa
    )
    target_guidance = baseline_kappa + residual_guidance
    proposal_residual_guidance = residual_guidance * (1.0 - float(rho))
    target_guidance_coeff = float(target_guidance.mean().item())
    residual_guidance_coeff = float(residual_guidance.mean().item())
    proposal_residual_guidance_coeff = float(
        proposal_residual_guidance.mean().item()
    )
    proposal_guidance_coeff = (
        baseline_kappa + proposal_residual_guidance_coeff
    )

    row: dict[str, Any] = {
        "step": step,
        "timestep": timestep,
        "c_t": c,
        "eta_t": eta,
        "lhat_temp": float(args.lhat_temp),
        "gate_power": float(gate_power),
        "gate_power_new": float(gate_power_new),
        "gate_power_terminal": float(args.gate_power),
        "gate_power_schedule": str(
            getattr(args, "gate_power_schedule", "constant")
        ),
        "gate_power_start_frac": float(
            getattr(args, "gate_power_start_frac", 1.0)
        ),
        "gate_power_end_frac": float(
            getattr(args, "gate_power_end_frac", 1.0)
        ),
        "gate_power_schedule_gamma": float(
            getattr(args, "gate_power_schedule_gamma", 1.0)
        ),
        "theta_mid_logit_shift": float(getattr(args, "theta_mid_logit_shift", 0.0)),
        "cfg_scale": cfg_scale,
        "base_lambda_neg": base_lambda_neg,
        "linear_baseline_kappa": baseline_kappa,
        "linear_baseline_kappa_new": baseline_kappa_new,
        "pa_residual_alpha": pa_residual_alpha,
        "kappa_delta": baseline_kappa_new - baseline_kappa,
        "kappa_start": float(args.kappa_start),
        "kappa_end": float(args.kappa_end),
        "kappa_schedule": str(args.kappa_schedule),
        "kappa_end_frac": float(args.kappa_end_frac),
        "kappa_gamma": float(args.kappa_gamma),
        "gate_ratio_mode": gate_ratio_mode,
        "raw_ab_beta_scale": raw_ab_beta_scale,
        "surrogate_gamma_delta": str(getattr(args, "surrogate_gamma_delta", "brownian")),
        "surrogate_poly_degree": opt.get("surrogate_poly_degree", float("nan")),
        "rho_used": float(rho),
        "rho_objective_start": opt.get("objective_start", float("nan")),
        "rho_objective_final": opt.get("objective_final", float("nan")),
        "rho_objective_rho0": opt.get("objective_rho0", float("nan")),
        "rho_total_ess_start": opt.get("rho_total_ess_start", float("nan")),
        "rho_total_ess_final": opt.get("rho_total_ess_final", float("nan")),
        "rho_total_ess_rho0": opt.get("rho_total_ess_rho0", float("nan")),
        "rho_inc_ess_start": opt.get("rho_inc_ess_start", float("nan")),
        "rho_inc_ess_final": opt.get("rho_inc_ess_final", float("nan")),
        "rho_inc_ess_rho0": opt.get("rho_inc_ess_rho0", float("nan")),
        "rho_grad_start": opt.get("grad_start", float("nan")),
        "rho_grad_final": opt.get("grad_final", float("nan")),
        "rho_grad_rho0": opt.get("grad_rho0", float("nan")),
        "rho_accepted": opt.get("accepted", float("nan")),
        "rho_selected_rho0": opt.get("selected_rho0", float("nan")),
        "rho_steps_accepted": opt.get("accepted_steps", float("nan")),
        "rho_backtracks": opt.get("backtracks", float("nan")),
        "rho_lr_last": opt.get("lr_last", float("nan")),
        "rho_line_search_failed": opt.get("line_search_failed", float("nan")),
        "rho_exact_candidates": opt.get("exact_candidates", float("nan")),
        "rho_search_candidates": opt.get(
            "search_candidates",
            opt.get("exact_candidates", float("nan")),
        ),
        "rho_optimizer_effective": opt.get(
            "rho_optimizer_effective",
            str(getattr(args, "rho_optimizer", "unknown")),
        ),
        "rho_optimizer_iterations": opt.get(
            "rho_optimizer_iterations", float("nan")
        ),
        "rho_objective_evaluations": opt.get(
            "rho_objective_evaluations", float("nan")
        ),
        "rho_exact_stationary": opt.get("exact_stationary", float("nan")),
        "rho_over1_penalty": opt.get("over1_penalty", float(args.rho_over1_penalty)),
        "rho_objective": str(getattr(args, "rho_objective", "total_logw")),
        "rho_ess_search_mode": str(getattr(args, "rho_ess_search_mode", "exact")),
        "rho_ess_coarse_candidates": int(getattr(args, "rho_ess_coarse_candidates", 11)),
        "hybrid_exact_gate_reverse": int(
            bool(getattr(args, "hybrid_exact_gate_reverse", False))
        ),
        "rho_ess_min_gain": float(
            getattr(args, "rho_ess_min_gain", 0.0)
        ),
        "raw_clip_jvp_in_rho_objective": opt.get("raw_clip_jvp_in_rho_objective", 0.0),
        "gate_temper_enabled": int(bool(getattr(args, "adaptive_gate_temper", False))),
        "gate_temper_ess_threshold": float(getattr(args, "gate_temper_ess", float("nan"))),
        "gate_temper_alpha": opt.get("gate_temper_alpha", float("nan")),
        "gate_temper_base_ess": opt.get("gate_temper_base_ess", float("nan")),
        "gate_temper_active_ess": opt.get("gate_temper_active_ess", float("nan")),
        "gate_temper_full_ess": opt.get("gate_temper_full_ess", float("nan")),
        "gate_temper_base_std": float(gate_base_f.std(unbiased=False).item()),
        "gate_temper_term_std": float(gate_term_f.std(unbiased=False).item()),
        "gate_temper_residual_std": float(gate_resid_f.std(unbiased=False).item()),
        "gate_temper_residual_abs_mean": float(gate_resid_f.abs().mean().item()),
        "gate_temper_residual_abs_max": float(gate_resid_f.abs().max().item()),
        "gate_temper_residual_nonzero_frac": gate_resid_nonzero_frac,
        "gate_temper_residual_ess": global_ess_frac_from_logw(gate_resid_f),
        "gate_temper_carry_std": float(gate_carry_f.std(unbiased=False).item()),
        "gate_temper_carry_ess": global_ess_frac_from_logw(gate_carry_f),
        "jvp_temper_enabled": int(bool(getattr(args, "adaptive_jvp_temper", False))),
        "jvp_temper_ess_threshold": float(getattr(args, "jvp_temper_ess", float("nan"))),
        "jvp_temper_alpha": opt.get("jvp_temper_alpha", float("nan")),
        "jvp_temper_base_ess": opt.get("jvp_temper_base_ess", float("nan")),
        "jvp_temper_active_ess": opt.get("jvp_temper_active_ess", float("nan")),
        "jvp_temper_full_ess": opt.get("jvp_temper_full_ess", float("nan")),
        "jvp_temper_base_std": float(jvp_base_f.std(unbiased=False).item()),
        "jvp_temper_term_std": float(jvp_term_f.std(unbiased=False).item()),
        "jvp_temper_residual_std": float(jvp_resid_f.std(unbiased=False).item()),
        "jvp_temper_residual_abs_mean": float(jvp_resid_f.abs().mean().item()),
        "jvp_temper_residual_abs_max": float(jvp_resid_f.abs().max().item()),
        "jvp_temper_residual_nonzero_frac": jvp_resid_nonzero_frac,
        "jvp_temper_residual_ess": global_ess_frac_from_logw(jvp_resid_f),
        "jvp_temper_carry_std": float(jvp_carry_f.std(unbiased=False).item()),
        "jvp_temper_carry_ess": global_ess_frac_from_logw(jvp_carry_f),
        "carry_weight_temper_enabled": int(bool(getattr(args, "adaptive_carry_weight_temper", False))),
        "carry_weight_temper_ess_threshold": float(getattr(args, "carry_weight_temper_ess", float("nan"))),
        "carry_weight_temper_alpha": opt.get("carry_weight_temper_alpha", float("nan")),
        "carry_weight_temper_base_ess": opt.get("carry_weight_temper_base_ess", float("nan")),
        "carry_weight_temper_active_ess": opt.get("carry_weight_temper_active_ess", float("nan")),
        "carry_weight_temper_full_ess": opt.get("carry_weight_temper_full_ess", float("nan")),
        "carry_weight_temper_term_std": opt.get("carry_weight_temper_term_std", float("nan")),
        "carry_weight_temper_residual_std": opt.get("carry_weight_temper_residual_std", float("nan")),
        "active_weight_temper_enabled": int(bool(getattr(args, "adaptive_active_weight_temper", False))),
        "active_weight_temper_ess_threshold": float(getattr(args, "active_weight_temper_ess", float("nan"))),
        "active_weight_temper_alpha": opt.get("active_weight_temper_alpha", float("nan")),
        "active_weight_temper_base_ess": opt.get("active_weight_temper_base_ess", float("nan")),
        "active_weight_temper_active_ess": opt.get("active_weight_temper_active_ess", float("nan")),
        "active_weight_temper_full_ess": opt.get("active_weight_temper_full_ess", float("nan")),
        "active_weight_temper_base_std": float(active_weight_base_f.std(unbiased=False).item()),
        "active_weight_temper_term_std": float(active_weight_term_f.std(unbiased=False).item()),
        "active_weight_temper_residual_std": float(active_weight_resid_f.std(unbiased=False).item()),
        "active_weight_temper_residual_abs_mean": float(active_weight_resid_f.abs().mean().item()),
        "active_weight_temper_residual_abs_max": float(active_weight_resid_f.abs().max().item()),
        "active_weight_temper_residual_nonzero_frac": active_weight_resid_nonzero_frac,
        "active_weight_temper_residual_ess": global_ess_frac_from_logw(active_weight_resid_f),
        "active_weight_temper_carry_std": float(active_weight_carry_f.std(unbiased=False).item()),
        "active_weight_temper_carry_ess": global_ess_frac_from_logw(active_weight_carry_f),
        "surrogate_logw_clip": surrogate_clip,
        "surrogate_logw_clip_frac": surrogate_clip_frac_global,
        "surrogate_logw_preclip_std": float(surrogate_raw_f.std(unbiased=False).item()),
        "raw_clip_jvp_threshold": opt.get("raw_clip_jvp_threshold", float(getattr(args, "raw_clip_jvp", 0.0))),
        "rank_clip_jvp_topk": opt.get("rank_clip_jvp_topk", float(getattr(args, "rank_clip_jvp_topk", 0))),
        "adaptive_rank_clip_jvp": opt.get(
            "adaptive_rank_clip_jvp",
            float(bool(getattr(args, "adaptive_rank_clip_jvp", False))),
        ),
        "rank_clip_jvp_effective_topk": opt.get(
            "rank_clip_jvp_effective_topk",
            float(getattr(args, "rank_clip_jvp_topk", 0)),
        ),
        "raw_clip_count_eps": opt.get("raw_clip_count_eps", float(getattr(args, "raw_clip_count_eps", 1e-6))),
        "raw_clip_reverse_kernel_threshold": opt.get(
            "raw_clip_reverse_kernel_threshold",
            float(getattr(args, "raw_clip_reverse_kernel", 0.0)),
        ),
        "raw_clip_jvp_enabled": opt.get("raw_clip_jvp_enabled", 0.0),
        "raw_clip_jvp_frac": opt.get("raw_clip_jvp_frac", 0.0),
        "raw_clip_jvp_residual_std": opt.get("raw_clip_jvp_residual_std", 0.0),
        "raw_clip_jvp_residual_abs_mean": opt.get("raw_clip_jvp_residual_abs_mean", 0.0),
        "raw_clip_jvp_residual_abs_max": opt.get("raw_clip_jvp_residual_abs_max", 0.0),
        "raw_clip_jvp_term_std": opt.get("raw_clip_jvp_term_std", float("nan")),
        "raw_clip_jvp_clipped_term_std": opt.get("raw_clip_jvp_clipped_term_std", float("nan")),
        "raw_clip_jvp_center": opt.get("raw_clip_jvp_center", float("nan")),
        "raw_clip_jvp_effective_threshold": opt.get("raw_clip_jvp_effective_threshold", float("nan")),
        "raw_clip_jvp_rank_topk": opt.get("raw_clip_jvp_rank_topk", float(getattr(args, "rank_clip_jvp_topk", 0))),
        "raw_clip_jvp_orig_mass": opt.get("raw_clip_jvp_orig_mass", 0.0),
        "raw_clip_jvp_clipped_mass": opt.get("raw_clip_jvp_clipped_mass", 0.0),
        "raw_clip_reverse_kernel_enabled": opt.get("raw_clip_reverse_kernel_enabled", 0.0),
        "raw_clip_reverse_kernel_frac": opt.get("raw_clip_reverse_kernel_frac", 0.0),
        "raw_clip_reverse_kernel_residual_std": opt.get("raw_clip_reverse_kernel_residual_std", 0.0),
        "raw_clip_reverse_kernel_residual_abs_mean": opt.get("raw_clip_reverse_kernel_residual_abs_mean", 0.0),
        "raw_clip_reverse_kernel_residual_abs_max": opt.get("raw_clip_reverse_kernel_residual_abs_max", 0.0),
        "raw_clip_reverse_kernel_term_std": opt.get("raw_clip_reverse_kernel_term_std", float("nan")),
        "raw_clip_reverse_kernel_clipped_term_std": opt.get(
            "raw_clip_reverse_kernel_clipped_term_std",
            float("nan"),
        ),
        "raw_clip_reverse_kernel_center": opt.get("raw_clip_reverse_kernel_center", float("nan")),
        "raw_clip_reverse_kernel_orig_mass": opt.get("raw_clip_reverse_kernel_orig_mass", 0.0),
        "raw_clip_reverse_kernel_clipped_mass": opt.get("raw_clip_reverse_kernel_clipped_mass", 0.0),
        "raw_clip_any_frac": opt.get("raw_clip_any_frac", 0.0),
        "raw_clip_total_residual_std": opt.get("raw_clip_total_residual_std", 0.0),
        "raw_clip_total_residual_abs_mean": opt.get("raw_clip_total_residual_abs_mean", 0.0),
        "raw_clip_total_residual_abs_max": opt.get("raw_clip_total_residual_abs_max", 0.0),
        "raw_clip_surrogate_orig_std": opt.get("raw_clip_surrogate_orig_std", float("nan")),
        "raw_clip_surrogate_clipped_std": opt.get("raw_clip_surrogate_clipped_std", float("nan")),
        "raw_clip_surrogate_orig_ess": opt.get("raw_clip_surrogate_orig_ess", float("nan")),
        "raw_clip_surrogate_clipped_ess": opt.get("raw_clip_surrogate_clipped_ess", float("nan")),
        "raw_clip_surrogate_corr": opt.get("raw_clip_surrogate_corr", float("nan")),
        "raw_clip_any_orig_mass": opt.get("raw_clip_any_orig_mass", 0.0),
        "raw_clip_any_clipped_mass": opt.get("raw_clip_any_clipped_mass", 0.0),
        "timing_enabled": opt.get("timing_enabled", 0.0),
        "timing_model_eval_ms": opt.get("timing_model_eval_ms", float("nan")),
        "timing_base_setup_ms": opt.get("timing_base_setup_ms", float("nan")),
        "timing_jvp_ms": opt.get("timing_jvp_ms", float("nan")),
        "timing_rho_search_ms": opt.get("timing_rho_search_ms", float("nan")),
        "timing_surrogate_rho_ms": opt.get("timing_surrogate_rho_ms", float("nan")),
        "timing_proposal_weight_ms": opt.get("timing_proposal_weight_ms", float("nan")),
        "timing_clip_and_select_ms": opt.get("timing_clip_and_select_ms", float("nan")),
        "timing_weight_temper_ms": opt.get("timing_weight_temper_ms", float("nan")),
        "timing_distributed_metrics_ms": opt.get("timing_distributed_metrics_ms", float("nan")),
        "timing_state_update_ms": opt.get("timing_state_update_ms", float("nan")),
        "timing_step_total_ms": opt.get("timing_step_total_ms", float("nan")),
        "timing_pre_row_total_ms": opt.get("timing_pre_row_total_ms", float("nan")),
        "guidance_mean": target_guidance_coeff,
        "raw_pa_guidance_mean": float(
            raw_target_guidance.mean().item()
        ),
        "residual_guidance_mean": residual_guidance_coeff,
        "proposal_residual_guidance_mean": proposal_residual_guidance_coeff,
        "proposal_guidance_mean": proposal_guidance_coeff,
        "pred_noise_guidance_factor": pred_noise_factor,
        "target_pred_noise_guidance_mean": target_guidance_coeff * pred_noise_factor,
        "proposal_pred_noise_guidance_mean": (
            baseline_kappa
            + proposal_residual_guidance_coeff * pred_noise_factor
        ),
        "proposal_raw_pred_noise_guidance_mean": (
            (
                baseline_kappa
                + proposal_residual_guidance_coeff * pred_noise_factor
            )
            * raw_ab_beta_scale
        ),
        "beta_dim": dim,
        "std_logw_used": float(logw_used.float().std(unbiased=False).item()),
        "std_logw_true": float(logw_true.float().std(unbiased=False).item()),
        "std_logw_true_active": float(logw_true_active.float().std(unbiased=False).item()),
        "std_logw_true_rho0": float(logw_true_rho0.float().std(unbiased=False).item()),
        "std_logw_true_active_rho0": float(logw_true_active_rho0.float().std(unbiased=False).item()),
        "std_logw_surrogate": float(logw_surrogate.float().std(unbiased=False).item()),
        "std_logw_surrogate_unclipped": float(surrogate_unclipped_f.std(unbiased=False).item()),
        "std_logw_surrogate_pre_jvp_shrink": float(
            surrogate_pre_jvp_shrink_f.std(unbiased=False).item()
        ),
        "std_logw_surrogate_no_jvp": float(surrogate_no_jvp_f.std(unbiased=False).item()),
        "used_var_rho": used_var,
        "true_var_rho": true_var,
        "true_active_var_rho": true_active_var,
        "true_var_rho0": true0_var,
        "true_active_var_rho0": true_active0_var,
        "cum_true_var_rho": cum_true_var,
        "cum_true_active_var_rho": cum_true_active_var,
        "cum_true_var_rho0": cum_true0_var,
        "cum_true_active_var_rho0": cum_true_active0_var,
        "surrogate_total_var_rho": surrogate_total_var,
        "surrogate_total_var_rho0": surrogate_total0_var,
        "surrogate_var_rho": surr_var,
        "surrogate_unclipped_var_rho": surr_unclipped_var,
        "surrogate_no_jvp_var_rho": surr_no_jvp_var,
        "surrogate_no_jvp_var_ratio": (
            float(surr_no_jvp_var / surr_unclipped_var)
            if math.isfinite(surr_unclipped_var) and abs(surr_unclipped_var) > 1e-30
            else float("nan")
        ),
        "surrogate_jvp_var_delta": (
            float(surr_unclipped_var - surr_no_jvp_var)
            if math.isfinite(surr_unclipped_var) and math.isfinite(surr_no_jvp_var)
            else float("nan")
        ),
        "surrogate_var_rho0": weighted_variance_np(logw_surrogate_rho0),
        "corr_surrogate_true": finite_corr(logw_surrogate, logw_true),
        "corr_surrogate_unclipped_true": finite_corr(surrogate_unclipped_f, logw_true),
        "corr_surrogate_pre_jvp_shrink_true": finite_corr(
            surrogate_pre_jvp_shrink_f, logw_true
        ),
        "corr_surrogate_no_jvp_true": finite_corr(surrogate_no_jvp_f, logw_true),
        "corr_surrogate_true_rho0": finite_corr(logw_surrogate_rho0, logw_true_rho0),
        "inc_ess_frac": global_ess_frac_from_logw(logw_used),
        "inc_ess_surrogate_frac": global_ess_frac_from_logw(logw_surrogate),
        "inc_ess_surrogate_unclipped_frac": global_ess_frac_from_logw(surrogate_unclipped_f),
        "inc_ess_surrogate_pre_jvp_shrink_frac": global_ess_frac_from_logw(
            surrogate_pre_jvp_shrink_f
        ),
        "inc_ess_surrogate_no_jvp_frac": global_ess_frac_from_logw(surrogate_no_jvp_f),
        "inc_ess_surrogate_no_gate_frac": global_ess_frac_from_logw(
            counterfactual_components["gate"]
        ),
        "inc_ess_surrogate_no_reverse_frac": global_ess_frac_from_logw(
            counterfactual_components["reverse"]
        ),
        "inc_ess_surrogate_no_forward_frac": global_ess_frac_from_logw(
            counterfactual_components["forward"]
        ),
        "jvp_shrink_alpha_effective": float(
            getattr(args, "jvp_shrink_alpha_effective", 1.0)
        ),
        "jvp_finite_fit_cov": jvp_fit_cov,
        "jvp_finite_fit_var": jvp_fit_var,
        "jvp_finite_fit_alpha_clipped": jvp_fit_alpha,
        "cess_ess_frac": float(cess_ess),
        "inc_ess_true_frac": ess_true,
        "inc_ess_true_active_frac": ess_true_active,
        "inc_ess_true_rho0_frac": ess_true0,
        "inc_ess_true_active_rho0_frac": ess_true_active0,
        "inc_ess_true_gain_vs_rho0": float(ess_true - ess_true0),
        "inc_ess_true_active_gain_vs_rho0": float(ess_true_active - ess_true_active0),
        "cum_ess_true_frac": cum_ess_true,
        "cum_ess_true_active_frac": cum_ess_true_active,
        "cum_ess_true_rho0_frac": cum_ess_true0,
        "cum_ess_true_active_rho0_frac": cum_ess_true_active0,
        "true_total_ess_frac": cum_ess_true,
        "true_active_total_ess_frac": cum_ess_true_active,
        "true_total_ess_rho0_frac": cum_ess_true0,
        "surrogate_total_ess_frac": global_ess_frac_from_logw(surrogate_total),
        "surrogate_active_total_ess_frac": global_ess_frac_from_logw(active_total_f),
        "surrogate_total_ess_rho0_frac": global_ess_frac_from_logw(surrogate_total0),
        "cum_ess_true_gain_vs_rho0": float(cum_ess_true - cum_ess_true0),
        "cum_ess_true_active_gain_vs_rho0": float(cum_ess_true_active - cum_ess_true_active0),
        "cess_ess_true_frac": cess_true,
        "cess_ess_true_active_frac": cess_true_active,
        "cess_ess_true_rho0_frac": cess_true0,
        "cess_ess_true_active_rho0_frac": cess_true_active0,
        "cess_ess_true_gain_vs_rho0": float(cess_true - cess_true0),
        "cess_ess_true_active_gain_vs_rho0": float(cess_true_active - cess_true_active0),
        "true_var_improvement_vs_rho0": float(true0_var - true_var),
        "true_active_var_improvement_vs_rho0": float(true_active0_var - true_active_var),
        "cum_true_var_improvement_vs_rho0": float(cum_true0_var - cum_true_var),
        "cum_true_active_var_improvement_vs_rho0": float(cum_true_active0_var - cum_true_active_var),
        "cum_ess_frac": global_ess_frac_from_logw(active_total_f),
        "full_cum_ess_frac": global_ess_frac_from_logw(logw_total_new),
        "cum_ess_after_resample": cum_ess_after_resample,
        "post_step_logw_std": float(post_step_logw.std(unbiased=False).item()),
        "post_step_logw_abs_mean": float(post_step_logw.abs().mean().item()),
        "post_step_logw_abs_max": float(post_step_logw.abs().max().item()),
        "post_step_ess_frac": global_ess_frac_from_logw(post_step_logw),
        "resample_ess_mode": str(getattr(args, "resample_ess_mode", "cum")),
        "resample_weight_mode": str(getattr(args, "resample_weight_mode", "cum")),
        "resample_ess_used": float(resample_ess_used),
        "resample_weight_ess_frac": float(resample_weight_ess),
        "resample_allowed": int(resample_allowed),
        "resample_ess_rearm": float(getattr(args, "resample_ess_rearm", 0.0)),
        "resample_hysteresis_armed": int(opt.get("resample_hysteresis_armed", 1.0)),
        "resample_hysteresis_rearmed": int(opt.get("resample_hysteresis_rearmed", 0.0)),
        "resample_hysteresis_blocked": int(opt.get("resample_hysteresis_blocked", 0.0)),
        "no_resample_last_steps": int(getattr(args, "no_resample_last_steps", 0)),
        "no_rho_last_steps": int(getattr(args, "no_rho_last_steps", 0)),
        "no_monitor_last_steps": int(getattr(args, "no_monitor_last_steps", 0)),
        "resampled": int(do_resample),
        "skipped_weight": int(skipped_weight),
    }
    row.update(beta_variance_stats(beta_flat, lhat_temp=args.lhat_temp))
    row.update(tensor_stats("theta", theta_f))
    row.update(tensor_stats("lhat", lhat))
    row.update(tensor_stats("lhat_new", lhat_new))
    row.update(
        tensor_stats(
            "centered_lhat_increment_exact",
            centered_lhat_increment(
                lhat_new,
                lhat,
                kappa_new=baseline_kappa_new,
                kappa=baseline_kappa,
                lhat_temp=float(args.lhat_temp),
            ),
        )
    )
    if origin_ids is not None:
        row.update(genealogy_stats(origin_ids, int(args.n_particles)))
    row.update(
        lhat_resampling_diagnostics(
            lhat=lhat,
            lhat_new=lhat_new,
            kernel_lr=kernel_lr,
            logw_used=logw_used,
            logw_true=logw_true,
            logw_surrogate=logw_surrogate,
            ancestor_idx=ancestor_idx if do_resample else None,
        )
    )
    row.update(tensor_stats("theta_term_exact", theta_term_exact))
    row.update(tensor_stats("kernel_lr", kernel_lr))
    row.update(drift_component_stats(drift_components))
    row.update(decomposition_stats("decomp", surrogate_decomp, logw_surrogate))
    row.update(decomposition_stats("decomp_rho0", surrogate_decomp_rho0, logw_surrogate_rho0))
    row.update(jvp_tail_diagnostics(surrogate_decomp, logw_surrogate, drift_components))
    row.update(decomposition_stats("finite", finite_components, logw_true, names=FINITE_DECOMP_NAMES))
    row.update(
        decomposition_stats(
            "finite_rho0",
            finite_components_rho0,
            logw_true_rho0,
            names=FINITE_DECOMP_NAMES,
        )
    )
    row.update(kernel_tail_diagnostics(surrogate_decomp, finite_components, logw_surrogate, logw_true))
    row.update(expansion_math_stats("math", math_components, finite_components, logw_true))
    row.update(comparison_stats("surrogate_true", logw_surrogate, logw_true))
    row.update(
        comparison_stats(
            "surrogate_unclipped_true",
            surrogate_unclipped_f,
            logw_true,
        )
    )
    row.update(
        comparison_stats(
            "surrogate_pre_jvp_shrink_true",
            surrogate_pre_jvp_shrink_f,
            logw_true,
        )
    )
    row.update(comparison_stats("surrogate_true_rho0", logw_surrogate_rho0, logw_true_rho0))
    row.update(agreement_tail_stats("surrogate_true", logw_surrogate, logw_true))
    row.update(agreement_tail_stats("surrogate_true_rho0", logw_surrogate_rho0, logw_true_rho0))
    return row


def distributed_worker(rank: int, args: argparse.Namespace, devices: list[str]) -> None:
    world_size = len(devices)
    device = devices[rank]
    args.device = device
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        prompt, negative, _ = resolve_prompt(args)
        local_n = args.n_particles // world_size
        global_start = rank * local_n

        pipe = load_sd_pipeline(device, args.model_id)
        if args.model_dtype == "fp32":
            pipe.unet.to(dtype=torch.float32)
            pipe.vae.to(dtype=torch.float32)
            if pipe.text_encoder is not None:
                pipe.text_encoder.to(dtype=torch.float32)
        if args.attention_processor == "legacy":
            from diffusers.models.attention_processor import AttnProcessor

            pipe.unet.set_attn_processor(AttnProcessor())
        if args.attention_slicing != "none":
            pipe.enable_attention_slicing(args.attention_slicing)
        pipe._dng_unet_chunk_size = int(args.unet_chunk_size)
        pipe._pa_gate_cfg_scale = float(args.cfg_scale)
        pipe._pa_gate_base_lambda_neg = float(args.base_lambda_neg)
        pipe._pa_gate_linear_baseline_kappa = float(args.kappa_start)
        pipe._pa_gate_gate_ratio_mode = str(args.gate_ratio_mode)
        height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        uncond_embeds = encode_text(pipe, [""] * local_n, device)
        pos_embeds = encode_text(pipe, [prompt["positive"]] * local_n, device)
        neg_embeds = encode_text(pipe, [negative] * local_n, device)
        prompt_embeds = torch.cat([uncond_embeds, pos_embeds, neg_embeds], dim=0)

        pipe.scheduler.set_timesteps(args.steps, device=device)
        timesteps = list(pipe.scheduler.timesteps)
        linear_baseline_kappa_path = [
            linear_baseline_kappa_at_state(
                state, len(timesteps), args
            )
            for state in range(len(timesteps) + 1)
        ]
        mean_linear_baseline_kappa_path = float(
            np.mean(linear_baseline_kappa_path[:-1])
        )
        init_generators = [
            torch.Generator(device=device).manual_seed(args.seed + global_start + i)
            for i in range(local_n)
        ]
        noise_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + rank)
        resample_gen = torch.Generator(device=device).manual_seed(args.seed + 200000)

        latents = pipe.prepare_latents(
            local_n,
            pipe.unet.config.in_channels,
            height,
            width,
            pos_embeds.dtype,
            device,
            init_generators,
            latents=None,
        )
        lhat = torch.zeros(local_n, device=device, dtype=torch.float32)
        logw_total = torch.zeros(local_n, device=device, dtype=torch.float32)
        origin_ids = torch.arange(
            global_start,
            global_start + local_n,
            device=device,
            dtype=torch.long,
        )
        rho_current = float(args.rho_init)
        resample_armed = True
        rows: list[dict[str, Any]] = []
        timing_records: list[dict[str, float]] = []
        early_stopped = False
        early_stop_step = -1
        early_stop_unique_roots = args.n_particles
        early_stop_effective_roots = float(args.n_particles)
        start_time = time.time()

        if rank == 0:
            print(
                f"[run] {len(timesteps)} reverse steps, {args.n_particles} particles, "
                f"rho_every={args.rho_every}",
                flush=True,
            )

        # Use no_grad rather than inference_mode because inference_mode disables
        # forward-mode AD, which is needed by --jvp-mode forward-ad/auto.
        with torch.no_grad():
            for step, t in enumerate(timesteps):
                timing_enabled = timing_step_enabled(args, step, len(timesteps))
                timing_values: dict[str, float] = {"timing_enabled": float(timing_enabled)}
                timing_last_box = [timing_now(device, timing_enabled)]
                timing_total_start = timing_last_box[0]
                timing_jvp_ms = [0.0]
                timing_values["timing_rho_search_ms"] = 0.0
                timing_values["timing_rho_search_active"] = 0.0

                def timing_mark(name: str) -> None:
                    if not timing_enabled:
                        return
                    now = timing_now(device, True)
                    timing_values[f"timing_{name}_ms"] = float(
                        (now - timing_last_box[0]) * 1000.0
                    )
                    timing_last_box[0] = now

                def compute_jbeta_z_timed(*cb_args: Any, **cb_kwargs: Any) -> tuple[torch.Tensor, str]:
                    if not timing_enabled:
                        return compute_jbeta_z(*cb_args, **cb_kwargs)
                    start = timing_now(device, True)
                    out = compute_jbeta_z(*cb_args, **cb_kwargs)
                    end = timing_now(device, True)
                    timing_jvp_ms[0] += float((end - start) * 1000.0)
                    return out

                t_int = int(t)
                c, eta = schedule_at_state(step, len(timesteps), args)
                c_new, eta_new = schedule_at_state(step + 1, len(timesteps), args)
                gate_power = gate_power_at_state(
                    step, len(timesteps), args
                )
                gate_power_new = gate_power_at_state(
                    step + 1, len(timesteps), args
                )
                linear_baseline_kappa = linear_baseline_kappa_path[step]
                linear_baseline_kappa_new = (
                    linear_baseline_kappa_path[step + 1]
                )
                # Keep the compatibility fallback synchronized even though all
                # production calls below pass the state value explicitly.
                pipe._pa_gate_linear_baseline_kappa = float(
                    linear_baseline_kappa
                )

                out_a_base, out_a_gate, out_b_gate, beta_l = ab_outputs_and_beta(
                    pipe,
                    latents,
                    t,
                    prompt_embeds,
                    linear_baseline_kappa=linear_baseline_kappa,
                )
                timing_mark("model_eval")
                mu_a, variance = ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_a_base, t_int, latents
                )
                gate_mu_a, _ = ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_a_gate, t_int, latents
                )
                gate_mu_b, _ = ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_b_gate, t_int, latents
                )
                var_f = variance.to(device=latents.device, dtype=torch.float32)
                forward_mean_mu_y, forward_variance = forward_kernel_mean_variance(
                    pipe.scheduler, t_int, latents
                )
                forward_displacement = forward_mean_mu_y.float() - latents.float()
                skipped_weight = should_skip_smc_weight(
                    pipe.scheduler, t_int, var_f, args.min_weight_variance
                )
                if skipped_weight and not math.isclose(
                    float(linear_baseline_kappa),
                    float(linear_baseline_kappa_new),
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "Cannot skip an SMC transition while the centered "
                        "kappa schedule changes; plateau the schedule before "
                        "the zero-variance transition"
                    )
                theta = theta_from_lhat(lhat, c, eta, args.theta_eps).float()
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=latents.dtype,
                    generator=noise_gen,
                )
                z = var_f.sqrt().to(dtype=latents.dtype) * noise
                timing_mark("base_setup")

                opt = {
                    "rho": rho_current,
                    "objective_start": float("nan"),
                    "objective_final": float("nan"),
                    "grad_start": float("nan"),
                    "grad_final": float("nan"),
                    "accepted": float("nan"),
                    "surrogate_poly_degree": float("nan"),
                    "raw_clip_jvp_in_rho_objective": float(
                        bool(
                            args.raw_clip_jvp_in_rho_objective
                            and (
                                float(args.raw_clip_jvp) > 0.0
                                or int(getattr(args, "rank_clip_jvp_topk", 0)) > 0
                            )
                        )
                    ),
                }
                coeffs = None
                local_decomp_terms: dict[str, torch.Tensor] | None = None
                local_decomp_terms_rho0: dict[str, torch.Tensor] | None = None
                local_decomp_terms_rho_start: dict[str, torch.Tensor] | None = None
                local_finite_components: dict[str, torch.Tensor] | None = None
                local_finite_components_rho0: dict[str, torch.Tensor] | None = None
                local_finite_components_rho_start: dict[str, torch.Tensor] | None = None
                local_math_components: dict[str, torch.Tensor] | None = None
                h_alpha_s: torch.Tensor | None = None
                jvp_used = "none"
                no_rho_last_steps = max(0, int(getattr(args, "no_rho_last_steps", 0)))
                rho_blocked_final = no_rho_last_steps > 0 and step >= len(timesteps) - no_rho_last_steps
                monitor_last_skip = max(0, int(getattr(args, "no_monitor_last_steps", 0)))
                monitor_allowed = not (monitor_last_skip > 0 and step >= len(timesteps) - monitor_last_skip)
                needs_opt = (
                    bool(args.rho_update)
                    and not rho_blocked_final
                    and step % max(1, int(args.rho_every)) == 0
                )
                # Freeze the optimizer's actual starting point before any
                # search update.  Finite fact-check runs compare the raw,
                # pre-JVP-temper surrogate and exact finite log weight at this
                # same rho and with the same Brownian draw.
                rho_opt_start = (
                    rho_current
                    if args.rho_warm_start
                    else float(args.rho_init)
                )
                needs_surrogate = (
                    (not skipped_weight)
                    and (
                        args.prop_weight_mode == "surrogate"
                        or args.surrogate_every_step
                        or needs_opt
                    )
                )
                if needs_surrogate:
                    jbeta_z = None
                    jbeta_base_brownian = None
                    jbeta_base_guidance0 = None
                    jbeta_full_deterministic = None
                    jbeta_guidance_rho0 = None
                    jvp_configured = bool(args.surrogate_gamma_jvp)
                    jvp_window_active = jvp_step_window_active(
                        args,
                        step,
                        len(timesteps),
                    )
                    # Outside this fixed, ESS-independent window, disable only
                    # J beta[Delta]. Rank-one gate curvature and every non-JVP
                    # term remain in both rho selection and propagated weights.
                    use_gamma_jvp = bool(
                        jvp_configured and jvp_window_active
                    )
                    jvp_skip_frac = max(
                        0.0,
                        float(getattr(args, "jvp_skip_below_gate_power_frac", 0.0)),
                    )
                    gate_power_frac = (
                        abs(float(gate_power)) / max(abs(float(args.gate_power)), 1e-30)
                    )
                    skip_small_gate_jvp = bool(
                        use_gamma_jvp
                        and jvp_skip_frac > 0.0
                        and gate_power_frac < jvp_skip_frac
                    )
                    if not jvp_configured:
                        jvp_used = "disabled"
                    elif not jvp_window_active:
                        jvp_used = (
                            "fixed-step-window-skip:"
                            f"step={step},first="
                            f"{int(getattr(args, 'no_jvp_first_steps', 0))},"
                            "last="
                            f"{int(getattr(args, 'no_jvp_last_steps', 0))}"
                        )
                    elif skip_small_gate_jvp:
                        # Every retained JVP log-weight term is linear in the
                        # current gate exponent.  Screening sweeps may therefore
                        # omit it below one fixed, predeclared relative
                        # gate-power tolerance.  This rule is independent of
                        # ESS and particle scores; final candidates can set the
                        # tolerance to zero for full-JVP confirmation.
                        jvp_zero = torch.zeros_like(beta_l.float())
                        jbeta_z = jvp_zero
                        if args.surrogate_gamma_delta == "base_guidance0":
                            jbeta_base_guidance0 = jvp_zero
                        elif args.surrogate_gamma_delta == "full_deterministic":
                            jbeta_full_deterministic = jvp_zero
                        elif args.surrogate_gamma_delta in {
                            "base_brownian",
                            "full",
                        }:
                            jbeta_base_brownian = jvp_zero
                        if args.surrogate_gamma_delta == "full":
                            jbeta_guidance_rho0 = jvp_zero
                        jvp_used = (
                            "fixed-small-gate-skip:"
                            f"{gate_power_frac:.4g}<{jvp_skip_frac:.4g}"
                        )
                    elif args.surrogate_gamma_delta in {"base_brownian", "base_guidance0", "full_deterministic", "full"}:
                        base_plus_brownian = mu_a.float() - latents.float() + z.float()
                        base_deterministic = mu_a.float() - latents.float()
                        tempered_gate = (
                            float(gate_power)
                            * theta.float()[:, None, None, None]
                            * float(eta)
                            * float(args.lhat_temp)
                        )
                        residual_gate = (
                            float(args.pa_residual_alpha)
                            * (
                                tempered_gate
                                - float(linear_baseline_kappa)
                            )
                        )
                        guidance_rho0 = -var_f * residual_gate * beta_l.float()
                        if args.surrogate_gamma_delta == "base_guidance0":
                            jbeta_base_guidance0, jvp_used_base = compute_jbeta_z_timed(
                                pipe,
                                latents,
                                t,
                                prompt_embeds,
                                base_plus_brownian + guidance_rho0,
                                mode=args.jvp_mode,
                                fd_eps=args.jvp_eps,
                                rank=rank,
                            )
                            jvp_used = f"{jvp_used_base}:base+brownian+guidance_rho0"
                        elif args.surrogate_gamma_delta == "full_deterministic":
                            jbeta_full_deterministic, jvp_used_base = compute_jbeta_z_timed(
                                pipe,
                                latents,
                                t,
                                prompt_embeds,
                                base_plus_brownian + guidance_rho0,
                                mode=args.jvp_mode,
                                fd_eps=args.jvp_eps,
                                rank=rank,
                            )
                            jvp_used = f"{jvp_used_base}:base+brownian+guidance_rho0"
                        else:
                            jbeta_base_brownian, jvp_used_base = compute_jbeta_z_timed(
                                pipe,
                                latents,
                                t,
                                prompt_embeds,
                                base_plus_brownian,
                                mode=args.jvp_mode,
                                fd_eps=args.jvp_eps,
                                rank=rank,
                            )
                            jvp_used = f"{jvp_used_base}:base+brownian"
                        if args.surrogate_gamma_delta == "full":
                            jbeta_guidance_rho0, jvp_used_guidance = compute_jbeta_z_timed(
                                pipe,
                                latents,
                                t,
                                prompt_embeds,
                                guidance_rho0,
                                mode=args.jvp_mode,
                                fd_eps=args.jvp_eps,
                                rank=rank,
                            )
                            jvp_used = f"{jvp_used_base}:base+brownian,{jvp_used_guidance}:guidance"
                        jbeta_z = torch.zeros_like(beta_l.float())
                    else:
                        jbeta_z, jvp_used = compute_jbeta_z_timed(
                            pipe,
                            latents,
                            t,
                            prompt_embeds,
                            z,
                            mode=args.jvp_mode,
                            fd_eps=args.jvp_eps,
                            rank=rank,
                        )
                    gate_nu_a = (latents.float() - gate_mu_a.float()).flatten(1)
                    gate_nu_b = (latents.float() - gate_mu_b.float()).flatten(1)
                    alpha_l = (gate_nu_a.pow(2).sum(dim=1) - gate_nu_b.pow(2).sum(dim=1)) / (2.0 * var_f)
                    h_alpha_s = (
                        float(eta) * float(args.lhat_temp) * alpha_l
                        + math.log(max(float(c_new), 1e-30) / max(float(c), 1e-30))
                        + (float(eta_new) - float(eta)) * lhat.float()
                    )
                    h_lhat_s = float(args.lhat_temp) * alpha_l
                    surrogate_kwargs = {
                        "latents": latents,
                        "mu_a": mu_a,
                        "variance": var_f,
                        "forward_variance": forward_variance,
                        "forward_displacement": forward_displacement,
                        "z": z,
                        "beta_l": beta_l,
                        "jbeta_z": jbeta_z,
                        "jbeta_base_brownian": jbeta_base_brownian,
                        "jbeta_base_guidance0": jbeta_base_guidance0,
                        "jbeta_full_deterministic": jbeta_full_deterministic,
                        "jbeta_guidance_rho0": jbeta_guidance_rho0,
                        "gamma_delta_mode": args.surrogate_gamma_delta,
                        "gamma_jvp": use_gamma_jvp,
                        "theta": theta,
                        "eta": eta,
                        "lhat_temp": args.lhat_temp,
                        "gate_power": gate_power,
                        "h_alpha_s": h_alpha_s,
                        "h_lhat_s": h_lhat_s,
                        "lhat": lhat,
                        "linear_baseline_kappa": linear_baseline_kappa,
                        "linear_baseline_kappa_new": (
                            linear_baseline_kappa_new
                        ),
                        "pa_residual_alpha": args.pa_residual_alpha,
                        "jvp_shrink_alpha": args.jvp_shrink_alpha_effective,
                    }
                    hybrid_weight_mode = bool(
                        getattr(args, "hybrid_exact_gate_reverse", False)
                    )
                    if hybrid_weight_mode:
                        # The hybrid optimizer reconstructs the retained
                        # component polynomials together in one node pass.
                        # Avoid a duplicate latent-sized coefficient pass here.
                        opt["surrogate_poly_degree"] = float(
                            3 if args.surrogate_gamma_delta == "full" else 2
                        )
                    elif args.surrogate_gamma_delta == "full":
                        coeffs = surrogate_cubic_coefficients(**surrogate_kwargs)
                        opt["surrogate_poly_degree"] = 3.0
                    else:
                        coeffs = surrogate_coefficients(**surrogate_kwargs)
                        opt["surrogate_poly_degree"] = 2.0
                    if needs_opt:
                        rho_search_start = timing_now(device, True) if timing_enabled else 0.0
                        rho_objective = str(getattr(args, "rho_objective", "total_logw"))
                        hybrid_rho_enabled = hybrid_weight_mode
                        prepared_rho_ess: dict[str, Any] | None = None
                        global_logw_total: torch.Tensor | None = None
                        if not hybrid_rho_enabled:
                            assert coeffs is not None
                            coeffs_for_opt = coeffs
                            if args.surrogate_gamma_delta == "full":
                                coeffs_for_ess = coeffs
                            else:
                                coeffs_for_ess = (
                                    coeffs[2],
                                    coeffs[1],
                                    coeffs[0],
                                )
                            prepared_rho_ess = prepare_global_rho_ess_state(
                                coeffs_for_ess,
                                logw_total.float(),
                                surrogate_kwargs=surrogate_kwargs,
                                raw_clip_jvp=float(args.raw_clip_jvp),
                                rank_clip_jvp_topk=int(args.rank_clip_jvp_topk),
                                adaptive_rank_clip_jvp=bool(args.adaptive_rank_clip_jvp),
                                raw_clip_jvp_in_rho_objective=bool(args.raw_clip_jvp_in_rho_objective),
                                world_size=world_size,
                            )
                            if int(prepared_rho_ess["carry_logw"].numel()) != int(args.n_particles):
                                raise RuntimeError(
                                    "Rho ESS search must use the complete global population: "
                                    f"expected {args.n_particles}, "
                                    f"got {prepared_rho_ess['carry_logw'].numel()}"
                                )
                            global_logw_total = prepared_rho_ess["carry_logw"]
                        if hybrid_rho_enabled:
                            # The hybrid search below uses exact cheap gate and
                            # reverse factors, so do not also run the retained
                            # polynomial ESS optimizer.
                            pass
                        elif rho_objective == "total_ess":
                            opt = optimize_rho_global_ess_polynomial(
                                coeffs_for_ess,
                                global_logw_total,
                                rho_init=rho_opt_start,
                                rho_steps=args.rho_steps,
                                rho_cap=args.rho_cap,
                                guard_rho0=args.rho_guard_rho0,
                                accept_only_improve=args.rho_accept_only_improve,
                                surrogate_logw_clip=float(args.surrogate_logw_clip),
                                raw_clip_jvp=float(args.raw_clip_jvp),
                                rank_clip_jvp_topk=int(args.rank_clip_jvp_topk),
                                adaptive_rank_clip_jvp=bool(args.adaptive_rank_clip_jvp),
                                adaptive_rank_clip_jvp_ess_thresholds=str(args.adaptive_rank_clip_jvp_ess_thresholds),
                                adaptive_rank_clip_jvp_topks=str(args.adaptive_rank_clip_jvp_topks),
                                raw_clip_center=str(args.raw_clip_center),
                                raw_clip_count_eps=float(args.raw_clip_count_eps),
                                raw_clip_jvp_in_rho_objective=bool(args.raw_clip_jvp_in_rho_objective),
                                surrogate_kwargs=surrogate_kwargs,
                                world_size=world_size,
                                global_start=global_start,
                                local_n=local_n,
                                ess_prefix="rho_total_ess",
                                prepared_global_state=prepared_rho_ess,
                                ess_search_mode=str(args.rho_ess_search_mode),
                                ess_coarse_candidates=int(args.rho_ess_coarse_candidates),
                            )
                        elif rho_objective == "incremental_ess":
                            opt = optimize_rho_global_ess_polynomial(
                                coeffs_for_ess,
                                torch.zeros_like(global_logw_total),
                                rho_init=rho_opt_start,
                                rho_steps=args.rho_steps,
                                rho_cap=args.rho_cap,
                                guard_rho0=args.rho_guard_rho0,
                                accept_only_improve=args.rho_accept_only_improve,
                                surrogate_logw_clip=float(args.surrogate_logw_clip),
                                raw_clip_jvp=float(args.raw_clip_jvp),
                                rank_clip_jvp_topk=int(args.rank_clip_jvp_topk),
                                adaptive_rank_clip_jvp=bool(args.adaptive_rank_clip_jvp),
                                adaptive_rank_clip_jvp_ess_thresholds=str(args.adaptive_rank_clip_jvp_ess_thresholds),
                                adaptive_rank_clip_jvp_topks=str(args.adaptive_rank_clip_jvp_topks),
                                raw_clip_center=str(args.raw_clip_center),
                                raw_clip_count_eps=float(args.raw_clip_count_eps),
                                raw_clip_jvp_in_rho_objective=bool(args.raw_clip_jvp_in_rho_objective),
                                surrogate_kwargs=surrogate_kwargs,
                                world_size=world_size,
                                global_start=global_start,
                                local_n=local_n,
                                ess_prefix="rho_inc_ess",
                                prepared_global_state=prepared_rho_ess,
                                ess_search_mode=str(args.rho_ess_search_mode),
                                ess_coarse_candidates=int(args.rho_ess_coarse_candidates),
                            )
                        elif rho_objective == "total_logw":
                            local_w = torch.full_like(
                                logw_total.float(),
                                1.0 / float(args.n_particles),
                            )
                            if args.surrogate_gamma_delta == "full":
                                coeffs_for_opt = (
                                    coeffs[0] + logw_total.float(),
                                    *coeffs[1:],
                                )
                            else:
                                coeffs_for_opt = (
                                    coeffs[0],
                                    coeffs[1],
                                    coeffs[2] + logw_total.float(),
                                )
                        elif rho_objective == "incremental_weighted":
                            global_w = global_normalized_weights_from_logw(global_logw_total)
                            local_w = global_w[global_start : global_start + local_n]
                        else:
                            raise ValueError(f"Unknown --rho-objective={rho_objective!r}")
                        if (
                            not hybrid_rho_enabled
                            and rho_objective
                            not in {"total_ess", "incremental_ess"}
                        ):
                            init = rho_opt_start
                            if args.surrogate_gamma_delta == "full":
                                opt = optimize_rho_global_polynomial(
                                    coeffs_for_opt,
                                    local_w,
                                    rho_init=init,
                                    rho_lr=args.rho_lr,
                                    rho_steps=args.rho_steps,
                                    rho_cap=args.rho_cap,
                                    optimizer=args.rho_optimizer,
                                    accept_only_improve=args.rho_accept_only_improve,
                                    line_search_halvings=args.rho_line_search_halvings,
                                    guard_rho0=args.rho_guard_rho0,
                                    over1_penalty=args.rho_over1_penalty,
                                )
                            else:
                                opt = optimize_rho_global(
                                    coeffs_for_opt[0],
                                    coeffs_for_opt[1],
                                    coeffs_for_opt[2],
                                    local_w,
                                    rho_init=init,
                                    rho_lr=args.rho_lr,
                                    rho_steps=args.rho_steps,
                                    rho_cap=args.rho_cap,
                                    optimizer=args.rho_optimizer,
                                    accept_only_improve=args.rho_accept_only_improve,
                                    line_search_halvings=args.rho_line_search_halvings,
                                    guard_rho0=args.rho_guard_rho0,
                                    over1_penalty=args.rho_over1_penalty,
                                )
                        hybrid_rho_diag: dict[str, Any] | None = None
                        if hybrid_rho_enabled:
                            if rho_objective not in {
                                "incremental_ess",
                                "total_ess",
                            }:
                                raise ValueError(
                                    "--hybrid-exact-gate-reverse currently "
                                    "requires --rho-objective incremental_ess "
                                    "or total_ess"
                                )
                            opt = optimize_rho_hybrid_incremental_ess(
                                surrogate_kwargs=surrogate_kwargs,
                                lhat=lhat,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power_new=gate_power_new,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                                lhat_clip=float(args.lhat_clip),
                                rho_init=rho_opt_start,
                                rho_cap=float(args.rho_cap),
                                rho_steps=int(args.rho_steps),
                                ess_search_mode=str(args.rho_ess_search_mode),
                                ess_coarse_candidates=int(
                                    args.rho_ess_coarse_candidates
                                ),
                                rho_lr=float(args.rho_lr),
                                rho_optimizer=str(args.rho_optimizer),
                                line_search_halvings=int(
                                    args.rho_line_search_halvings
                                ),
                                guard_rho0=bool(args.rho_guard_rho0),
                                accept_only_improve=bool(
                                    args.rho_accept_only_improve
                                ),
                                min_ess_gain=float(args.rho_ess_min_gain),
                                surrogate_logw_clip=float(
                                    args.surrogate_logw_clip
                                ),
                                raw_clip_jvp=float(args.raw_clip_jvp),
                                rank_clip_jvp_topk=int(
                                    args.rank_clip_jvp_topk
                                ),
                                adaptive_rank_clip_jvp=bool(
                                    args.adaptive_rank_clip_jvp
                                ),
                                adaptive_rank_clip_jvp_ess_thresholds=str(
                                    args.adaptive_rank_clip_jvp_ess_thresholds
                                ),
                                adaptive_rank_clip_jvp_topks=str(
                                    args.adaptive_rank_clip_jvp_topks
                                ),
                                raw_clip_center=str(args.raw_clip_center),
                                raw_clip_count_eps=float(
                                    args.raw_clip_count_eps
                                ),
                                raw_clip_jvp_in_rho_objective=bool(
                                    args.raw_clip_jvp_in_rho_objective
                                ),
                                world_size=world_size,
                                expected_particles=int(args.n_particles),
                                carry_logw=logw_total.float(),
                                rho_objective=rho_objective,
                            )
                            hybrid_rho_diag = dict(opt)
                        opt["rho_objective"] = rho_objective
                        if "raw_clip_jvp_in_rho_objective" not in opt:
                            opt["raw_clip_jvp_in_rho_objective"] = 0.0
                        rho_current = float(opt["rho"])
                        if not hybrid_rho_enabled:
                            assert prepared_rho_ess is not None
                            assert global_logw_total is not None
                            ess_diag_kwargs = {
                                "rho_start": rho_opt_start,
                                "rho_final": rho_current,
                                "surrogate_logw_clip": float(args.surrogate_logw_clip),
                                "raw_clip_jvp": float(args.raw_clip_jvp),
                                "rank_clip_jvp_topk": int(args.rank_clip_jvp_topk),
                                "adaptive_rank_clip_jvp": bool(args.adaptive_rank_clip_jvp),
                                "adaptive_rank_clip_jvp_ess_thresholds": str(args.adaptive_rank_clip_jvp_ess_thresholds),
                                "adaptive_rank_clip_jvp_topks": str(args.adaptive_rank_clip_jvp_topks),
                                "raw_clip_center": str(args.raw_clip_center),
                                "raw_clip_count_eps": float(args.raw_clip_count_eps),
                                "raw_clip_jvp_in_rho_objective": bool(args.raw_clip_jvp_in_rho_objective),
                                "surrogate_kwargs": surrogate_kwargs,
                                "world_size": world_size,
                                "global_start": global_start,
                                "local_n": local_n,
                                "prepared_global_state": prepared_rho_ess,
                            }
                            opt.update(
                                rho_ess_triplet_from_polynomial(
                                    coeffs_for_ess,
                                    global_logw_total,
                                    prefix="rho_total_ess",
                                    **ess_diag_kwargs,
                                )
                            )
                            opt.update(
                                rho_ess_triplet_from_polynomial(
                                    coeffs_for_ess,
                                    torch.zeros_like(global_logw_total),
                                    prefix="rho_inc_ess",
                                    **ess_diag_kwargs,
                                )
                            )
                        if hybrid_rho_diag is not None:
                            # Keep the polynomial diagnostics in the CSV under
                            # their usual fields only for comparison; the
                            # operational rho/ESS/objective must be the hybrid
                            # exact-gate/reverse result.
                            opt.update(hybrid_rho_diag)
                            opt["rho_objective"] = rho_objective
                        if timing_enabled:
                            timing_values["timing_rho_search_ms"] = float(
                                (timing_now(device, True) - rho_search_start) * 1000.0
                            )
                            timing_values["timing_rho_search_active"] = 1.0
                    if not hybrid_weight_mode:
                        assert coeffs is not None
                        local_decomp_terms = surrogate_decomposition_from_terms(
                            local_surrogate_terms(
                                rho_current, **surrogate_kwargs
                            )
                        )
                        local_decomp_terms_rho0 = (
                            surrogate_decomposition_from_terms(
                                local_surrogate_terms(0.0, **surrogate_kwargs)
                            )
                        )
                if timing_enabled:
                    timing_values["timing_jvp_ms"] = timing_jvp_ms[0]
                timing_mark("surrogate_rho")

                if skipped_weight:
                    mu_rho, _ = pa_gate_proposal_mean(
                        rho_current,
                        latents=latents,
                        mu_a=mu_a,
                        variance=var_f,
                        beta_l=beta_l,
                        theta=theta,
                        eta=eta,
                        lhat_temp=args.lhat_temp,
                        gate_power=gate_power,
                        linear_baseline_kappa=linear_baseline_kappa,
                        pa_residual_alpha=args.pa_residual_alpha,
                    )
                    x_new = mu_rho.to(dtype=latents.dtype)
                    lhat_new = lhat
                    logw_true = torch.zeros_like(lhat)
                    kernel_lr = torch.zeros_like(lhat)
                    logw_true_rho0 = torch.zeros_like(lhat)
                    logw_true_rho_start = torch.zeros_like(lhat)
                    logw_surrogate_pretemper_rho_start = torch.zeros_like(
                        lhat
                    )
                    local_finite_components = {
                        name: torch.zeros_like(lhat.float()) for name in FINITE_DECOMP_NAMES
                    }
                    local_finite_components_rho0 = {
                        name: torch.zeros_like(lhat.float()) for name in FINITE_DECOMP_NAMES
                    }
                    local_finite_components_rho_start = {
                        name: torch.zeros_like(lhat.float())
                        for name in FINITE_DECOMP_NAMES
                    }
                    local_math_components = nan_terms(MATH_ALL_NAMES, lhat)
                    if coeffs is None:
                        logw_surrogate = torch.zeros_like(lhat)
                        logw_surrogate_rho0 = torch.zeros_like(lhat)
                else:
                    diag_every = int(args.finite_diagnostic_every)
                    wants_finite_diagnostic = (
                        monitor_allowed
                        and diag_every > 0
                        and (step % diag_every == 0 or step == len(timesteps) - 1)
                    )

                    def finite_diagnostic_at_rho(
                        rho_value: float,
                    ) -> tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        dict[str, torch.Tensor],
                    ]:
                        return finite_weight_diagnostics_for_rho(
                            rho_value,
                            pipe=pipe,
                            scheduler=pipe.scheduler,
                            timestep=t_int,
                            latents=latents,
                            prompt_embeds=prompt_embeds,
                            lhat=lhat,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            mu_a=mu_a,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            variance=var_f,
                            forward_variance=forward_variance,
                            forward_displacement=forward_displacement,
                            beta_l=beta_l,
                            theta=theta,
                            z=z,
                            lhat_temp=args.lhat_temp,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_clip=args.lhat_clip,
                            theta_eps=args.theta_eps,
                            linear_baseline_kappa=linear_baseline_kappa,
                            linear_baseline_kappa_new=(
                                linear_baseline_kappa_new
                            ),
                            pa_residual_alpha=args.pa_residual_alpha,
                        )

                    needs_current_finite = args.prop_weight_mode == "true" or wants_finite_diagnostic
                    if needs_current_finite:
                        (
                            x_new,
                            lhat_new,
                            logw_true,
                            kernel_lr,
                            local_finite_components,
                        ) = finite_diagnostic_at_rho(
                            rho_current
                        )
                    else:
                        x_new, lhat_new, kernel_lr = sample_proposal_and_lhat(
                            rho_current,
                            latents=latents,
                            lhat=lhat,
                            mu_a=mu_a,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            variance=var_f,
                            beta_l=beta_l,
                            theta=theta,
                            eta=eta,
                            z=z,
                            lhat_temp=args.lhat_temp,
                            gate_power=gate_power,
                            linear_baseline_kappa=linear_baseline_kappa,
                            pa_residual_alpha=args.pa_residual_alpha,
                            lhat_clip=args.lhat_clip,
                        )
                        logw_true = torch.full_like(lhat, float("nan"))
                        local_finite_components = None

                    if wants_finite_diagnostic:
                        if math.isclose(
                            float(rho_current), 0.0, abs_tol=1e-12
                        ):
                            logw_true_rho0 = logw_true
                            local_finite_components_rho0 = (
                                local_finite_components
                            )
                        else:
                            (
                                _,
                                _,
                                logw_true_rho0,
                                _,
                                local_finite_components_rho0,
                            ) = finite_diagnostic_at_rho(0.0)

                        if math.isclose(
                            float(rho_opt_start),
                            float(rho_current),
                            abs_tol=1e-12,
                        ):
                            logw_true_rho_start = logw_true
                            local_finite_components_rho_start = (
                                local_finite_components
                            )
                        elif math.isclose(
                            float(rho_opt_start), 0.0, abs_tol=1e-12
                        ):
                            logw_true_rho_start = logw_true_rho0
                            local_finite_components_rho_start = (
                                local_finite_components_rho0
                            )
                        else:
                            (
                                _,
                                _,
                                logw_true_rho_start,
                                _,
                                local_finite_components_rho_start,
                            ) = finite_diagnostic_at_rho(rho_opt_start)
                    else:
                        logw_true_rho0 = torch.full_like(lhat, float("nan"))
                        local_finite_components_rho0 = None
                        logw_true_rho_start = torch.full_like(
                            lhat, float("nan")
                        )
                        local_finite_components_rho_start = None

                    if bool(
                        getattr(args, "hybrid_exact_gate_reverse", False)
                    ):
                        (
                            logw_surrogate,
                            local_decomp_terms,
                        ) = hybrid_exact_gate_reverse_logw(
                            rho_current,
                            surrogate_kwargs=surrogate_kwargs,
                            lhat=lhat,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power_new=gate_power_new,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            lhat_clip=float(args.lhat_clip),
                        )
                        (
                            logw_surrogate_rho0,
                            local_decomp_terms_rho0,
                        ) = hybrid_exact_gate_reverse_logw(
                            0.0,
                            surrogate_kwargs=surrogate_kwargs,
                            lhat=lhat,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power_new=gate_power_new,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            lhat_clip=float(args.lhat_clip),
                        )
                        if math.isclose(
                            float(rho_opt_start),
                            float(rho_current),
                            abs_tol=1e-12,
                        ):
                            logw_surrogate_rho_start_unclipped = (
                                logw_surrogate
                            )
                            local_decomp_terms_rho_start = (
                                local_decomp_terms
                            )
                        elif math.isclose(
                            float(rho_opt_start), 0.0, abs_tol=1e-12
                        ):
                            logw_surrogate_rho_start_unclipped = (
                                logw_surrogate_rho0
                            )
                            local_decomp_terms_rho_start = (
                                local_decomp_terms_rho0
                            )
                        else:
                            (
                                logw_surrogate_rho_start_unclipped,
                                local_decomp_terms_rho_start,
                            ) = hybrid_exact_gate_reverse_logw(
                                rho_opt_start,
                                surrogate_kwargs=surrogate_kwargs,
                                lhat=lhat,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power_new=gate_power_new,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                                lhat_clip=float(args.lhat_clip),
                            )
                    elif coeffs is None:
                        logw_surrogate = torch.full_like(lhat, float("nan"))
                        logw_surrogate_rho0 = torch.full_like(lhat, float("nan"))
                        logw_surrogate_rho_start_unclipped = (
                            torch.full_like(lhat, float("nan"))
                        )
                    else:
                        rr = torch.tensor(rho_current, device=device, dtype=torch.float32)
                        rr_start = torch.tensor(
                            rho_opt_start,
                            device=device,
                            dtype=torch.float32,
                        )
                        if args.surrogate_gamma_delta == "full":
                            logw_surrogate = evaluate_surrogate_polynomial(coeffs, rr)
                            logw_surrogate_rho0 = coeffs[0]
                            logw_surrogate_rho_start_unclipped = (
                                evaluate_surrogate_polynomial(
                                    coeffs, rr_start
                                )
                            )
                        else:
                            a, b, cc = coeffs
                            logw_surrogate = a * rr * rr + b * rr + cc
                            logw_surrogate_rho0 = cc
                            logw_surrogate_rho_start_unclipped = (
                                a * rr_start * rr_start
                                + b * rr_start
                                + cc
                            )
                        if math.isclose(
                            float(rho_opt_start),
                            float(rho_current),
                            abs_tol=1e-12,
                        ):
                            local_decomp_terms_rho_start = (
                                local_decomp_terms
                            )
                        elif math.isclose(
                            float(rho_opt_start), 0.0, abs_tol=1e-12
                        ):
                            local_decomp_terms_rho_start = (
                                local_decomp_terms_rho0
                            )
                        else:
                            local_decomp_terms_rho_start = (
                                surrogate_decomposition_from_terms(
                                    local_surrogate_terms(
                                        rho_opt_start,
                                        **surrogate_kwargs,
                                    )
                                )
                            )

                    if wants_finite_diagnostic:
                        raw_jvp_start = (
                            None
                            if local_decomp_terms_rho_start is None
                            else local_decomp_terms_rho_start.get(
                                "proposal_hessian_jvp_raw"
                            )
                        )
                        if raw_jvp_start is not None and bool(
                            torch.isfinite(
                                raw_jvp_start.float()
                            ).any().item()
                        ):
                            raw_jvp_start_f = torch.nan_to_num(
                                raw_jvp_start.float(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                        else:
                            raw_jvp_start_f = torch.zeros_like(
                                logw_surrogate_rho_start_unclipped.float()
                            )
                        alpha_eff = float(
                            getattr(
                                args,
                                "jvp_shrink_alpha_effective",
                                1.0,
                            )
                        )
                        logw_surrogate_pretemper_rho_start = (
                            logw_surrogate_rho_start_unclipped.float()
                            + (1.0 - alpha_eff) * raw_jvp_start_f
                        )
                    else:
                        logw_surrogate_pretemper_rho_start = (
                            torch.full_like(lhat, float("nan"))
                        )

                    if bool(getattr(args, "expansion_math_diagnostics", True)):
                        local_math_components = local_expansion_math_components(
                            rho_current,
                            latents=latents,
                            mu_a=mu_a,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            variance=var_f,
                            forward_variance=forward_variance,
                            forward_displacement=forward_displacement,
                            beta_l=beta_l,
                            theta=theta,
                            eta=eta,
                            c=c,
                            eta_new=eta_new,
                            c_new=c_new,
                            lhat=lhat,
                            lhat_new=lhat_new,
                            z=z,
                            lhat_temp=args.lhat_temp,
                            gate_power=gate_power,
                            h_alpha_s=h_alpha_s,
                            finite_components=local_finite_components,
                            surrogate_decomp=local_decomp_terms,
                        )
                    else:
                        local_math_components = nan_terms(MATH_ALL_NAMES, lhat)
                timing_mark("proposal_weight")

                logw_surrogate_unclipped = logw_surrogate.float()
                logw_surrogate_rho0_unclipped = logw_surrogate_rho0.float()
                logw_surrogate, raw_clip_stats = raw_clip_surrogate_terms_global(
                    logw_surrogate=logw_surrogate.float(),
                    local_decomp_terms=local_decomp_terms,
                    args=args,
                    world_size=world_size,
                    global_start=global_start,
                    local_n=local_n,
                )
                opt.update(raw_clip_stats)

                logw_surrogate_raw = logw_surrogate.float()
                logw_surrogate_rho0_raw = logw_surrogate_rho0_unclipped.float()
                surrogate_clip = float(getattr(args, "surrogate_logw_clip", 0.0))
                if surrogate_clip > 0.0:
                    finite_raw = torch.isfinite(logw_surrogate_raw)
                    opt["surrogate_logw_clip"] = surrogate_clip
                    opt["surrogate_logw_clip_frac"] = (
                        float(
                            (
                                finite_raw
                                & (logw_surrogate_raw.abs() > surrogate_clip)
                            )
                            .float()
                            .mean()
                            .item()
                        )
                        if logw_surrogate_raw.numel() > 0
                        else float("nan")
                    )
                    opt["surrogate_logw_preclip_std"] = float(
                        logw_surrogate_raw[finite_raw].std(unbiased=False).item()
                    ) if bool(finite_raw.any().item()) else float("nan")
                    logw_surrogate = clip_final_surrogate_logw(logw_surrogate_raw, surrogate_clip)
                    logw_surrogate_rho0 = clip_final_surrogate_logw(logw_surrogate_rho0_raw, surrogate_clip)
                else:
                    opt["surrogate_logw_clip"] = 0.0
                    opt["surrogate_logw_clip_frac"] = 0.0
                    opt["surrogate_logw_preclip_std"] = float(
                        logw_surrogate_raw.float().std(unbiased=False).item()
                    )

                if args.prop_weight_mode == "surrogate":
                    if (
                        coeffs is None
                        and not bool(
                            getattr(args, "hybrid_exact_gate_reverse", False)
                        )
                        and not skipped_weight
                    ):
                        raise RuntimeError("Surrogate propagation requested but local coefficients are unavailable")
                    logw_used = torch.nan_to_num(logw_surrogate.float(), nan=0.0)
                else:
                    logw_used = torch.nan_to_num(logw_true.float(), nan=0.0)

                theta_term_exact = (
                    float(gate_power_new)
                    * log1m_theta_from_lhat(lhat_new, c_new, eta_new)
                    - float(gate_power)
                    * log1m_theta_from_lhat(lhat, c, eta)
                )
                theta_term_exact = (
                    theta_term_exact
                    + centered_lhat_increment(
                        lhat_new,
                        lhat,
                        kappa_new=linear_baseline_kappa_new,
                        kappa=linear_baseline_kappa,
                        lhat_temp=float(args.lhat_temp),
                    )
                )
                theta_term_exact = (
                    float(args.pa_residual_alpha) * theta_term_exact
                )
                timing_mark("clip_and_select")

                jvp_temper_residual = torch.zeros_like(logw_used.float())
                jvp_temper_carry = torch.zeros_like(logw_used.float())
                if bool(getattr(args, "adaptive_jvp_temper", False)) and not skipped_weight:
                    jvp_temper_term_maybe = raw_clipped_local_decomp_term(
                        local_decomp_terms=local_decomp_terms,
                        key="proposal_hessian_jvp",
                        clip=float(getattr(args, "raw_clip_jvp", 0.0)),
                        rank_topk=int(getattr(args, "rank_clip_jvp_topk", 0)),
                        center_mode=str(getattr(args, "raw_clip_center", "median")),
                        count_eps=float(getattr(args, "raw_clip_count_eps", 1e-6)),
                        world_size=world_size,
                        global_start=global_start,
                        local_n=local_n,
                    )
                    if jvp_temper_term_maybe is not None:
                        jvp_temper_term = torch.nan_to_num(
                            jvp_temper_term_maybe.float(), nan=0.0
                        )
                        logw_used_full = logw_used.float()
                        jvp_temper_base = logw_used_full - jvp_temper_term
                        jvp_temper_selection_base = logw_total.float() + jvp_temper_base
                        global_jvp_temper_selection_base = all_gather_cat(
                            jvp_temper_selection_base.contiguous(), world_size
                        )
                        global_jvp_temper_term = all_gather_cat(
                            jvp_temper_term.contiguous(), world_size
                        )
                        (
                            jvp_temper_alpha,
                            jvp_temper_base_ess,
                            jvp_temper_active_ess,
                            jvp_temper_full_ess,
                        ) = choose_gate_temper_alpha(
                            global_jvp_temper_selection_base,
                            global_jvp_temper_term,
                            float(args.jvp_temper_ess),
                        )
                        jvp_alpha_t = torch.tensor(
                            jvp_temper_alpha, device=device, dtype=torch.float32
                        )
                        logw_used = jvp_temper_base + jvp_alpha_t * jvp_temper_term
                        jvp_temper_residual = (1.0 - jvp_alpha_t) * jvp_temper_term
                        jvp_temper_carry = logw_total.float() + jvp_temper_residual
                        opt["jvp_temper_alpha"] = float(jvp_temper_alpha)
                        opt["jvp_temper_base_ess"] = float(jvp_temper_base_ess)
                        opt["jvp_temper_active_ess"] = float(jvp_temper_active_ess)
                        opt["jvp_temper_full_ess"] = float(jvp_temper_full_ess)
                        opt["jvp_temper_term_std"] = float(
                            jvp_temper_term.std(unbiased=False).item()
                        )
                        opt["jvp_temper_residual_std"] = float(
                            jvp_temper_residual.std(unbiased=False).item()
                        )
                        opt["jvp_temper_carried_std"] = float(
                            jvp_temper_carry.std(unbiased=False).item()
                        )
                    else:
                        jvp_temper_base = logw_used.float()
                        jvp_temper_term = torch.zeros_like(logw_used.float())
                        opt["jvp_temper_alpha"] = 1.0
                        opt["jvp_temper_base_ess"] = global_ess_frac_from_logw(logw_used)
                        opt["jvp_temper_active_ess"] = opt["jvp_temper_base_ess"]
                        opt["jvp_temper_full_ess"] = opt["jvp_temper_base_ess"]
                        opt["jvp_temper_term_std"] = 0.0
                        opt["jvp_temper_residual_std"] = 0.0
                        opt["jvp_temper_carried_std"] = 0.0
                else:
                    jvp_temper_base = logw_used.float()
                    jvp_temper_term = torch.zeros_like(logw_used.float())
                    opt["jvp_temper_alpha"] = 1.0
                    opt["jvp_temper_base_ess"] = float("nan")
                    opt["jvp_temper_active_ess"] = global_ess_frac_from_logw(logw_used)
                    opt["jvp_temper_full_ess"] = opt["jvp_temper_active_ess"]
                    opt["jvp_temper_term_std"] = 0.0
                    opt["jvp_temper_residual_std"] = 0.0
                    opt["jvp_temper_carried_std"] = 0.0

                gate_temper_residual = torch.zeros_like(logw_used.float())
                gate_temper_carry = torch.zeros_like(logw_used.float())
                if bool(getattr(args, "adaptive_gate_temper", False)) and not skipped_weight:
                    if (
                        str(getattr(args, "gate_temper_term", "exact")) == "surrogate"
                        and local_decomp_terms is not None
                        and "group_gate" in local_decomp_terms
                    ):
                        gate_temper_term = torch.nan_to_num(
                            local_decomp_terms["group_gate"].float(), nan=0.0
                        )
                    else:
                        gate_temper_term = theta_term_exact.float()
                    logw_used_full = logw_used.float()
                    gate_temper_base = logw_used_full - gate_temper_term
                    gate_temper_selection_base = logw_total.float() + gate_temper_base
                    global_gate_temper_selection_base = all_gather_cat(
                        gate_temper_selection_base.contiguous(), world_size
                    )
                    global_gate_temper_term = all_gather_cat(
                        gate_temper_term.contiguous(), world_size
                    )
                    (
                        gate_temper_alpha,
                        gate_temper_base_ess,
                        gate_temper_active_ess,
                        gate_temper_full_ess,
                    ) = choose_gate_temper_alpha(
                        global_gate_temper_selection_base,
                        global_gate_temper_term,
                        float(args.gate_temper_ess),
                    )
                    alpha_t = torch.tensor(
                        gate_temper_alpha, device=device, dtype=torch.float32
                    )
                    logw_used = gate_temper_base + alpha_t * gate_temper_term
                    gate_temper_residual = (1.0 - alpha_t) * gate_temper_term
                    gate_temper_carry = logw_total.float() + gate_temper_residual
                    opt["gate_temper_alpha"] = float(gate_temper_alpha)
                    opt["gate_temper_base_ess"] = float(gate_temper_base_ess)
                    opt["gate_temper_active_ess"] = float(gate_temper_active_ess)
                    opt["gate_temper_full_ess"] = float(gate_temper_full_ess)
                    opt["gate_temper_term_std"] = float(
                        gate_temper_term.std(unbiased=False).item()
                    )
                    opt["gate_temper_residual_std"] = float(
                        gate_temper_residual.std(unbiased=False).item()
                    )
                    opt["gate_temper_carried_std"] = float(
                        gate_temper_carry.std(unbiased=False).item()
                    )
                else:
                    opt["gate_temper_alpha"] = 1.0
                    opt["gate_temper_base_ess"] = float("nan")
                    opt["gate_temper_active_ess"] = global_ess_frac_from_logw(logw_used)
                    opt["gate_temper_full_ess"] = opt["gate_temper_active_ess"]
                    opt["gate_temper_term_std"] = 0.0
                    opt["gate_temper_residual_std"] = 0.0
                    opt["gate_temper_carried_std"] = 0.0

                active_logw_base = logw_total.float()
                carry_weight_temper_residual = torch.zeros_like(logw_total.float())
                if bool(getattr(args, "adaptive_carry_weight_temper", False)) and not skipped_weight:
                    carry_weight_temper_term = torch.nan_to_num(logw_total.float(), nan=0.0)
                    global_carry_weight_temper_term = all_gather_cat(
                        carry_weight_temper_term.contiguous(), world_size
                    )
                    global_carry_weight_temper_base = torch.zeros_like(global_carry_weight_temper_term)
                    (
                        carry_weight_temper_alpha,
                        carry_weight_temper_base_ess,
                        carry_weight_temper_active_ess,
                        carry_weight_temper_full_ess,
                    ) = choose_gate_temper_alpha(
                        global_carry_weight_temper_base,
                        global_carry_weight_temper_term,
                        float(args.carry_weight_temper_ess),
                    )
                    carry_weight_alpha_t = torch.tensor(
                        carry_weight_temper_alpha, device=device, dtype=torch.float32
                    )
                    active_logw_base = carry_weight_alpha_t * carry_weight_temper_term
                    carry_weight_temper_residual = (
                        (1.0 - carry_weight_alpha_t) * carry_weight_temper_term
                    )
                    opt["carry_weight_temper_alpha"] = float(carry_weight_temper_alpha)
                    opt["carry_weight_temper_base_ess"] = float(carry_weight_temper_base_ess)
                    opt["carry_weight_temper_active_ess"] = float(carry_weight_temper_active_ess)
                    opt["carry_weight_temper_full_ess"] = float(carry_weight_temper_full_ess)
                    opt["carry_weight_temper_term_std"] = float(
                        carry_weight_temper_term.std(unbiased=False).item()
                    )
                    opt["carry_weight_temper_residual_std"] = float(
                        carry_weight_temper_residual.std(unbiased=False).item()
                    )
                else:
                    opt["carry_weight_temper_alpha"] = 1.0
                    opt["carry_weight_temper_base_ess"] = float("nan")
                    opt["carry_weight_temper_active_ess"] = global_ess_frac_from_logw(
                        logw_total.float()
                    )
                    opt["carry_weight_temper_full_ess"] = opt["carry_weight_temper_active_ess"]
                    opt["carry_weight_temper_term_std"] = 0.0
                    opt["carry_weight_temper_residual_std"] = 0.0

                active_weight_temper_residual = torch.zeros_like(logw_used.float())
                active_weight_temper_carry = active_logw_base + carry_weight_temper_residual
                if bool(getattr(args, "adaptive_active_weight_temper", False)) and not skipped_weight:
                    active_weight_temper_base = active_logw_base
                    active_weight_temper_term = torch.nan_to_num(logw_used.float(), nan=0.0)
                    global_active_weight_temper_selection_base = all_gather_cat(
                        active_logw_base.contiguous(), world_size
                    )
                    global_active_weight_temper_term = all_gather_cat(
                        active_weight_temper_term.contiguous(), world_size
                    )
                    (
                        active_weight_temper_alpha,
                        active_weight_temper_base_ess,
                        active_weight_temper_active_ess,
                        active_weight_temper_full_ess,
                    ) = choose_gate_temper_alpha(
                        global_active_weight_temper_selection_base,
                        global_active_weight_temper_term,
                        float(args.active_weight_temper_ess),
                    )
                    active_weight_alpha_t = torch.tensor(
                        active_weight_temper_alpha, device=device, dtype=torch.float32
                    )
                    logw_used = active_weight_alpha_t * active_weight_temper_term
                    active_weight_temper_residual = (
                        (1.0 - active_weight_alpha_t) * active_weight_temper_term
                    )
                    active_weight_temper_carry = (
                        active_logw_base
                        + carry_weight_temper_residual
                        + active_weight_temper_residual
                    )
                    opt["active_weight_temper_alpha"] = float(active_weight_temper_alpha)
                    opt["active_weight_temper_base_ess"] = float(active_weight_temper_base_ess)
                    opt["active_weight_temper_active_ess"] = float(active_weight_temper_active_ess)
                    opt["active_weight_temper_full_ess"] = float(active_weight_temper_full_ess)
                    opt["active_weight_temper_term_std"] = float(
                        active_weight_temper_term.std(unbiased=False).item()
                    )
                    opt["active_weight_temper_residual_std"] = float(
                        active_weight_temper_residual.std(unbiased=False).item()
                    )
                    opt["active_weight_temper_carried_std"] = float(
                        active_weight_temper_carry.std(unbiased=False).item()
                    )
                else:
                    active_weight_temper_base = active_logw_base
                    active_weight_temper_term = torch.zeros_like(logw_used.float())
                    opt["active_weight_temper_alpha"] = 1.0
                    opt["active_weight_temper_base_ess"] = float("nan")
                    opt["active_weight_temper_active_ess"] = global_ess_frac_from_logw(
                        active_logw_base + logw_used.float()
                    )
                    opt["active_weight_temper_full_ess"] = opt["active_weight_temper_active_ess"]
                    opt["active_weight_temper_term_std"] = 0.0
                    opt["active_weight_temper_residual_std"] = 0.0
                    opt["active_weight_temper_carried_std"] = float(
                        active_weight_temper_carry.std(unbiased=False).item()
                    )
                timing_mark("weight_temper")

                alpha_for_diagnostics = torch.tensor(
                    float(opt.get("gate_temper_alpha", 1.0)),
                    device=device,
                    dtype=torch.float32,
                )
                if local_finite_components is not None and "theta" in local_finite_components:
                    true_gate_term = torch.nan_to_num(
                        local_finite_components["theta"].float(), nan=0.0
                    )
                    logw_true_active = logw_true.float() - (1.0 - alpha_for_diagnostics) * true_gate_term
                else:
                    logw_true_active = logw_true.float()
                if (
                    local_finite_components_rho0 is not None
                    and "theta" in local_finite_components_rho0
                ):
                    true_gate_term_rho0 = torch.nan_to_num(
                        local_finite_components_rho0["theta"].float(), nan=0.0
                    )
                    logw_true_active_rho0 = (
                        logw_true_rho0.float()
                        - (1.0 - alpha_for_diagnostics) * true_gate_term_rho0
                    )
                else:
                    logw_true_active_rho0 = logw_true_rho0.float()

                temper_residual = (
                    carry_weight_temper_residual
                    + jvp_temper_residual
                    + gate_temper_residual
                    + active_weight_temper_residual
                )
                temper_carry = active_logw_base + temper_residual
                logw_active_total = active_logw_base + logw_used
                logw_total_new = logw_active_total + temper_residual

                monitor_this_step = bool(monitor_allowed)
                needs_temper_state = (
                    (
                        bool(getattr(args, "adaptive_gate_temper", False))
                        or bool(getattr(args, "adaptive_jvp_temper", False))
                        or bool(getattr(args, "adaptive_active_weight_temper", False))
                    )
                    and not skipped_weight
                )
                global_weight_rows = all_gather_particle_rows(
                    torch.stack(
                        [
                            active_logw_base.float(),
                            logw_used.float(),
                            logw_active_total.float(),
                            logw_total_new.float(),
                        ],
                        dim=0,
                    ),
                    world_size,
                )
                (
                    global_logw_active_base,
                    global_logw_used,
                    global_logw_active_total,
                    global_logw_total_new,
                ) = global_weight_rows
                global_particle_count = int(global_logw_used.numel())
                if global_particle_count != int(args.n_particles):
                    raise RuntimeError(
                        "Global particle gather is inconsistent: "
                        f"expected {args.n_particles}, got {global_particle_count} "
                        f"(world_size={world_size}, local_n={local_n})"
                    )
                if monitor_this_step:
                    global_logw_total_prev = all_gather_cat(logw_total.float(), world_size)
                    global_logw_true = all_gather_cat(logw_true.float(), world_size)
                    global_logw_true_active = all_gather_cat(logw_true_active.float(), world_size)
                    global_logw_surrogate_unclipped = all_gather_cat(logw_surrogate_unclipped.float(), world_size)
                    global_logw_surrogate_raw = all_gather_cat(logw_surrogate_raw.float(), world_size)
                    global_logw_surrogate = all_gather_cat(logw_surrogate.float(), world_size)
                    global_gate_temper_base = all_gather_cat(
                        gate_temper_base.float().contiguous(), world_size
                    ) if bool(getattr(args, "adaptive_gate_temper", False)) and not skipped_weight else torch.full_like(global_logw_used, float("nan"))
                    global_gate_temper_term_for_row = all_gather_cat(
                        gate_temper_term.float().contiguous(), world_size
                    ) if bool(getattr(args, "adaptive_gate_temper", False)) and not skipped_weight else torch.full_like(global_logw_used, float("nan"))
                    global_gate_temper_residual = all_gather_cat(
                        gate_temper_residual.float().contiguous(), world_size
                    )
                    global_gate_temper_carry = all_gather_cat(
                        gate_temper_carry.float(), world_size
                    )
                    global_jvp_temper_base = all_gather_cat(
                        jvp_temper_base.float().contiguous(), world_size
                    ) if bool(getattr(args, "adaptive_jvp_temper", False)) and not skipped_weight else torch.full_like(global_logw_used, float("nan"))
                    global_jvp_temper_term_for_row = all_gather_cat(
                        jvp_temper_term.float().contiguous(), world_size
                    ) if bool(getattr(args, "adaptive_jvp_temper", False)) and not skipped_weight else torch.full_like(global_logw_used, float("nan"))
                    global_jvp_temper_residual = all_gather_cat(
                        jvp_temper_residual.float().contiguous(), world_size
                    )
                    global_jvp_temper_carry = all_gather_cat(
                        jvp_temper_carry.float(), world_size
                    )
                    global_active_weight_temper_base = all_gather_cat(
                        active_weight_temper_base.float().contiguous(), world_size
                    ) if (
                        (
                            bool(getattr(args, "adaptive_active_weight_temper", False))
                            or bool(getattr(args, "adaptive_carry_weight_temper", False))
                        )
                        and not skipped_weight
                    ) else torch.full_like(global_logw_used, float("nan"))
                    global_active_weight_temper_term_for_row = all_gather_cat(
                        active_weight_temper_term.float().contiguous(), world_size
                    ) if (
                        (
                            bool(getattr(args, "adaptive_active_weight_temper", False))
                            or bool(getattr(args, "adaptive_carry_weight_temper", False))
                        )
                        and not skipped_weight
                    ) else torch.full_like(global_logw_used, float("nan"))
                    global_active_weight_temper_residual = all_gather_cat(
                        active_weight_temper_residual.float().contiguous(), world_size
                    )
                    global_active_weight_temper_carry = all_gather_cat(
                        active_weight_temper_carry.float(), world_size
                    )
                    global_logw_true_rho0 = all_gather_cat(logw_true_rho0.float(), world_size)
                    global_logw_true_active_rho0 = all_gather_cat(
                        logw_true_active_rho0.float(), world_size
                    )
                    global_logw_true_rho_start = all_gather_cat(
                        logw_true_rho_start.float(), world_size
                    )
                    global_logw_surrogate_pretemper_rho_start = (
                        all_gather_cat(
                            logw_surrogate_pretemper_rho_start.float(),
                            world_size,
                        )
                    )
                    global_logw_surrogate_rho0 = all_gather_cat(logw_surrogate_rho0.float(), world_size)
                    if local_decomp_terms is None:
                        local_decomp_terms = {
                            name: torch.full_like(lhat.float(), float("nan"))
                            for name in SURROGATE_DECOMP_NAMES
                        }
                    if local_decomp_terms_rho0 is None:
                        local_decomp_terms_rho0 = {
                            name: torch.full_like(lhat.float(), float("nan"))
                            for name in SURROGATE_DECOMP_NAMES
                        }
                    global_decomp_terms = {
                        name: all_gather_cat(value.float(), world_size)
                        for name, value in local_decomp_terms.items()
                    }
                    global_decomp_terms_rho0 = {
                        name: all_gather_cat(value.float(), world_size)
                        for name, value in local_decomp_terms_rho0.items()
                    }
                    if local_finite_components is None:
                        local_finite_components = nan_terms(FINITE_DECOMP_NAMES, lhat)
                    if local_finite_components_rho0 is None:
                        local_finite_components_rho0 = nan_terms(FINITE_DECOMP_NAMES, lhat)
                    if local_math_components is None:
                        local_math_components = nan_terms(MATH_ALL_NAMES, lhat)
                    global_finite_components = {
                        name: all_gather_cat(value.float(), world_size)
                        for name, value in local_finite_components.items()
                    }
                    global_finite_components_rho0 = {
                        name: all_gather_cat(value.float(), world_size)
                        for name, value in local_finite_components_rho0.items()
                    }
                    global_math_components = {
                        name: all_gather_cat(value.float(), world_size)
                        for name, value in local_math_components.items()
                    }
                    global_theta = all_gather_cat(theta.float(), world_size)
                    global_lhat = all_gather_cat(lhat.float(), world_size)
                    global_lhat_new = all_gather_cat(lhat_new.float(), world_size)
                    global_theta_term_exact = all_gather_cat(theta_term_exact.float(), world_size)
                    global_kernel_lr = all_gather_cat(kernel_lr.float(), world_size)
                    global_beta = all_gather_cat(beta_l.float(), world_size)
                    global_origin_ids = all_gather_cat(origin_ids.contiguous(), world_size)

                    base_drift = mu_a.float() - latents.float()
                    tempered_gate = (
                        float(gate_power)
                        * theta.float()[:, None, None, None]
                        * float(eta)
                        * float(args.lhat_temp)
                    )
                    residual_gate = (
                        tempered_gate - float(linear_baseline_kappa)
                    )
                    guidance_rho0 = -var_f * residual_gate * beta_l.float()
                    guidance_rho_correction = (
                        var_f
                        * float(rho_current)
                        * residual_gate
                        * beta_l.float()
                    )
                    guidance_proposal = guidance_rho0 + guidance_rho_correction
                    brownian_drift = z.float()
                    deterministic_drift = base_drift + guidance_proposal
                    sample_drift = deterministic_drift + brownian_drift
                    global_drift_components = {
                        "base": all_gather_cat(base_drift.contiguous(), world_size),
                        "guidance_rho0": all_gather_cat(guidance_rho0.contiguous(), world_size),
                        "guidance_rho_correction": all_gather_cat(
                            guidance_rho_correction.contiguous(), world_size
                        ),
                        "guidance_proposal": all_gather_cat(guidance_proposal.contiguous(), world_size),
                        "brownian": all_gather_cat(brownian_drift.contiguous(), world_size),
                        "deterministic_total": all_gather_cat(deterministic_drift.contiguous(), world_size),
                        "sample_total": all_gather_cat(sample_drift.contiguous(), world_size),
                    }

                if monitor_this_step or needs_temper_state:
                    global_temper_residual = all_gather_cat(
                        temper_residual.float().contiguous(), world_size
                    )
                    global_temper_carry = all_gather_cat(
                        temper_carry.float().contiguous(), world_size
                    )
                else:
                    global_temper_residual = torch.zeros_like(global_logw_used)
                    global_temper_carry = torch.zeros_like(global_logw_used)

                global_w_active_total = global_normalized_weights_from_logw(global_logw_active_total)
                cum_ess = float(1.0 / (args.n_particles * torch.sum(global_w_active_total * global_w_active_total)).item())
                full_cum_ess = global_ess_frac_from_logw(global_logw_total_new)
                inc_ess = global_ess_frac_from_logw(global_logw_used)
                cess_ess = conditional_ess_frac_from_logw(global_logw_active_base, global_logw_used)
                opt["pred_noise_guidance_factor"] = predicted_noise_guidance_factor(
                    pipe.scheduler,
                    t_int,
                )
                resample_weight_mode = str(getattr(args, "resample_weight_mode", "cum"))
                if resample_weight_mode == "cum":
                    global_resample_w = global_w_active_total
                    resample_weight_ess = cum_ess
                elif resample_weight_mode == "inc":
                    global_resample_w = global_normalized_weights_from_logw(global_logw_used)
                    resample_weight_ess = inc_ess
                else:
                    raise ValueError(f"Unknown --resample-weight-mode={resample_weight_mode!r}")
                if int(global_resample_w.numel()) != int(args.n_particles):
                    raise RuntimeError(
                        "Resampling weights must cover the complete global population: "
                        f"expected {args.n_particles}, got {global_resample_w.numel()}"
                    )
                (
                    do_resample,
                    resample_ess_used,
                    resample_allowed,
                    resample_armed,
                    resample_rearmed,
                    resample_block_reason,
                ) = should_resample_from_ess(
                    args,
                    step=step,
                    n_steps=len(timesteps),
                    cum_ess=cum_ess,
                    inc_ess=inc_ess,
                    cess_ess=cess_ess,
                    resample_armed=resample_armed,
                )
                opt["resample_hysteresis_armed"] = float(resample_armed)
                opt["resample_hysteresis_rearmed"] = float(resample_rearmed)
                opt["resample_hysteresis_blocked"] = float(
                    resample_block_reason == "hysteresis"
                )

                ancestor_idx = None
                if do_resample:
                    if rank == 0:
                        ancestor_idx = systematic_resample(global_resample_w, resample_gen).long()
                    else:
                        ancestor_idx = torch.empty(args.n_particles, device=device, dtype=torch.long)
                    if int(ancestor_idx.numel()) != int(args.n_particles):
                        raise RuntimeError(
                            "Global resampling must return one ancestor per global particle: "
                            f"expected {args.n_particles}, got {ancestor_idx.numel()}"
                        )
                    if not monitor_this_step:
                        global_origin_ids = all_gather_cat(origin_ids.contiguous(), world_size)

                global_logw_total_after_resample = None
                if (
                    do_resample
                    and (
                        bool(getattr(args, "adaptive_gate_temper", False))
                        or bool(getattr(args, "adaptive_jvp_temper", False))
                        or bool(getattr(args, "adaptive_active_weight_temper", False))
                    )
                    and rank == 0
                ):
                    if resample_weight_mode == "cum":
                        global_logw_total_after_resample = global_temper_residual[
                            ancestor_idx
                        ].float()
                    else:
                        global_logw_total_after_resample = global_temper_carry[
                            ancestor_idx
                        ].float()
                timing_mark("distributed_metrics")
                if timing_enabled:
                    timing_values["timing_pre_row_total_ms"] = float(
                        (timing_now(device, True) - timing_total_start) * 1000.0
                    )
                    opt.update(timing_values)

                if rank == 0 and monitor_allowed:
                    global_origin_ids_after_step = (
                        global_origin_ids[ancestor_idx]
                        if do_resample and ancestor_idx is not None
                        else global_origin_ids
                    )
                    row = make_row(
                        args=args,
                        step=step,
                        timestep=t_int,
                        c=c,
                        eta=eta,
                        gate_power=gate_power,
                        gate_power_new=gate_power_new,
                        linear_baseline_kappa=linear_baseline_kappa,
                        linear_baseline_kappa_new=(
                            linear_baseline_kappa_new
                        ),
                        rho=rho_current,
                        theta=global_theta,
                        lhat=global_lhat,
                        lhat_new=global_lhat_new,
                        beta_l=global_beta,
                        logw_used=global_logw_used,
                        logw_true=global_logw_true,
                        logw_true_active=global_logw_true_active,
                        logw_surrogate_raw=global_logw_surrogate_raw,
                        logw_surrogate=global_logw_surrogate,
                        logw_surrogate_unclipped=global_logw_surrogate_unclipped,
                        logw_active_total=global_logw_active_total,
                        logw_total_new=global_logw_total_new,
                        logw_true_rho0=global_logw_true_rho0,
                        logw_true_active_rho0=global_logw_true_active_rho0,
                        logw_surrogate_rho0=global_logw_surrogate_rho0,
                        logw_total_prev=global_logw_total_prev,
                        surrogate_decomp=global_decomp_terms,
                        surrogate_decomp_rho0=global_decomp_terms_rho0,
                        finite_components=global_finite_components,
                        finite_components_rho0=global_finite_components_rho0,
                        math_components=global_math_components,
                        theta_term_exact=global_theta_term_exact,
                        kernel_lr=global_kernel_lr,
                        drift_components=global_drift_components,
                        opt=opt,
                        do_resample=do_resample,
                        cess_ess=cess_ess,
                        resample_ess_used=resample_ess_used,
                        resample_weight_ess=resample_weight_ess,
                        resample_allowed=resample_allowed,
                        skipped_weight=skipped_weight,
                        ancestor_idx=ancestor_idx,
                        logw_total_after_resample=global_logw_total_after_resample,
                        gate_temper_base=global_gate_temper_base,
                        gate_temper_term=global_gate_temper_term_for_row,
                        gate_temper_residual=global_gate_temper_residual,
                        gate_temper_carry=global_gate_temper_carry,
                        jvp_temper_base=global_jvp_temper_base,
                        jvp_temper_term=global_jvp_temper_term_for_row,
                        jvp_temper_residual=global_jvp_temper_residual,
                        jvp_temper_carry=global_jvp_temper_carry,
                        active_weight_temper_base=global_active_weight_temper_base,
                        active_weight_temper_term=global_active_weight_temper_term_for_row,
                        active_weight_temper_residual=global_active_weight_temper_residual,
                        active_weight_temper_carry=global_active_weight_temper_carry,
                        origin_ids=global_origin_ids_after_step,
                    )
                    row["jvp_step_window_active"] = int(
                        bool(args.surrogate_gamma_jvp)
                        and (
                            jvp_window_active
                            if needs_surrogate
                            else False
                        )
                    )
                    row["no_jvp_first_steps"] = int(
                        getattr(args, "no_jvp_first_steps", 0)
                    )
                    row["no_jvp_last_steps"] = int(
                        getattr(args, "no_jvp_last_steps", 0)
                    )
                    preopt_prefix = (
                        "surrogate_pretemper_preopt_start_true"
                    )
                    row["rho_preopt_start"] = float(rho_opt_start)
                    row[
                        "std_logw_surrogate_pretemper_preopt_start"
                    ] = float(
                        global_logw_surrogate_pretemper_rho_start.float()
                        .std(unbiased=False)
                        .item()
                    )
                    preopt_true_std = float(
                        global_logw_true_rho_start.float()
                        .std(unbiased=False)
                        .item()
                    )
                    row["std_logw_true_preopt_start"] = preopt_true_std
                    row[f"corr_{preopt_prefix}"] = finite_corr(
                        global_logw_surrogate_pretemper_rho_start,
                        global_logw_true_rho_start,
                    )
                    row.update(
                        comparison_stats(
                            preopt_prefix,
                            global_logw_surrogate_pretemper_rho_start,
                            global_logw_true_rho_start,
                        )
                    )
                    preopt_resid_std = row[
                        f"{preopt_prefix}_resid_std"
                    ]
                    row[
                        f"{preopt_prefix}_centered_relative_rmse"
                    ] = (
                        float(preopt_resid_std / preopt_true_std)
                        if math.isfinite(preopt_resid_std)
                        and math.isfinite(preopt_true_std)
                        and preopt_true_std > 1e-30
                        else float("nan")
                    )
                    rows.append(row)
                    if step % max(args.log_every, 1) == 0 or step == len(timesteps) - 1:
                        print(
                            f"[step {step:03d}/{len(timesteps)} t={t_int:4d}] "
                            f"rho={rho_current:+.3f} gate={row['guidance_mean']:.4g} "
                            f"prop_gate={row['proposal_guidance_mean']:.4g} "
                            f"eps_gate={row['proposal_pred_noise_guidance_mean']:.4g} "
                            f"raw_eps_gate={row['proposal_raw_pred_noise_guidance_mean']:.4g} "
                            f"incESS={inc_ess:.3f} cessESS={cess_ess:.3f} "
                            f"activeCumESS={cum_ess:.3f} fullCumESS={full_cum_ess:.3f} "
                            f"trueESS={row['cum_ess_true_frac']:.3f} "
                            f"trueActESS={row['cum_ess_true_active_frac']:.3f} "
                            f"trueESS0={row['cum_ess_true_rho0_frac']:.3f} "
                            f"dTrueESS={row['cum_ess_true_gain_vs_rho0']:+.3f} "
                            f"dActTrueESS={row['cum_ess_true_active_gain_vs_rho0']:+.3f} "
                            f"trueCESS={row['cess_ess_true_frac']:.3f} "
                            f"incTrueESS={row['inc_ess_true_frac']:.3f} "
                            f"used_std={row['std_logw_used']:.3g} "
                            f"true_std={row['std_logw_true']:.3g} "
                            f"surr_std={row['std_logw_surrogate']:.3g} "
                            f"corr={row['corr_surrogate_true']:.3g} "
                            f"jvp_alpha={row.get('jvp_temper_alpha', float('nan')):.3g} "
                            f"gate_alpha={row.get('gate_temper_alpha', float('nan')):.3g} "
                            f"old_alpha={row.get('carry_weight_temper_alpha', float('nan')):.3g} "
                            f"full_alpha={row.get('active_weight_temper_alpha', float('nan')):.3g} "
                            f"carryESS={global_ess_frac_from_logw(global_temper_carry):.3f} "
                            f"roots={row.get('unique_initial_ancestors', 'nan')} "
                            f"clip={row.get('surrogate_logw_clip_frac', float('nan')):.2f} "
                            f"jvp={jvp_used} "
                            f"resample={int(do_resample)}({args.resample_ess_mode}:{resample_ess_used:.3f},"
                            f"w={resample_weight_mode}:{resample_weight_ess:.3f}"
                            f"{'' if not resample_block_reason else f',blocked-{resample_block_reason}'})",
                            flush=True,
                        )

                if do_resample:
                    dist.broadcast(ancestor_idx, src=0)
                    early_stop_packet = torch.zeros(
                        2, device=device, dtype=torch.int64
                    )
                    if rank == 0:
                        roots_after = genealogy_stats(
                            global_origin_ids[ancestor_idx],
                            int(args.n_particles),
                        )
                        observed_unique = int(
                            roots_after["unique_initial_ancestors"]
                        )
                        early_stop_packet[0] = int(observed_unique)
                        early_stop_packet[1] = int(
                            int(args.early_stop_min_unique_roots) > 0
                            and observed_unique
                            < int(args.early_stop_min_unique_roots)
                        )
                        early_stop_effective_roots = float(
                            roots_after["initial_ancestor_effective_roots"]
                        )
                    dist.broadcast(early_stop_packet, src=0)
                    early_stop_unique_roots = int(early_stop_packet[0].item())
                    early_stopped = bool(int(early_stop_packet[1].item()))
                    if early_stopped:
                        early_stop_step = int(step)
                    all_x_new = all_gather_cat(x_new.contiguous(), world_size)
                    all_lhat_new = all_gather_cat(lhat_new.float(), world_size)
                    all_origin_ids = global_origin_ids
                    local_idx = ancestor_idx[global_start : global_start + local_n]
                    latents = all_x_new[local_idx].to(dtype=latents.dtype)
                    lhat = all_lhat_new[local_idx]
                    origin_ids = all_origin_ids[local_idx].to(device=device, dtype=torch.long)
                    if (
                        bool(getattr(args, "adaptive_gate_temper", False))
                        or bool(getattr(args, "adaptive_jvp_temper", False))
                        or bool(getattr(args, "adaptive_active_weight_temper", False))
                    ):
                        if resample_weight_mode == "cum":
                            logw_total = global_temper_residual[local_idx].to(
                                device=device, dtype=torch.float32
                            )
                        else:
                            logw_total = global_temper_carry[local_idx].to(
                                device=device, dtype=torch.float32
                            )
                    else:
                        logw_total = torch.zeros_like(logw_total)
                else:
                    latents = x_new.to(dtype=latents.dtype)
                    lhat = lhat_new
                    logw_total = logw_total_new
                timing_mark("state_update")
                if timing_enabled:
                    timing_values["timing_step_total_ms"] = float(
                        (timing_now(device, True) - timing_total_start) * 1000.0
                    )
                    local_phase_times = torch.tensor(
                        [
                            float(timing_values.get(f"timing_{phase}_ms", 0.0))
                            for phase in TIMING_PHASES
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(local_phase_times, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        record = {
                            phase: float(local_phase_times[index].item())
                            for index, phase in enumerate(TIMING_PHASES)
                        }
                        record["rho_search_active"] = float(needs_opt)
                        record["jvp_step_window_active"] = float(
                            needs_surrogate
                            and jvp_step_window_active(
                                args,
                                step,
                                len(timesteps),
                            )
                            and bool(args.surrogate_gamma_jvp)
                        )
                        timing_records.append(record)
                if early_stopped:
                    if rank == 0:
                        print(
                            "[early-stop] infeasible genealogy: "
                            f"unique_roots={early_stop_unique_roots} < "
                            f"{int(args.early_stop_min_unique_roots)} "
                            f"at step={early_stop_step}",
                            flush=True,
                        )
                    break

        timing_summary: dict[str, Any] = {}
        if rank == 0 and timing_records:
            timing_summary = summarize_timing_records(timing_records)
            timing_path = args.out_dir / "timing_summary.json"
            with timing_path.open("w") as f:
                json.dump(timing_summary, f, indent=2)
            phase_means = timing_summary.get("phase_mean_ms", {})
            print(
                "[timing average ms] "
                + " ".join(
                    f"{phase}={float(phase_means.get(phase, float('nan'))):.2f}"
                    for phase in TIMING_PHASES[:9]
                ),
                flush=True,
            )
            print(
                f"[timing rho] active_steps={timing_summary['rho_search_active_steps']} "
                f"active_mean_ms={timing_summary['rho_search_active_mean_ms']:.2f} "
                f"(nested inside surrogate_rho)",
                flush=True,
            )
            print(f"[saved] {timing_path}", flush=True)

        if early_stopped:
            if rank == 0:
                elapsed = time.time() - start_time
                rows_path = args.out_dir / "smc_pa_gate_rho_monitor.csv"
                if rows:
                    write_rows(rows_path, rows)
                marker = {
                    "status": "early_stopped_infeasible_roots",
                    "early_stopped": True,
                    "reason": "unique_initial_ancestors_below_required_minimum",
                    "requested_steps": len(timesteps),
                    "completed_steps": early_stop_step + 1,
                    "stop_step": early_stop_step,
                    "minimum_required_unique_roots": int(
                        args.early_stop_min_unique_roots
                    ),
                    "observed_unique_roots": int(early_stop_unique_roots),
                    "observed_effective_roots": float(
                        early_stop_effective_roots
                    ),
                    "elapsed_sec": elapsed,
                }
                marker_tmp = args.out_dir / "early_stop.json.tmp"
                marker_path = args.out_dir / "early_stop.json"
                with marker_tmp.open("w") as f:
                    json.dump(marker, f, indent=2)
                marker_tmp.replace(marker_path)
                with (args.out_dir / "summary.json").open("w") as f:
                    json.dump(
                        {
                            **marker,
                            "n_steps": len(timesteps),
                            "n_particles": args.n_particles,
                            "world_size": world_size,
                            "local_particles": local_n,
                            "resample_ess_mode": args.resample_ess_mode,
                            "resample_weight_mode": args.resample_weight_mode,
                            "linear_baseline_kappa": (
                                mean_linear_baseline_kappa_path
                            ),
                            "legacy_linear_baseline_kappa": float(
                                args.linear_baseline_kappa
                            ),
                            "kappa_start": float(args.kappa_start),
                            "kappa_end": float(args.kappa_end),
                            "kappa_schedule": str(args.kappa_schedule),
                            "kappa_end_frac": float(args.kappa_end_frac),
                            "kappa_gamma": float(args.kappa_gamma),
                            "final_unique_initial_ancestors": int(
                                early_stop_unique_roots
                            ),
                            "min_unique_initial_ancestors": int(
                                early_stop_unique_roots
                            ),
                            "final_initial_ancestor_effective_roots": float(
                                early_stop_effective_roots
                            ),
                            "min_initial_ancestor_effective_roots": float(
                                early_stop_effective_roots
                            ),
                            "timing": timing_summary,
                        },
                        f,
                        indent=2,
                    )
                print(f"[saved] {marker_path}", flush=True)
            dist.barrier()
            return

        if args.save_images:
            print(f"[decode rank {rank}] saving {local_n} final particle images", flush=True)
            decode_and_save_images(
                pipe,
                latents,
                args,
                start_index=global_start,
                make_grid_after=False,
            )
            write_decode_done_marker(args.out_dir, rank)

        if rank == 0:
            elapsed = time.time() - start_time
            rows_path = args.out_dir / "smc_pa_gate_rho_monitor.csv"
            if not rows:
                summary = {
                    "elapsed_sec": elapsed,
                    "n_steps": len(timesteps),
                    "monitored_steps": 0,
                    "last_monitored_step": -1,
                    "n_particles": args.n_particles,
                    "world_size": world_size,
                    "local_particles": local_n,
                    "prop_weight_mode": args.prop_weight_mode,
                    "resample_ess_mode": args.resample_ess_mode,
                    "resample_weight_mode": args.resample_weight_mode,
                    "no_resample_last_steps": int(args.no_resample_last_steps),
                    "no_rho_last_steps": int(args.no_rho_last_steps),
                    "no_jvp_first_steps": int(args.no_jvp_first_steps),
                    "no_jvp_last_steps": int(args.no_jvp_last_steps),
                    "no_monitor_last_steps": int(args.no_monitor_last_steps),
                    "rho_objective": args.rho_objective,
                    "rho_ess_search_mode": args.rho_ess_search_mode,
                    "rho_ess_coarse_candidates": int(args.rho_ess_coarse_candidates),
                    "cfg_scale": float(args.cfg_scale),
                    "base_lambda_neg": float(args.base_lambda_neg),
                    "linear_baseline_kappa": (
                        mean_linear_baseline_kappa_path
                    ),
                    "legacy_linear_baseline_kappa": float(
                        args.linear_baseline_kappa
                    ),
                    "kappa_start": float(args.kappa_start),
                    "kappa_end": float(args.kappa_end),
                    "kappa_schedule": str(args.kappa_schedule),
                    "kappa_end_frac": float(args.kappa_end_frac),
                    "kappa_gamma": float(args.kappa_gamma),
                    "theta_mid_logit_shift": float(args.theta_mid_logit_shift),
                    "gate_ratio_mode": str(args.gate_ratio_mode),
                    "surrogate_gamma_delta": args.surrogate_gamma_delta,
                    "surrogate_logw_clip": float(args.surrogate_logw_clip),
                    "raw_clip_jvp": float(args.raw_clip_jvp),
                    "rank_clip_jvp_topk": int(args.rank_clip_jvp_topk),
                    "adaptive_rank_clip_jvp": bool(args.adaptive_rank_clip_jvp),
                    "adaptive_rank_clip_jvp_ess_thresholds": str(args.adaptive_rank_clip_jvp_ess_thresholds),
                    "adaptive_rank_clip_jvp_topks": str(args.adaptive_rank_clip_jvp_topks),
                    "raw_clip_count_eps": float(args.raw_clip_count_eps),
                    "raw_clip_jvp_in_rho_objective": bool(args.raw_clip_jvp_in_rho_objective),
                    "raw_clip_reverse_kernel": float(args.raw_clip_reverse_kernel),
                    "raw_clip_center": str(args.raw_clip_center),
                    "rho_update": bool(args.rho_update),
                    "timing": timing_summary,
                    "monitor_output_disabled": True,
                }
                with (args.out_dir / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=2)
                if args.save_images:
                    print("[decode] waiting for per-rank image saves", flush=True)
                    wait_for_decode_markers(args.out_dir, world_size, args.image_marker_timeout)
                    print("[decode] assembling particle grid on rank 0", flush=True)
                    assemble_grid_from_saved_images(args, args.n_particles)
                print(f"[done] {elapsed:.1f}s", flush=True)
                print("[saved] monitor output disabled; wrote summary.json only", flush=True)
                return
            write_rows(rows_path, rows)
            make_rho_monitor_plot(
                rows,
                args.out_dir,
                args.resample_ess,
                no_resample_last_steps=args.no_resample_last_steps,
                mask_final_inc_ess=args.plot_mask_final_inc_ess,
            )
            make_genealogy_plot(rows, args.out_dir)
            make_gate_schedule_diagnostic_plot(rows, args.out_dir)
            make_rho_surrogate_optimization_plot(rows, args.out_dir)
            make_rho_inc_ess_optimization_plot(rows, args.out_dir)
            make_gate_temper_plot(rows, args.out_dir)
            make_jvp_temper_plot(rows, args.out_dir)
            make_active_weight_temper_plot(rows, args.out_dir)
            make_beta_variance_plot(rows, args.out_dir)
            make_drift_magnitude_plot(rows, args.out_dir)
            make_decomposition_plot(rows, args.out_dir)
            make_rho0_decomposition_plot(rows, args.out_dir)
            make_jvp_diagnostic_plot(rows, args.out_dir)
            make_kernel_tail_diagnostic_plot(rows, args.out_dir)
            make_raw_clip_diagnostic_plot(rows, args.out_dir)
            make_surrogate_agreement_plot(rows, args.out_dir)
            make_inc_ess_agreement_plot(rows, args.out_dir)
            make_math_diagnostic_plot(rows, args.out_dir)

            def row_nanmean(key: str) -> float:
                vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
                finite = np.isfinite(vals)
                return float(np.mean(vals[finite])) if bool(finite.any()) else float("nan")

            def row_nanmax(key: str) -> float:
                vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
                finite = np.isfinite(vals)
                return float(np.max(vals[finite])) if bool(finite.any()) else float("nan")

            def row_nanmin(key: str) -> float:
                vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
                finite = np.isfinite(vals)
                return float(np.min(vals[finite])) if bool(finite.any()) else float("nan")

            summary = {
                "elapsed_sec": elapsed,
                "n_steps": len(timesteps),
                "monitored_steps": len(rows),
                "last_monitored_step": int(rows[-1]["step"]) if rows else -1,
                "n_particles": args.n_particles,
                "world_size": world_size,
                "local_particles": local_n,
                "resamples": int(sum(r["resampled"] for r in rows)),
                "final_unique_initial_ancestors": int(rows[-1].get("unique_initial_ancestors", 0)),
                "min_unique_initial_ancestors": row_nanmin("unique_initial_ancestors"),
                "final_initial_ancestor_effective_roots": float(
                    rows[-1].get("initial_ancestor_effective_roots", float("nan"))
                ),
                "min_initial_ancestor_effective_roots": row_nanmin("initial_ancestor_effective_roots"),
                "final_max_initial_ancestor_copies": int(rows[-1].get("max_initial_ancestor_copies", 0)),
                "mean_rho": float(np.mean([r["rho_used"] for r in rows])),
                "prop_weight_mode": args.prop_weight_mode,
                "resample_ess_mode": args.resample_ess_mode,
                "resample_weight_mode": args.resample_weight_mode,
                "resample_ess_rearm": float(args.resample_ess_rearm),
                "no_resample_last_steps": int(args.no_resample_last_steps),
                "no_rho_last_steps": int(args.no_rho_last_steps),
                "no_jvp_first_steps": int(args.no_jvp_first_steps),
                "no_jvp_last_steps": int(args.no_jvp_last_steps),
                "no_monitor_last_steps": int(args.no_monitor_last_steps),
                "rho_objective": args.rho_objective,
                "rho_ess_search_mode": args.rho_ess_search_mode,
                "rho_ess_coarse_candidates": int(args.rho_ess_coarse_candidates),
                "rho_update": bool(args.rho_update),
                "timing": timing_summary,
                "cfg_scale": float(args.cfg_scale),
                "base_lambda_neg": float(args.base_lambda_neg),
                "linear_baseline_kappa": (
                    mean_linear_baseline_kappa_path
                ),
                "legacy_linear_baseline_kappa": float(
                    args.linear_baseline_kappa
                ),
                "kappa_start": float(args.kappa_start),
                "kappa_end": float(args.kappa_end),
                "kappa_schedule": str(args.kappa_schedule),
                "kappa_end_frac": float(args.kappa_end_frac),
                "kappa_gamma": float(args.kappa_gamma),
                "mean_linear_baseline_kappa": row_nanmean(
                    "linear_baseline_kappa"
                ),
                "theta_mid_logit_shift": float(args.theta_mid_logit_shift),
                "gate_ratio_mode": str(args.gate_ratio_mode),
                "mean_true_var": float(np.nanmean([r["true_var_rho"] for r in rows])),
                "mean_cum_true_var": float(np.nanmean([r["cum_true_var_rho"] for r in rows])),
                "mean_cum_true_var_rho0": float(np.nanmean([r["cum_true_var_rho0"] for r in rows])),
                "mean_cum_true_var_improvement_vs_rho0": float(
                    np.nanmean([r["cum_true_var_improvement_vs_rho0"] for r in rows])
                ),
                "mean_used_var": float(np.nanmean([r["used_var_rho"] for r in rows])),
                "mean_surrogate_var": float(np.nanmean([r["surrogate_var_rho"] for r in rows])),
                "mean_surrogate_unclipped_var": row_nanmean("surrogate_unclipped_var_rho"),
                "mean_surrogate_no_jvp_var": row_nanmean("surrogate_no_jvp_var_rho"),
                "mean_surrogate_no_jvp_var_ratio": row_nanmean("surrogate_no_jvp_var_ratio"),
                "mean_surrogate_jvp_var_delta": row_nanmean("surrogate_jvp_var_delta"),
                "mean_surrogate_true_corr": float(np.nanmean([r["corr_surrogate_true"] for r in rows])),
                "mean_surrogate_unclipped_true_corr": row_nanmean("corr_surrogate_unclipped_true"),
                "mean_surrogate_no_jvp_true_corr": row_nanmean("corr_surrogate_no_jvp_true"),
                "min_surrogate_true_corr": row_nanmin("corr_surrogate_true"),
                "mean_cum_true_ess": float(np.nanmean([r["cum_ess_true_frac"] for r in rows])),
                "mean_cum_true_ess_rho0": float(np.nanmean([r["cum_ess_true_rho0_frac"] for r in rows])),
                "mean_cum_true_ess_gain_vs_rho0": float(
                    np.nanmean([r["cum_ess_true_gain_vs_rho0"] for r in rows])
                ),
                "mean_true_active_total_ess": row_nanmean("true_active_total_ess_frac"),
                "mean_surrogate_total_ess": row_nanmean("surrogate_total_ess_frac"),
                "mean_surrogate_total_ess_rho0": row_nanmean("surrogate_total_ess_rho0_frac"),
                "mean_surrogate_active_total_ess": row_nanmean("surrogate_active_total_ess_frac"),
                "mean_active_cum_ess": row_nanmean("cum_ess_frac"),
                "mean_full_cum_ess": row_nanmean("full_cum_ess_frac"),
                "mean_rho_total_ess_start": row_nanmean("rho_total_ess_start"),
                "mean_rho_total_ess_final": row_nanmean("rho_total_ess_final"),
                "mean_rho_total_ess_rho0": row_nanmean("rho_total_ess_rho0"),
                "mean_rho_inc_ess_start": row_nanmean("rho_inc_ess_start"),
                "mean_rho_inc_ess_final": row_nanmean("rho_inc_ess_final"),
                "mean_rho_inc_ess_rho0": row_nanmean("rho_inc_ess_rho0"),
                "rho_optimizer": str(args.rho_optimizer),
                "mean_rho_optimizer_iterations": row_nanmean(
                    "rho_optimizer_iterations"
                ),
                "mean_rho_objective_evaluations": row_nanmean(
                    "rho_objective_evaluations"
                ),
                "total_rho_objective_evaluations": float(
                    np.nansum(
                        [
                            float(r.get("rho_objective_evaluations", np.nan))
                            for r in rows
                        ]
                    )
                ),
                "mean_cess_true_ess": float(np.nanmean([r["cess_ess_true_frac"] for r in rows])),
                "mean_cess_true_ess_rho0": float(np.nanmean([r["cess_ess_true_rho0_frac"] for r in rows])),
                "mean_cess_true_ess_gain_vs_rho0": float(
                    np.nanmean([r["cess_ess_true_gain_vs_rho0"] for r in rows])
                ),
                "mean_target_guidance": row_nanmean("guidance_mean"),
                "mean_proposal_guidance": row_nanmean("proposal_guidance_mean"),
                "mean_pred_noise_guidance_factor": row_nanmean("pred_noise_guidance_factor"),
                "mean_target_pred_noise_guidance": row_nanmean("target_pred_noise_guidance_mean"),
                "mean_proposal_pred_noise_guidance": row_nanmean("proposal_pred_noise_guidance_mean"),
                "max_proposal_pred_noise_guidance": row_nanmax("proposal_pred_noise_guidance_mean"),
                "mean_proposal_raw_pred_noise_guidance": row_nanmean(
                    "proposal_raw_pred_noise_guidance_mean"
                ),
                "max_proposal_raw_pred_noise_guidance": row_nanmax(
                    "proposal_raw_pred_noise_guidance_mean"
                ),
                "surrogate_gamma_delta": args.surrogate_gamma_delta,
                "surrogate_logw_clip": float(args.surrogate_logw_clip),
                "raw_clip_jvp": float(args.raw_clip_jvp),
                "rank_clip_jvp_topk": int(args.rank_clip_jvp_topk),
                "adaptive_rank_clip_jvp": bool(args.adaptive_rank_clip_jvp),
                "adaptive_rank_clip_jvp_ess_thresholds": str(args.adaptive_rank_clip_jvp_ess_thresholds),
                "adaptive_rank_clip_jvp_topks": str(args.adaptive_rank_clip_jvp_topks),
                "mean_rank_clip_jvp_effective_topk": row_nanmean("rank_clip_jvp_effective_topk"),
                "max_rank_clip_jvp_effective_topk": row_nanmax("rank_clip_jvp_effective_topk"),
                "raw_clip_count_eps": float(args.raw_clip_count_eps),
                "raw_clip_jvp_in_rho_objective": bool(args.raw_clip_jvp_in_rho_objective),
                "raw_clip_reverse_kernel": float(args.raw_clip_reverse_kernel),
                "raw_clip_center": str(args.raw_clip_center),
                "mean_gate_temper_alpha": float(np.nanmean([r["gate_temper_alpha"] for r in rows])),
                "min_gate_temper_alpha": float(np.nanmin([r["gate_temper_alpha"] for r in rows])),
                "alpha_lt_one_frac": float(np.nanmean([float(r["gate_temper_alpha"] < 0.999999) for r in rows])),
                "mean_gate_temper_carry_ess": float(np.nanmean([r["gate_temper_carry_ess"] for r in rows])),
                "min_gate_temper_carry_ess": float(np.nanmin([r["gate_temper_carry_ess"] for r in rows])),
                "final_gate_temper_residual_abs_mean": float(rows[-1]["gate_temper_residual_abs_mean"]),
                "final_gate_temper_residual_abs_max": float(rows[-1]["gate_temper_residual_abs_max"]),
                "adaptive_jvp_temper": bool(args.adaptive_jvp_temper),
                "jvp_temper_ess": float(args.jvp_temper_ess),
                "mean_jvp_temper_alpha": row_nanmean("jvp_temper_alpha"),
                "min_jvp_temper_alpha": row_nanmin("jvp_temper_alpha"),
                "jvp_alpha_lt_one_frac": float(
                    np.nanmean([float(r["jvp_temper_alpha"] < 0.999999) for r in rows])
                ),
                "mean_jvp_temper_carry_ess": row_nanmean("jvp_temper_carry_ess"),
                "min_jvp_temper_carry_ess": row_nanmin("jvp_temper_carry_ess"),
                "mean_jvp_temper_residual_std": row_nanmean("jvp_temper_residual_std"),
                "final_jvp_temper_residual_abs_mean": float(rows[-1]["jvp_temper_residual_abs_mean"]),
                "final_jvp_temper_residual_abs_max": float(rows[-1]["jvp_temper_residual_abs_max"]),
                "adaptive_carry_weight_temper": bool(args.adaptive_carry_weight_temper),
                "carry_weight_temper_ess": float(args.carry_weight_temper_ess),
                "mean_carry_weight_temper_alpha": row_nanmean("carry_weight_temper_alpha"),
                "min_carry_weight_temper_alpha": row_nanmin("carry_weight_temper_alpha"),
                "carry_weight_alpha_lt_one_frac": float(
                    np.nanmean([
                        float(r["carry_weight_temper_alpha"] < 0.999999)
                        for r in rows
                    ])
                ),
                "mean_carry_weight_temper_active_ess": row_nanmean("carry_weight_temper_active_ess"),
                "min_carry_weight_temper_active_ess": row_nanmin("carry_weight_temper_active_ess"),
                "mean_carry_weight_temper_residual_std": row_nanmean("carry_weight_temper_residual_std"),
                "adaptive_active_weight_temper": bool(args.adaptive_active_weight_temper),
                "active_weight_temper_ess": float(args.active_weight_temper_ess),
                "mean_active_weight_temper_alpha": row_nanmean("active_weight_temper_alpha"),
                "min_active_weight_temper_alpha": row_nanmin("active_weight_temper_alpha"),
                "active_weight_alpha_lt_one_frac": float(
                    np.nanmean([
                        float(r["active_weight_temper_alpha"] < 0.999999)
                        for r in rows
                    ])
                ),
                "mean_active_weight_temper_carry_ess": row_nanmean("active_weight_temper_carry_ess"),
                "min_active_weight_temper_carry_ess": row_nanmin("active_weight_temper_carry_ess"),
                "mean_active_weight_temper_residual_std": row_nanmean("active_weight_temper_residual_std"),
                "final_active_weight_temper_residual_abs_mean": float(rows[-1]["active_weight_temper_residual_abs_mean"]),
                "final_active_weight_temper_residual_abs_max": float(rows[-1]["active_weight_temper_residual_abs_max"]),
                "final_post_step_logw_std": float(rows[-1]["post_step_logw_std"]),
                "final_post_step_logw_abs_mean": float(rows[-1]["post_step_logw_abs_mean"]),
                "final_post_step_logw_abs_max": float(rows[-1]["post_step_logw_abs_max"]),
                "final_post_step_ess": float(rows[-1]["post_step_ess_frac"]),
                "mean_surrogate_logw_clip_frac": float(np.nanmean([r["surrogate_logw_clip_frac"] for r in rows])),
                "mean_raw_clip_jvp_frac": row_nanmean("raw_clip_jvp_frac"),
                "mean_raw_clip_jvp_effective_threshold": row_nanmean("raw_clip_jvp_effective_threshold"),
                "mean_raw_clip_jvp_residual_std": row_nanmean("raw_clip_jvp_residual_std"),
                "mean_raw_clip_jvp_orig_mass": row_nanmean("raw_clip_jvp_orig_mass"),
                "mean_raw_clip_jvp_clipped_mass": row_nanmean("raw_clip_jvp_clipped_mass"),
                "mean_raw_clip_reverse_kernel_frac": row_nanmean("raw_clip_reverse_kernel_frac"),
                "mean_raw_clip_reverse_kernel_residual_std": row_nanmean(
                    "raw_clip_reverse_kernel_residual_std"
                ),
                "mean_raw_clip_reverse_kernel_orig_mass": row_nanmean("raw_clip_reverse_kernel_orig_mass"),
                "mean_raw_clip_reverse_kernel_clipped_mass": row_nanmean("raw_clip_reverse_kernel_clipped_mass"),
                "mean_raw_clip_any_frac": row_nanmean("raw_clip_any_frac"),
                "mean_raw_clip_total_residual_std": row_nanmean("raw_clip_total_residual_std"),
                "mean_raw_clip_surrogate_orig_std": row_nanmean("raw_clip_surrogate_orig_std"),
                "mean_raw_clip_surrogate_clipped_std": row_nanmean("raw_clip_surrogate_clipped_std"),
                "mean_raw_clip_surrogate_orig_ess": row_nanmean("raw_clip_surrogate_orig_ess"),
                "mean_raw_clip_surrogate_clipped_ess": row_nanmean("raw_clip_surrogate_clipped_ess"),
                "mean_raw_clip_surrogate_corr": row_nanmean("raw_clip_surrogate_corr"),
                "mean_raw_clip_any_orig_mass": row_nanmean("raw_clip_any_orig_mass"),
                "mean_raw_clip_any_clipped_mass": row_nanmean("raw_clip_any_clipped_mass"),
                "mean_jvp_var_share": row_nanmean("decomp_proposal_hessian_jvp_var_share"),
                "max_jvp_var_share": row_nanmax("decomp_proposal_hessian_jvp_var_share"),
                "mean_jvp_top1_energy_frac": row_nanmean("jvp_top1_energy_frac"),
                "mean_jvp_top4_energy_frac": row_nanmean("jvp_top4_energy_frac"),
                "max_jvp_top1_energy_frac": row_nanmax("jvp_top1_energy_frac"),
                "max_jvp_top4_energy_frac": row_nanmax("jvp_top4_energy_frac"),
                "mean_jvp_top4_abs_cov_frac": row_nanmean("jvp_top4_abs_cov_frac"),
                "mean_jvp_trim_top4_std_ratio": row_nanmean("jvp_trim_top4_std_ratio"),
                "mean_surrogate_reverse_kernel_top4_energy_frac": row_nanmean(
                    "surrogate_reverse_kernel_top4_energy_frac"
                ),
                "max_surrogate_reverse_kernel_top4_energy_frac": row_nanmax(
                    "surrogate_reverse_kernel_top4_energy_frac"
                ),
                "mean_surrogate_reverse_kernel_trim_top4_std_ratio": row_nanmean(
                    "surrogate_reverse_kernel_trim_top4_std_ratio"
                ),
                "mean_finite_reverse_kernel_tail_top4_energy_frac": row_nanmean(
                    "finite_reverse_kernel_tail_top4_energy_frac"
                ),
                "max_finite_reverse_kernel_tail_top4_energy_frac": row_nanmax(
                    "finite_reverse_kernel_tail_top4_energy_frac"
                ),
                "mean_finite_reverse_kernel_tail_trim_top4_std_ratio": row_nanmean(
                    "finite_reverse_kernel_tail_trim_top4_std_ratio"
                ),
                "mean_surrogate_reverse_kernel_jvp_top4_overlap_frac": row_nanmean(
                    "surrogate_reverse_kernel_jvp_top4_overlap_frac"
                ),
                "mean_finite_reverse_kernel_jvp_top4_overlap_frac": row_nanmean(
                    "finite_reverse_kernel_jvp_top4_overlap_frac"
                ),
                "mean_jvp_brownian_norm_abs_corr_jvp_abs": row_nanmean(
                    "jvp_brownian_norm_per_sqrt_dim_abs_corr_jvp_abs"
                ),
                "mean_jvp_delta_norm_abs_corr_jvp_abs": row_nanmean("jvp_gamma_delta_norm_abs_corr_jvp_abs"),
                "mean_jvp_jbeta_delta_norm_abs_corr_jvp_abs": row_nanmean(
                    "jvp_jbeta_delta_norm_abs_corr_jvp_abs"
                ),
                "mean_jvp_core_abs_corr_jvp_abs": row_nanmean("jvp_core_abs_corr_jvp_abs"),
                "mean_surrogate_true_bad_top1_affine_resid_energy_frac": row_nanmean(
                    "surrogate_true_bad_top1_affine_resid_energy_frac"
                ),
                "mean_surrogate_true_bad_top4_affine_resid_energy_frac": row_nanmean(
                    "surrogate_true_bad_top4_affine_resid_energy_frac"
                ),
                "max_surrogate_true_bad_top1_affine_resid_energy_frac": row_nanmax(
                    "surrogate_true_bad_top1_affine_resid_energy_frac"
                ),
                "max_surrogate_true_bad_top4_affine_resid_energy_frac": row_nanmax(
                    "surrogate_true_bad_top4_affine_resid_energy_frac"
                ),
                "mean_surrogate_true_trim_bad_top4_corr": row_nanmean("surrogate_true_trim_bad_top4_corr"),
                "mean_surrogate_true_trim_bad_top4_affine_resid_rel_std": row_nanmean(
                    "surrogate_true_trim_bad_top4_affine_resid_rel_std"
                ),
                "mean_surrogate_true_bad_top4_approx_var_frac": row_nanmean(
                    "surrogate_true_bad_top4_approx_var_frac"
                ),
                "mean_surrogate_true_bad_top4_target_var_frac": row_nanmean(
                    "surrogate_true_bad_top4_target_var_frac"
                ),
                "mean_surrogate_true_bad_top4_approx_weight_mass_frac": row_nanmean(
                    "surrogate_true_bad_top4_approx_weight_mass_frac"
                ),
                "mean_surrogate_true_bad_top4_target_weight_mass_frac": row_nanmean(
                    "surrogate_true_bad_top4_target_weight_mass_frac"
                ),
            }
            with (args.out_dir / "summary.json").open("w") as f:
                json.dump(summary, f, indent=2)
            if args.save_images:
                print("[decode] waiting for per-rank image saves", flush=True)
                wait_for_decode_markers(args.out_dir, world_size, args.image_marker_timeout)
                print("[decode] assembling particle grid on rank 0", flush=True)
                assemble_grid_from_saved_images(args, args.n_particles)
            print(f"[done] {elapsed:.1f}s", flush=True)
            print(f"[saved] {rows_path}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_rho_monitor.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_genealogy.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_surrogate_rho_optimization.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_surrogate_rho_inc_ess_optimization.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_temper_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_jvp_temper_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_active_weight_temper_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_beta_variance.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_drift_magnitudes.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_term_decomposition.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_jvp_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_kernel_tail_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_raw_clip_diagnostics.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_surrogate_agreement.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'smc_pa_gate_inc_ess_agreement.png'}", flush=True)
    finally:
        dist.destroy_process_group()


def run(args: argparse.Namespace) -> None:
    legacy_kappa = float(args.linear_baseline_kappa)
    if (args.kappa_start is None) != (args.kappa_end is None):
        raise ValueError(
            "--kappa-start and --kappa-end must be provided together"
        )
    if args.kappa_start is None:
        args.kappa_start = legacy_kappa
        args.kappa_end = legacy_kappa
    else:
        args.kappa_start = float(args.kappa_start)
        args.kappa_end = float(args.kappa_end)

    devices = parse_devices(args.devices, args.device)
    if args.n_particles % len(devices) != 0:
        raise ValueError("--n-particles must be divisible by number of devices")
    if not 0.0 <= float(args.resample_ess_rearm) <= 1.0:
        raise ValueError("--resample-ess-rearm must be in [0, 1]")
    if (
        float(args.resample_ess_rearm) > 0.0
        and float(args.resample_ess_rearm) < float(args.resample_ess)
    ):
        raise ValueError(
            "--resample-ess-rearm must be 0 (disabled) or at least --resample-ess"
        )
    if not 0.0 <= float(args.jvp_shrink_alpha) <= 1.0:
        raise ValueError("--jvp-shrink-alpha must be in [0, 1]")
    if (
        not math.isfinite(float(args.pa_residual_alpha))
        or not 0.0 < float(args.pa_residual_alpha) <= 1.0
    ):
        raise ValueError("--pa-residual-alpha must be finite and in (0, 1]")
    if not 0.0 <= float(args.jvp_skip_below_gate_power_frac) <= 1.0:
        raise ValueError("--jvp-skip-below-gate-power-frac must be in [0, 1]")
    if int(args.no_jvp_first_steps) < 0:
        raise ValueError("--no-jvp-first-steps must be nonnegative")
    if int(args.no_jvp_last_steps) < 0:
        raise ValueError("--no-jvp-last-steps must be nonnegative")
    if int(args.no_jvp_first_steps) + int(args.no_jvp_last_steps) > int(
        args.steps
    ):
        raise ValueError(
            "--no-jvp-first-steps + --no-jvp-last-steps cannot exceed "
            "--steps"
        )
    if not math.isfinite(legacy_kappa) or legacy_kappa < 0.0:
        raise ValueError(
            "--linear-baseline-kappa must be finite and nonnegative"
        )
    if (
        not math.isfinite(float(args.kappa_start))
        or not math.isfinite(float(args.kappa_end))
        or float(args.kappa_start) < 0.0
        or float(args.kappa_end) < 0.0
    ):
        raise ValueError("--kappa-start/--kappa-end must be finite and nonnegative")
    if not 0.0 < float(args.kappa_end_frac) <= 1.0:
        raise ValueError("--kappa-end-frac must be in (0, 1]")
    if float(args.kappa_gamma) <= 0.0 or not math.isfinite(
        float(args.kappa_gamma)
    ):
        raise ValueError("--kappa-gamma must be finite and positive")
    if str(args.kappa_schedule) == "constant" and not math.isclose(
        float(args.kappa_start),
        float(args.kappa_end),
        abs_tol=1e-12,
    ):
        raise ValueError(
            "--kappa-schedule constant requires equal start/end values"
        )
    centered_mode = max(
        float(args.kappa_start), float(args.kappa_end)
    ) > 0.0
    scheduled_kappa = str(args.kappa_schedule) != "constant"
    if centered_mode:
        if (
            not math.isclose(float(args.cfg_scale), 1.0, abs_tol=1e-12)
            or not math.isclose(float(args.base_lambda_neg), 0.0, abs_tol=1e-12)
            or str(args.gate_ratio_mode) != "raw"
        ):
            raise ValueError(
                "Centered pure-PA mode requires cfg_scale=1, "
                "base_lambda_neg=0, and gate_ratio_mode=raw"
            )
        if float(args.lhat_temp) <= 0.0:
            raise ValueError(
                "Centered pure-PA mode requires --lhat-temp > 0"
            )
        if float(args.lhat_clip) > 0.0:
            raise ValueError(
                "Centered pure-PA mode requires --lhat-clip 0 so the "
                "linear factor and residual compensation cancel exactly"
            )
        if float(args.jvp_skip_below_gate_power_frac) > 0.0:
            raise ValueError(
                "Centered pure-PA mode requires "
                "--jvp-skip-below-gate-power-frac 0 because the centered "
                "residual can be nonzero at small gate power"
            )
        if bool(args.expansion_math_diagnostics):
            raise ValueError(
                "Centered pure-PA mode does not yet support "
                "--expansion-math-diagnostics"
            )
        if (
            scheduled_kappa
            and str(args.prop_weight_mode) == "surrogate"
            and not bool(args.hybrid_exact_gate_reverse)
        ):
            raise ValueError(
                "A nonconstant centered kappa schedule with surrogate "
                "propagation requires --hybrid-exact-gate-reverse so the "
                "cross-state compensation is inserted exactly"
            )
    if int(args.early_stop_min_unique_roots) < 0:
        raise ValueError("--early-stop-min-unique-roots must be nonnegative")
    if not 0.0 <= float(args.gate_power_start_frac) <= 1.0:
        raise ValueError("--gate-power-start-frac must be in [0, 1]")
    if not 0.0 < float(args.gate_power_end_frac) <= 1.0:
        raise ValueError("--gate-power-end-frac must be in (0, 1]")
    if float(args.gate_power_schedule_gamma) <= 0.0:
        raise ValueError("--gate-power-schedule-gamma must be positive")
    if float(args.rho_ess_min_gain) < 0.0:
        raise ValueError("--rho-ess-min-gain must be nonnegative")
    if bool(args.hybrid_exact_gate_reverse) and (
        str(args.rho_objective) not in {"incremental_ess", "total_ess"}
    ):
        raise ValueError(
            "--hybrid-exact-gate-reverse requires "
            "--rho-objective incremental_ess or total_ess"
        )
    if bool(args.hybrid_exact_gate_reverse) and (
        float(args.raw_clip_reverse_kernel) > 0.0
    ):
        raise ValueError(
            "--hybrid-exact-gate-reverse currently requires "
            "--raw-clip-reverse-kernel 0 so rho selection and propagation "
            "use the same exact reverse factor"
        )
    reference_steps = int(args.jvp_variance_reference_steps)
    if reference_steps < 0:
        raise ValueError("--jvp-variance-reference-steps must be nonnegative")
    variance_match = (
        min(1.0, math.sqrt(float(args.steps) / float(reference_steps)))
        if reference_steps > 0
        else 1.0
    )
    args.jvp_shrink_alpha_effective = (
        float(args.jvp_shrink_alpha) * float(variance_match)
    )

    prompt, negative, _ = resolve_prompt(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    early_stop_marker = args.out_dir / "early_stop.json"
    if early_stop_marker.exists():
        early_stop_marker.unlink()
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "prompts_file": str(args.prompts_file),
        "devices": devices,
        "prompt": prompt,
        "negative_prompt_used": negative,
    }
    with (args.out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    print("[task]", prompt["name"], flush=True)
    print("[positive]", prompt["positive"], flush=True)
    print(f"[negative:{args.negative_kind}]", negative, flush=True)
    print(f"[model-dtype] {args.model_dtype}", flush=True)
    print(f"[attention-processor] {args.attention_processor}", flush=True)
    print(f"[attention-slicing] {args.attention_slicing}", flush=True)
    print(f"[unet-chunk-size] {args.unet_chunk_size or 'all'}", flush=True)
    print("[target] configured base path * (1-theta)^gate_power", flush=True)
    if centered_mode:
        print(
            "[conditional/base: scheduled pure-PA linear factor] "
            f"kappa={args.kappa_start:g}->{args.kappa_end:g} "
            f"({args.kappa_schedule}, end_frac={args.kappa_end_frac:g}, "
            f"gamma={args.kappa_gamma:g}); "
            "the inverse linear factor is subtracted from the PA residual",
            flush=True,
        )
        print(
            "[centered PA residual] "
            f"alpha={args.pa_residual_alpha:g}; fixed and ESS-independent "
            "(1 is exact PA)",
            flush=True,
        )
    else:
        print(
            "[conditional/base] "
            f"eps_u + {args.cfg_scale:g}*(eps_A-eps_u) "
            f"+ {args.base_lambda_neg:g}*(eps_B-eps_u)",
            flush=True,
        )
    if str(args.gate_ratio_mode) == "raw":
        print(
            "[gate-ratio] raw A/B ratio with "
            f"ratio_exponent(lhat_temp)={args.lhat_temp:g}; no CFG-powered "
            "p_B_real is used in the gate",
            flush=True,
        )
    else:
        print(
            "[conditional/gate-branches] "
            f"p_A_real proportional to p_uncond * r_A_model^{args.cfg_scale:g}; "
            f"p_B_real proportional to p_uncond * r_B_model^{args.cfg_scale:g}",
            flush=True,
        )
        print("[gate-ratio] cfg-real powered A/B ratio", flush=True)
    print(
        f"[gate] ratio_exponent(lhat_temp)={args.lhat_temp:g} "
        "(not a diffusion-noise/sampling temperature) "
        f"gate_power={args.gate_power:g} lhat_cap={args.lhat_clip:g}",
        flush=True,
    )
    print(
        "[gate-power-path] "
        f"schedule={args.gate_power_schedule} "
        f"start_frac={args.gate_power_start_frac:g} "
        f"end_frac={args.gate_power_end_frac:g} "
        f"gamma={args.gate_power_schedule_gamma:g}; "
        f"terminal_power={args.gate_power:g}, ESS-independent, "
        "exact exponent increments leave no residual",
        flush=True,
    )
    print(
        f"[theta-shift] mid_logit_shift={args.theta_mid_logit_shift:g} "
        "(endpoint preserving; negative lowers mid-trajectory theta)",
        flush=True,
    )
    if args.rho_update:
        if args.rho_objective in {"total_ess", "incremental_ess"}:
            if bool(args.hybrid_exact_gate_reverse):
                if args.rho_ess_search_mode == "gradient":
                    rho_search_desc = (
                        f"gradient_hybrid optimizer={args.rho_optimizer} "
                        f"iterations={args.rho_steps}"
                    )
                elif args.rho_ess_search_mode in {"dense", "exact"}:
                    rho_search_desc = (
                        f"dense_hybrid candidates={max(17, int(args.rho_steps) + 1)}"
                    )
                else:
                    rho_search_desc = (
                        "coarse_hybrid candidates="
                        f"{max(3, int(args.rho_ess_coarse_candidates))}"
                    )
            else:
                rho_search_desc = (
                    f"dense_surrogate candidates={max(17, int(args.rho_steps) + 1)}+refinement"
                    if args.rho_ess_search_mode in {"dense", "exact"}
                    else f"coarse candidates={max(3, int(args.rho_ess_coarse_candidates))}"
                )
        else:
            rho_search_desc = f"optimizer={args.rho_optimizer} steps={args.rho_steps}"
        print(
            f"[rho] update=True every={args.rho_every} "
            f"lr={args.rho_lr:g} search={rho_search_desc} "
            f"cap={args.rho_cap:g} over1_penalty={args.rho_over1_penalty:g} "
            f"objective={args.rho_objective}",
            flush=True,
        )
        print(
            f"[rho-start] {'warm-start previous rho' if args.rho_warm_start else f'fresh start rho_init={args.rho_init:g}'}",
            flush=True,
        )
        print(
            "[rho-hybrid] "
            f"exact_gate_reverse={args.hybrid_exact_gate_reverse} "
            f"min_objective_ESS_gain_over_rho0={args.rho_ess_min_gain:g}; "
            "no extra diffusion-model evaluation",
            flush=True,
        )
    else:
        print(
            f"[rho] update=False; rho search ignored; fixed rho={args.rho_init:g}",
            flush=True,
        )
    print(f"[jvp] mode={args.jvp_mode} finite_diff_eps={args.jvp_eps:g}", flush=True)
    print(
        "[jvp-fixed-noise-shrink] "
        f"base_alpha={args.jvp_shrink_alpha:g} "
        f"variance_reference_steps={args.jvp_variance_reference_steps} "
        f"effective_alpha={args.jvp_shrink_alpha_effective:.6g}; "
        "fixed before rho optimization, no ESS targeting or residual carry",
        flush=True,
    )
    print(
        "[jvp-small-gate-screening] "
        f"skip_below_relative_gate_power={args.jvp_skip_below_gate_power_frac:g}; "
        "fixed and ESS-independent (use 0 for final full-JVP confirmation)",
        flush=True,
    )
    print(
        "[jvp-fixed-step-window] "
        f"skip_first={args.no_jvp_first_steps} "
        f"skip_last={args.no_jvp_last_steps}; "
        f"active=[{args.no_jvp_first_steps},"
        f"{args.steps - args.no_jvp_last_steps})",
        flush=True,
    )
    print("[jvp-beta] UNet branches=A,B only (unconditional branch cancels)", flush=True)
    print(f"[surrogate-gamma-delta] {args.surrogate_gamma_delta}", flush=True)
    print(f"[surrogate-gamma-jvp] {args.surrogate_gamma_jvp}", flush=True)
    print(f"[surrogate-logw-clip] {args.surrogate_logw_clip:g}", flush=True)
    print(
        f"[raw-term-clip] jvp={args.raw_clip_jvp:g} "
        f"rank_jvp_topk={args.rank_clip_jvp_topk} "
        f"adaptive_rank_jvp={args.adaptive_rank_clip_jvp} "
        f"adaptive_thresholds={args.adaptive_rank_clip_jvp_ess_thresholds} "
        f"adaptive_topks={args.adaptive_rank_clip_jvp_topks} "
        f"count_eps={args.raw_clip_count_eps:g} "
        f"jvp_in_rho_objective={args.raw_clip_jvp_in_rho_objective} "
        f"reverse_kernel={args.raw_clip_reverse_kernel:g} center={args.raw_clip_center}",
        flush=True,
    )
    print(
        f"[adaptive-gate-temper] enabled={args.adaptive_gate_temper} "
        f"ess={args.gate_temper_ess:g} term={args.gate_temper_term}",
        flush=True,
    )
    print(
        f"[adaptive-jvp-temper] enabled={args.adaptive_jvp_temper} "
        f"ess={args.jvp_temper_ess:g} term=raw_clipped_proposal_hessian_jvp",
        flush=True,
    )
    print(
        f"[adaptive-carry-weight-temper] enabled={args.adaptive_carry_weight_temper} "
        f"ess={args.carry_weight_temper_ess:g} term=old_carried_logw",
        flush=True,
    )
    print(
        f"[adaptive-active-weight-temper] enabled={args.adaptive_active_weight_temper} "
        f"ess={args.active_weight_temper_ess:g} term=full_current_logw",
        flush=True,
    )
    print(f"[prop-weight] {args.prop_weight_mode}", flush=True)
    print(
        f"[resample] trigger={args.resample_ess_mode} weight={args.resample_weight_mode} "
        f"threshold={args.resample_ess:g} "
        f"rearm={args.resample_ess_rearm:g} "
        f"no_last_steps={args.no_resample_last_steps}",
        flush=True,
    )
    print(
        f"[final-window] no_rho_last_steps={args.no_rho_last_steps} "
        f"no_monitor_last_steps={args.no_monitor_last_steps}",
        flush=True,
    )
    print(f"[finite-diagnostic-weight] pA-gate full-x every={args.finite_diagnostic_every}", flush=True)

    if args.dry_run:
        print("[dry-run] configuration written; model not loaded.", flush=True)
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.dist_port)
    mp.spawn(distributed_worker, args=(args, devices), nprocs=len(devices), join=True)
    if early_stop_marker.exists():
        print(
            f"[run early-stopped] {early_stop_marker}; skipping final image scoring",
            flush=True,
        )
        return
    if args.clip_score:
        if not args.save_images:
            raise ValueError("--clip-score requires --save-images so final particles exist on disk")
        score_final_images_with_clip(args, prompt, negative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pA-gate SMC with retained-order rho local variance optimization")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7")
    parser.add_argument("--dist-port", type=int, default=29901)
    parser.add_argument("--model-id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--model-dtype", choices=["fp16", "fp32"], default="fp32",
                        help="fp32 is useful for strict forward-mode AD JVP on CUDA")
    parser.add_argument("--attention-processor", choices=["default", "legacy"], default="legacy",
                        help="legacy avoids torch scaled_dot_product_attention and may permit forward-mode AD")
    parser.add_argument("--attention-slicing", choices=["none", "auto", "max"], default="max",
                        help="use max with --attention-processor legacy --model-dtype fp32 to reduce memory")
    parser.add_argument("--unet-chunk-size", type=int, default=1,
                        help="split the local particle batch for UNet calls; use 1 for legacy fp32 forward-AD")
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--negative-kind", choices=["related", "unrelated"], default="related")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "smc_pa_gate_rho_sd14_200steps_p1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--lambda-a", type=float, default=1.0,
                        help="Ignored compatibility argument; pA-gate uses its configured base path.")
    parser.add_argument("--lambda-b", type=float, default=1.0,
                        help="Ignored compatibility argument; use --gate-ratio-mode to choose the A/B ratio.")
    parser.add_argument("--cfg-scale", "--lambda-cfg", dest="cfg_scale", type=float, default=7.5,
                        help=(
                            "Conditional model exponent lambda_cfg in "
                            "p_real proportional to p_uncond * r_model^lambda_cfg, "
                            "where r_model is the CFG prompt likelihood-ratio factor. "
                            "Use bs0/bs1 divided by this value to keep eta * beta close "
                            "to an un-CFG run."
                        ))
    parser.add_argument(
        "--base-lambda-neg",
        type=float,
        default=0.0,
        help=(
            "Optional negative-prompt coefficient in the base epsilon: "
            "eps_u + cfg_scale*(eps_A-eps_u) + base_lambda_neg*(eps_B-eps_u). "
            "It does not alter the separate gate A/B branches or beta_l."
        ),
    )
    parser.add_argument(
        "--linear-baseline-kappa",
        type=float,
        default=0.0,
        help=(
            "Pure-PA algebraic centering coefficient. For kappa>0, factor "
            "V_PA into the NP-equivalent linear reference "
            "p_A*(p_A/p_B)^kappa and a centered nonlinear PA residual. "
            "Rho acts only on that residual; the total target remains PA."
        ),
    )
    parser.add_argument(
        "--pa-residual-alpha",
        type=float,
        default=1.0,
        help=(
            "Fixed, ESS-independent tempering of only the centered nonlinear "
            "PA residual. The operational log-potential is "
            "-kappa*lhat/lhat_temp + alpha*[log(V_PA) + "
            "kappa*lhat/lhat_temp]. Alpha=1 is exact PA; values just below "
            "one introduce a controlled bias toward the linear PA limit."
        ),
    )
    parser.add_argument(
        "--kappa-start",
        type=float,
        default=None,
        help=(
            "Initial deterministic centered-reference kappa. Omit together "
            "with --kappa-end to preserve constant "
            "--linear-baseline-kappa behavior."
        ),
    )
    parser.add_argument(
        "--kappa-end",
        type=float,
        default=None,
        help="Terminal deterministic centered-reference kappa.",
    )
    parser.add_argument(
        "--kappa-schedule",
        choices=["constant", "linear", "power", "cosine"],
        default="constant",
        help=(
            "Fixed, ESS-independent interpolation from --kappa-start to "
            "--kappa-end."
        ),
    )
    parser.add_argument(
        "--kappa-end-frac",
        type=float,
        default=1.0,
        help=(
            "Reverse-progress fraction by which a nonconstant kappa schedule "
            "reaches --kappa-end."
        ),
    )
    parser.add_argument(
        "--kappa-gamma",
        type=float,
        default=1.0,
        help="Positive exponent for --kappa-schedule power.",
    )
    parser.add_argument("--gate-ratio-mode", choices=["cfg-real", "raw"], default="cfg-real",
                        help=(
                            "cfg-real gates on p_B_real/p_A_real, preserving the legacy "
                            "lambda_cfg-powered A/B ratio. raw gates on the unpowered "
                            "conditional A/B ratio. Neither mode includes base_lambda_neg "
                            "in the gate ratio."
                        ))
    parser.add_argument("--c0", type=float, default=300.0)
    parser.add_argument("--c1", type=float, default=0.01)
    parser.add_argument("--bs0", type=float, default=0.02)
    parser.add_argument("--bs1", type=float, default=0.0002)
    parser.add_argument(
        "--c-schedule",
        choices=["constant", "linear", "log", "exp", "geometric"],
        default="exp",
        help=(
            "Interpolation for c_t. geometric interpolates linearly in log(c_t), "
            "so the deterministic c_t contribution to the theta logit has constant velocity."
        ),
    )
    parser.add_argument(
        "--bs-schedule",
        choices=["constant", "linear", "log", "exp", "geometric"],
        default="exp",
        help="Interpolation for eta_t; geometric requires positive endpoints.",
    )
    parser.add_argument("--gamma-c", type=float, default=2.0)
    parser.add_argument("--gamma-bs", type=float, default=3.0)
    parser.add_argument(
        "--theta-mid-logit-shift",
        type=float,
        default=0.0,
        help=(
            "Endpoint-preserving mid-trajectory shift added to the theta logit as "
            "shift*sin(pi*progress)^2. Negative values lower mid-stage theta and "
            "gate_power*eta*theta while leaving both endpoint targets unchanged."
        ),
    )
    parser.add_argument("--theta-eps", type=float, default=0.0,
                        help="Optional clamp for proposal theta; 0 keeps the gate algebra exact.")
    parser.add_argument("--lhat-temp", "--lambda-temp", dest="lhat_temp", type=float, default=0.2,
                        help="Multiplier on each raw A/B Gaussian log-ratio increment used to update lhat.")
    parser.add_argument("--gate-power", "--lambda-s", "--soft-gate-power",
                        dest="gate_power", type=float, default=200.0,
                        help="Exponent lambda_s on the soft gate (1-theta)^lambda_s.")
    parser.add_argument(
        "--gate-power-schedule",
        choices=["constant", "linear", "power", "cosine"],
        default="constant",
        help=(
            "Fixed ESS-independent annealing path for the gate exponent. "
            "Every nonconstant path reaches --gate-power at the data endpoint."
        ),
    )
    parser.add_argument(
        "--gate-power-start-frac",
        type=float,
        default=0.0,
        help=(
            "Initial fraction of --gate-power for a nonconstant gate-power "
            "schedule; must be in [0,1]."
        ),
    )
    parser.add_argument(
        "--gate-power-end-frac",
        type=float,
        default=1.0,
        help=(
            "Progress fraction by which a nonconstant gate-power schedule "
            "reaches its full terminal exponent; must be in (0,1]. Values "
            "below one form a fixed ESS-independent power-to-plateau path."
        ),
    )
    parser.add_argument(
        "--gate-power-schedule-gamma",
        type=float,
        default=1.0,
        help="Positive exponent for --gate-power-schedule power.",
    )
    parser.add_argument("--lhat-clip", "--lhat-cap", dest="lhat_clip", type=float, default=25.0,
                        help="Optional symmetric lhat clipping; 0 disables clipping for exact gate-ratio tests.")
    parser.add_argument("--min-weight-variance", type=float, default=1e-12)
    parser.add_argument("--prop-weight-mode", choices=["surrogate", "true"], default="true",
                        help="Incremental SMC weight used for ESS/resampling")
    parser.add_argument("--finite-diagnostic-every", type=int, default=1,
                        help="Compute expensive finite Gaussian diagnostic every N steps; 0 disables it unless --prop-weight-mode true needs the current finite weight.")
    parser.add_argument("--expansion-math-diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute auxiliary expansion-vs-finite math diagnostics; disable for faster surrogate-only sweeps.")
    parser.add_argument("--timing-steps", default="",
                        help="Comma-separated zero-based reverse-step indices to time; also supports 'last', negative indices, or 'all'. Empty disables timing.")
    parser.add_argument("--resample-ess", type=float, default=0.9)
    parser.add_argument(
        "--resample-ess-rearm",
        type=float,
        default=0.0,
        help=(
            "Optional ESS hysteresis threshold. After resampling, suppress another "
            "resample until the selected ESS statistic reaches this value. Use 0 "
            "to preserve legacy behavior; a value above --resample-ess prevents "
            "consecutive threshold chatter."
        ),
    )
    parser.add_argument("--resample-ess-mode",
                        choices=["cum", "inc", "cess", "cum_or_inc", "cum_or_cess"],
                        default="cum",
                        help=(
                            "ESS statistic used for the adaptive resampling trigger. "
                            "cum is the active accumulated selection-weight ESS; with gate "
                            "tempering this excludes the newly deferred residual. inc uses "
                            "only the latest active incremental weights; cess is conditional "
                            "ESS using previous normalized weights; cum_or_* resamples if "
                            "either source of degeneracy is severe."
                        ))
    parser.add_argument("--resample-weight-mode",
                        choices=["cum", "inc"],
                        default="cum",
                        help=(
                            "Ancestor weights used when resampling fires. cum uses the active "
                            "carried/old plus current selection weights and consumes those "
                            "weights; inc uses only the current incremental weights, matching "
                            "the local Linrui-style anti-collapse heuristic."
                        ))
    parser.add_argument("--no-resample-last-steps", type=int, default=0,
                        help="Forbid resampling in the final K reverse steps; 0 keeps the usual ESS trigger.")
    parser.add_argument(
        "--plot-mask-final-inc-ess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Plot-only convention: display incremental ESS/N curves as 1 in "
            "the --no-resample-last-steps window. Saved monitor rows and all "
            "sampling/resampling calculations retain their true values."
        ),
    )
    parser.add_argument("--no-rho-last-steps", type=int, default=0,
                        help="Forbid rho optimization in the final K reverse steps; 0 keeps the usual rho schedule.")
    parser.add_argument("--no-monitor-last-steps", type=int, default=0,
                        help="Do not append monitor rows/plots for the final K reverse steps; 0 monitors every step.")
    parser.add_argument(
        "--early-stop-min-unique-roots",
        type=int,
        default=0,
        help=(
            "Gracefully stop immediately after a resample leaves fewer than "
            "this many unique initial ancestors. Unique roots cannot recover. "
            "Zero disables early stopping; use 17 to enforce roots >16."
        ),
    )
    parser.add_argument("--rho-update", action=argparse.BooleanOptionalAction, default=False,
                        help="Optimize rho locally. Default off for capped-lhat runs where rho is not meaningful.")
    parser.add_argument("--rho-every", type=int, default=5)
    parser.add_argument("--rho-init", type=float, default=0.0)
    parser.add_argument("--rho-lr", type=float, default=0.05)
    parser.add_argument(
        "--rho-steps",
        type=int,
        default=20,
        help=(
            "Optimizer iterations for gradient rho objectives, including "
            "--rho-ess-search-mode gradient. For dense total_ess/incremental_ess, "
            "sets the grid to max(17, rho_steps+1) candidates."
        ),
    )
    parser.add_argument(
        "--rho-ess-search-mode",
        choices=["dense", "coarse", "gradient", "exact"],
        default="dense",
        help=(
            "Surrogate-only search used by total_ess/incremental_ess objectives. dense "
            "uses the max(17, rho_steps+1) grid plus golden-section refinement; it does "
            "not run a finite model validation. coarse searches only "
            "--rho-ess-coarse-candidates evenly spaced values and skips refinement. "
            "gradient uses --rho-optimizer adam/gd, --rho-lr, and --rho-steps; in the "
            "hybrid path it differentiates the same exact-gate/reverse ESS objective. "
            "exact is retained as a backward-compatible alias for dense."
        ),
    )
    parser.add_argument(
        "--rho-ess-coarse-candidates",
        type=int,
        default=11,
        help="Number of rho candidates for --rho-ess-search-mode coarse.",
    )
    parser.add_argument("--rho-cap", type=float, default=2.0,
                        help="Clamp |rho| to this value; 0 freezes rho, negative disables the cap.")
    parser.add_argument("--rho-line-search-halvings", type=int, default=12,
                        help="Per-rho-step backtracking retries; rollback and halve lr until surrogate variance decreases.")
    parser.add_argument("--rho-over1-penalty", type=float, default=0.0,
                        help="Soft semantic penalty alpha * max(0, rho - 1)^2 added to the rho variance objective.")
    parser.add_argument("--rho-optimizer", choices=["adam", "gd", "exact"], default="adam",
                        help="exact minimizes the one-dimensional quartic surrogate variance over the rho cap.")
    parser.add_argument("--rho-objective",
                        choices=["total_logw", "total_ess", "incremental_ess", "incremental_weighted"],
                        default="total_ess",
                        help=(
                            "Objective for rho optimization. total_logw minimizes the variance of "
                            "log omega_old plus the current surrogate expansion. incremental_weighted "
                            "is the older objective: variance of the current expansion under normalized "
                            "old particle weights. total_ess maximizes the ESS of carried residual/old "
                            "weights plus the current surrogate expansion. incremental_ess maximizes "
                            "the ESS of only the current surrogate increment."
                        ))
    parser.add_argument(
        "--hybrid-exact-gate-reverse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For incremental-ESS or total-ESS rho selection, replace the "
            "retained gate and reverse-kernel expansions by their exact finite "
            "Gaussian factors. This uses no additional diffusion-model "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--rho-ess-min-gain",
        type=float,
        default=0.0,
        help=(
            "Minimum fractional objective-ESS improvement over rho=0 needed to "
            "select a nonzero rho in the hybrid exact-gate/reverse search. The "
            "objective ESS is incremental or total according to "
            "--rho-objective. This stabilizes rho on numerically flat "
            "objectives."
        ),
    )
    parser.add_argument("--rho-warm-start", action="store_true", default=False,
                        help="Start rho optimization from previous step's rho instead of --rho-init.")
    parser.add_argument("--no-rho-warm-start", action="store_false", dest="rho_warm_start")
    parser.add_argument("--rho-accept-only-improve", action="store_true", default=True)
    parser.add_argument("--no-rho-accept-only-improve", action="store_false", dest="rho_accept_only_improve")
    parser.add_argument("--rho-guard-rho0", action=argparse.BooleanOptionalAction, default=True,
                        help="After local optimization, use rho=0 if its surrogate variance is no worse.")
    parser.add_argument("--jvp-mode", choices=["auto", "forward-ad", "finite-diff"], default="auto")
    parser.add_argument("--jvp-eps", type=float, default=1e-2)
    parser.add_argument(
        "--jvp-shrink-alpha",
        type=float,
        default=1.0,
        help=(
            "Fixed ESS-independent multiplier in [0,1] on the noisy JVP "
            "quadratic-form contribution. It is applied before rho optimization "
            "and carries no residual."
        ),
    )
    parser.add_argument(
        "--jvp-skip-below-gate-power-frac",
        type=float,
        default=0.0,
        help=(
            "Optional fixed screening tolerance: omit the retained JVP term "
            "while the current gate exponent is below this fraction of its "
            "terminal value. The JVP term is linear in gate power. Use zero "
            "for final full-JVP evaluations."
        ),
    )
    parser.add_argument(
        "--no-jvp-first-steps",
        type=int,
        default=0,
        help=(
            "Omit the expensive JVP curvature term in the first K reverse "
            "steps. This is fixed and ESS-independent; rho search and all "
            "non-JVP surrogate terms remain active."
        ),
    )
    parser.add_argument(
        "--no-jvp-last-steps",
        type=int,
        default=0,
        help=(
            "Omit the expensive JVP curvature term in the final K reverse "
            "steps. With 100 steps, first=20 and last=10 computes JVPs only "
            "on reverse steps 20 through 89."
        ),
    )
    parser.add_argument(
        "--jvp-variance-reference-steps",
        type=int,
        default=0,
        help=(
            "If positive, additionally multiply the JVP contribution by "
            "min(1,sqrt(steps/reference_steps)). This matches the centered "
            "fixed-noise variance to a trusted finer discretization; 0 disables "
            "the variance-matching correction."
        ),
    )
    parser.add_argument("--surrogate-gamma-delta",
                        choices=["brownian", "base_brownian", "base_guidance0", "full_deterministic", "full"],
                        default="brownian",
                        help=(
                            "Displacement used in the Gamma quadratic term of the local rho surrogate. "
                            "brownian is the legacy z-only expansion; base_brownian uses "
                            "Delta=base+Brownian with one JVP; base_guidance0 uses "
                            "Delta=base+Brownian+guidance_rho0 with one JVP; full_deterministic uses "
                            "Delta=base+Brownian+guidance_rho0 with one JVP; full uses "
                            "Delta=base+Brownian+(1-rho)*guidance_rho0 and computes separate "
                            "base+Brownian and score-guidance JVPs."
                        ))
    parser.add_argument("--surrogate-gamma-jvp", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the expensive J beta[Delta] term in Gamma; disabling keeps only the rank-one gate curvature for the chosen Delta.")
    parser.add_argument("--surrogate-logw-clip", type=float, default=0.0,
                        help="Clamp the final evaluated surrogate log-weight to +/- this value before propagation/resampling; 0 disables.")
    parser.add_argument("--raw-clip-jvp", type=float, default=0.0,
                        help="Raw biased centered clamp for the additive proposal_hessian_jvp surrogate term; 0 disables.")
    parser.add_argument("--rank-clip-jvp-topk", type=int, default=0,
                        help="Rank winsorize the top-K centered JVP particles to the (K+1)th magnitude; 0 disables.")
    parser.add_argument("--adaptive-rank-clip-jvp", action=argparse.BooleanOptionalAction, default=False,
                        help=(
                            "Choose JVP rank-clip K from the current unclipped surrogate incremental ESS "
                            "instead of using a fixed --rank-clip-jvp-topk."
                        ))
    parser.add_argument("--adaptive-rank-clip-jvp-ess-thresholds", default="0.97,0.93,0.88",
                        help=(
                            "High-to-low ESS/N cutoffs for --adaptive-rank-clip-jvp. "
                            "Default 0.97,0.93,0.88 with topks 0,2,4,8."
                        ))
    parser.add_argument("--adaptive-rank-clip-jvp-topks", default="0,2,4,8",
                        help=(
                            "Comma-separated rank-clip K values for the ESS bins. "
                            "Must have one more value than the threshold list."
                        ))
    parser.add_argument("--raw-clip-count-eps", type=float, default=1e-6,
                        help="Residual tolerance used when counting raw/rank clipped particles.")
    parser.add_argument("--raw-clip-jvp-in-rho-objective", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply --raw-clip-jvp inside the total_ess rho objective as well as the final surrogate log weight.")
    parser.add_argument("--raw-clip-reverse-kernel", type=float, default=0.0,
                        help="Raw biased centered clamp for the additive reverse Gaussian surrogate kernel group; 0 disables.")
    parser.add_argument("--raw-clip-center", choices=["median", "mean", "zero"], default="median",
                        help="Center used for raw term clipping before applying +/- clip thresholds.")
    parser.add_argument("--adaptive-jvp-temper", action=argparse.BooleanOptionalAction, default=False,
                        help="Temper the raw-clipped proposal_hessian_jvp term used for immediate selection and carry the residual.")
    parser.add_argument("--jvp-temper-ess", type=float, default=0.95,
                        help="Minimum active cumulative ESS fraction for the JVP bridge; default is 0.95.")
    parser.add_argument("--adaptive-carry-weight-temper", action=argparse.BooleanOptionalAction, default=False,
                        help="Temper old carried log-weight before adding the current active log-weight, and carry the deferred old residual.")
    parser.add_argument("--carry-weight-temper-ess", type=float, default=0.95,
                        help="Minimum active ESS fraction for the old carried log-weight bridge; default is 0.95.")
    parser.add_argument("--adaptive-active-weight-temper", action=argparse.BooleanOptionalAction, default=False,
                        help="Temper the whole current log-weight used for immediate selection and carry the residual.")
    parser.add_argument("--active-weight-temper-ess", type=float, default=0.9,
                        help="Minimum active cumulative ESS fraction for the full active-weight bridge; default is 0.9.")
    parser.add_argument("--adaptive-gate-temper", action=argparse.BooleanOptionalAction, default=True,
                        help="Temper the gate log-weight used for immediate selection and carry the residual gate weight.")
    parser.add_argument("--gate-temper-ess", type=float, default=0.8,
                        help="Minimum incremental ESS fraction for the active gate bridge; default is 0.8.")
    parser.add_argument("--gate-temper-term", choices=["exact", "surrogate"], default="exact",
                        help="Gate term to temper: exact finite gate ratio or the local surrogate gate group.")
    parser.add_argument("--surrogate-every-step", action=argparse.BooleanOptionalAction, default=False,
                        help="Compute local surrogate diagnostics every step, not only rho optimization steps")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-images", action="store_true", default=False)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--image-marker-timeout", type=float, default=1800.0)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--clip-score", action=argparse.BooleanOptionalAction, default=False,
                        help="After saving final images, compute CLIP(image, positive/related-negative/unrelated-negative) metrics.")
    parser.add_argument("--clip-model-id", default=DEFAULT_CLIP_MODEL_ID)
    parser.add_argument("--clip-score-device", default=None,
                        help="Device for post-run CLIP scoring; defaults to the first sampling device when CUDA is available.")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-text-batch-size", type=int, default=16)
    parser.add_argument("--clip-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
