#!/usr/bin/env python3
"""Finite-kernel/Proposition-2 SMC tests for the direct pA * soft-gate target.

This is a deliberately small experimental runner for one prompt pair.  It
uses the direct pA reference (no centered p^kappa factor), the two-parameter
forward field

    u_t = rho1 * (score_A - score_B) + rho2 * grad R_t,

and the matched reverse proposal.  The propagated incremental weight is the
exact finite Gaussian change of measure

    Delta R + log L_A - log L_rho + log K_rho - log K_A.

Candidate selection can use a finite Gaussian ratio (with a frozen or JVP
endpoint field) or the retained local expansion in Proposition 2.  After
selecting (rho1, rho2), the propagated weight can use that local expansion,
the endpoint-field JVP surrogate, or a direct score evaluation at the
realized endpoint.  All per-particle terms and optimizer surfaces are saved
for later reconstruction.
"""

from __future__ import annotations

import argparse
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prop2-two-rho")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SCRIPT_DIR = Path(__file__).resolve().parent
# Spawned workers can inherit the neighboring dng_eval directory ahead of this
# rebuttal directory.  Force the intended local runner to the front because a
# legacy file with the same module name also exists there.
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import run_smc_pa_gate_cfg_real_kernel_rawclip_monitor as base


PARTICLE_FIELDS = (
    "lhat_old",
    "lhat_new",
    "theta_old",
    "theta_new",
    "kernel_lr",
    "zeta_old",
    "zeta_new",
    "rho1",
    "rho2",
    "effective_r_old",
    "effective_r_new",
    "d_y_norm2",
    "g_y_norm2",
    "u_y_norm2",
    "d_y_dot_g_y",
    "d_x_norm2",
    "u_x_norm2",
    "nu_a_dot_d",
    "forward_displacement_dot_d",
    "alpha_l",
    "h_alpha_s",
    "alpha_gate_time",
    "alpha_forward_linear",
    "alpha_forward_quadratic",
    "alpha_reverse_linear",
    "alpha_reverse_quadratic",
    "alpha_no_hessian_total",
    "alpha_target_norm_term",
    "c48_center_correction",
    "prop2_gate_time",
    "prop2_target_curvature",
    "prop2_forward_alpha",
    "prop2_reverse_alpha",
    "prop2_proposal_curvature",
    "prop2_jvp_raw",
    "prop2_total_raw",
    "prop2_noise_total_raw",
    "prop2_jd_quadratic",
    "prop2_d_projection2",
    "prop2_curvature_coefficient",
    "diag_finite_exact_total",
    "diag_finite_exact_gate",
    "diag_finite_exact_reverse",
    "diag_finite_exact_forward",
    "diag_finite_no_jvp_total",
    "diag_finite_jvp_total",
    "diag_finite_forward_zero",
    "diag_finite_jvp_forward",
    "diag_finite_true_jvp_correction",
    "diag_finite_est_jvp_correction",
    "logw_gate",
    "logw_gate_used",
    "logw_gate_clip_delta",
    "logw_reverse_kernel",
    "logw_forward_kernel",
    "logw_forward_kernel_zero",
    "logw_jvp_correction",
    "logw_jvp_correction_used",
    "logw_kernel_correction",
    "logw_kernel_correction_used",
    "logw_kernel_correction_clip_delta",
    "logw_total",
    "logw_total_used",
    "logw_predicted_fixed_field",
    "logw_prediction_error",
    "logw_carry_before",
    "logw_cumulative_before_resample",
    "logw_resample_selection",
    "logw_resample_clip_residual",
    "origin_id_before",
    "origin_id_after",
)


def focused_schedule_at_state(
    k: int, n_steps: int, args: argparse.Namespace
) -> tuple[float, float]:
    """Return c/eta with an optional endpoint-preserving plateau in log c."""
    c, eta = base.schedule_at_state(k, n_steps, args)
    if str(args.c_path_shape) != "plateau":
        return float(c), float(eta)

    progress = min(
        1.0, max(0.0, float(k) / max(int(n_steps) - 1, 1))
    )
    down_end = float(args.c_path_down_end_frac)
    return_start = float(args.c_path_return_start_frac)
    if progress <= down_end:
        phase = progress / down_end
        weight = 0.5 - 0.5 * math.cos(math.pi * phase)
    elif progress < return_start:
        weight = 1.0
    else:
        phase = (progress - return_start) / (1.0 - return_start)
        weight = 0.5 + 0.5 * math.cos(math.pi * phase)
    return (
        float(c) * math.exp(float(args.c_path_logit_shift) * weight),
        float(eta),
    )


def fixed_rho1_at_state(
    k: int, n_steps: int, args: argparse.Namespace
) -> float:
    """Optional deterministic taper for the fixed score-difference field."""
    start = float(args.fixed_rho1)
    end = float(args.fixed_rho1_final)
    mode = str(args.fixed_rho1_schedule)
    if mode == "constant":
        return start
    progress = min(
        1.0,
        max(0.0, float(k) / max(int(n_steps) - 1, 1)),
    )
    start_frac = min(
        1.0, max(0.0, float(args.fixed_rho1_start_frac))
    )
    end_frac = min(1.0, max(1e-8, float(args.fixed_rho1_end_frac)))
    if end_frac <= start_frac:
        raise ValueError(
            "--fixed-rho1-end-frac must exceed --fixed-rho1-start-frac"
        )
    phase = min(
        1.0,
        max(0.0, (progress - start_frac) / (end_frac - start_frac)),
    )
    if mode == "linear":
        weight = phase
    elif mode == "cosine":
        weight = 0.5 - 0.5 * math.cos(math.pi * phase)
    else:
        raise ValueError(mode)
    return start + weight * (end - start)


def candidate_grid(args: argparse.Namespace, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if args.mode.endswith("_continuous"):
        # Continuous modes save optimizer diagnostics rather than a grid.
        return (
            torch.tensor([float("nan")], device=device),
            torch.tensor([float("nan")], device=device),
        )
    if args.mode == "c48_ratio":
        rho2 = torch.linspace(
            args.rho2_min, args.rho2_max, args.rho2_points,
            device=device, dtype=torch.float32,
        )
        rho1 = -float(args.c48_kappa) * rho2
    elif args.mode == "fixed_rho1":
        rho2 = torch.linspace(
            args.rho2_min, args.rho2_max, args.rho2_points,
            device=device, dtype=torch.float32,
        )
        rho1 = torch.full_like(rho2, float(args.fixed_rho1))
    elif args.mode == "free":
        r1 = torch.linspace(
            args.rho1_min, args.rho1_max, args.rho1_points,
            device=device, dtype=torch.float32,
        )
        r2 = torch.linspace(
            args.rho2_min, args.rho2_max, args.rho2_points,
            device=device, dtype=torch.float32,
        )
        mesh1, mesh2 = torch.meshgrid(r1, r2, indexing="ij")
        rho1, rho2 = mesh1.reshape(-1), mesh2.reshape(-1)
    else:
        raise ValueError(args.mode)
    return rho1, rho2


def _gather_sufficient(
    local: dict[str, torch.Tensor | float], world_size: int
) -> dict[str, torch.Tensor | float]:
    return {
        key: (
            base.all_gather_cat(value.reshape(-1).contiguous(), world_size)
            if torch.is_tensor(value)
            else value
        )
        for key, value in local.items()
    }


def _stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float().reshape(-1)
    return {
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
    }


def _prefixed_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in _stats(x).items()}


def _gate_reward_and_zeta(
    lhat: torch.Tensor,
    *,
    c: float,
    eta: float,
    gate_power: float,
    lhat_temp: float,
    reward_tanh_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the gate log-potential, theta, and its score coefficient.

    A positive ``reward_tanh_scale`` applies a centered smooth clipping,

        R_clip(l) = C tanh((R(l)-R(0))/C),

    which retains the local slope at l=0 while bounding the total potential.
    The returned zeta includes the exact sech^2 derivative of this transform.
    """
    logit = math.log(max(float(c), 1e-30)) + float(eta) * lhat.float()
    theta = torch.sigmoid(logit)
    log1m = -torch.nn.functional.softplus(logit)
    raw_reward = float(gate_power) * log1m
    scale = float(reward_tanh_scale)
    if scale > 0.0:
        log1m_zero = -torch.nn.functional.softplus(
            torch.as_tensor(
                math.log(max(float(c), 1e-30)),
                device=lhat.device,
                dtype=torch.float32,
            )
        )
        centered = raw_reward - float(gate_power) * log1m_zero
        squashed = torch.tanh(centered / scale)
        reward = scale * squashed
        derivative_multiplier = 1.0 - squashed.square()
    else:
        reward = raw_reward
        derivative_multiplier = torch.ones_like(theta)
    zeta = (
        float(gate_power)
        * float(eta)
        * float(lhat_temp)
        * theta
        * derivative_multiplier
    )
    return reward.float(), theta.float(), zeta.float()


def _ess_rows(logw: torch.Tensor) -> torch.Tensor:
    values = logw.float()
    values = values - values.max(dim=1, keepdim=True).values
    weights = torch.softmax(values, dim=1)
    return 1.0 / (float(values.shape[1]) * weights.square().sum(dim=1))


def _rho_grid_diagnostics(
    candidate_increments: torch.Tensor,
    cumulative_logw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return incremental ESS, full ESS, and full log-weight variance."""
    if candidate_increments.ndim != 2:
        raise ValueError("Candidate increments must be candidate-by-particle.")
    if cumulative_logw.ndim != 1:
        raise ValueError("Cumulative log weights must be one-dimensional.")
    if candidate_increments.shape[1] != cumulative_logw.shape[0]:
        raise ValueError(
            "Candidate and cumulative particle dimensions do not match: "
            f"{candidate_increments.shape} versus {cumulative_logw.shape}."
        )
    candidate_total = candidate_increments + cumulative_logw[None, :]
    return (
        _ess_rows(candidate_increments),
        _ess_rows(candidate_total),
        candidate_total.float().var(dim=1, unbiased=False),
    )


def _select_rho_grid_index(
    *,
    objective: str,
    total_ess: torch.Tensor,
    total_logw_variance: torch.Tensor,
) -> int:
    """Select a rho-grid row using the requested post-clip objective."""
    if objective == "full_ess":
        return int(torch.argmax(total_ess).item())
    if objective == "log_weight_variance":
        return int(torch.argmin(total_logw_variance).item())
    raise ValueError(f"Unknown rho-grid objective: {objective}")


def _rank_winsorize_rows(
    values: torch.Tensor,
    topk: int,
    max_abs: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Median-center and winsorize extremes per row.

    The threshold is the (topk + 1)-st largest absolute centered value, so
    values below the threshold are unchanged.  Ties at the threshold are also
    retained.  If ``max_abs`` is positive, the final threshold is no larger
    than this value.  Thus the rank rule limits isolated outliers, while the
    value rule controls a broadly dispersed reward increment. ``values`` may
    be a single particle row or a candidate-by-particle matrix.
    """
    original_shape = values.shape
    rows = values.float().reshape(1, -1) if values.ndim == 1 else values.float()
    if rows.ndim != 2:
        raise ValueError(f"Expected one or two dimensions, got {original_shape}")
    n = int(rows.shape[1])
    k = max(0, min(int(topk), max(0, n - 1)))
    center = rows.median(dim=1, keepdim=True).values
    residual = rows - center
    magnitude = residual.abs()
    if k == 0:
        threshold = torch.full_like(center, float("inf"))
    else:
        threshold = torch.topk(
            magnitude, k=k + 1, dim=1, largest=True, sorted=True
        ).values[:, -1:]
    if float(max_abs) > 0.0:
        threshold = torch.minimum(
            threshold,
            torch.full_like(threshold, float(max_abs)),
        )
    mask = magnitude > threshold
    used = center + residual.sign() * torch.minimum(magnitude, threshold)
    if values.ndim == 1:
        return used[0], mask[0], threshold[0, 0]
    return used, mask, threshold[:, 0]


def _candidate_matrix_after_gate_clip(
    local_matrix: torch.Tensor,
    local_components: dict[str, torch.Tensor],
    *,
    world_size: int,
    topk: int,
    max_abs: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gather candidate terms and clip gate outliers before ESS selection."""
    global_matrix_raw = base.all_gather_particle_rows(local_matrix, world_size)
    global_gate_raw = base.all_gather_particle_rows(
        local_components["gate"], world_size
    )
    global_gate_used, gate_clip_mask, gate_clip_threshold = (
        _rank_winsorize_rows(global_gate_raw, topk, max_abs)
    )
    global_matrix_used = global_matrix_raw + global_gate_used - global_gate_raw
    return global_matrix_used, {
        "raw": global_matrix_raw,
        "gate_raw": global_gate_raw,
        "gate_used": global_gate_used,
        "gate_clip_mask": gate_clip_mask,
        "gate_clip_threshold": gate_clip_threshold,
    }


def _candidate_matrix_after_gate_and_jvp_clip(
    local_matrix: torch.Tensor,
    local_components: dict[str, torch.Tensor],
    local_zero_components: dict[str, torch.Tensor],
    *,
    world_size: int,
    gate_topk: int,
    gate_max_abs: float,
    jvp_topk: int,
    jvp_max_abs: float,
    kernel_topk: int,
    kernel_max_abs: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gather candidates and robustify gate and kernel contributions.

    The kernel correction combines the reverse, forward, and already
    stabilized JVP terms.  Clipping the combined correction preserves their
    finite-step cancellations and applies the same approximation during
    proposal selection and weight accumulation.
    """
    global_matrix_raw = base.all_gather_particle_rows(local_matrix, world_size)
    global_gate_raw = base.all_gather_particle_rows(
        local_components["gate"], world_size
    )
    global_gate_used, gate_clip_mask, gate_clip_threshold = (
        _rank_winsorize_rows(global_gate_raw, gate_topk, gate_max_abs)
    )
    local_jvp = (
        local_components["forward"] - local_zero_components["forward"]
    )
    global_jvp_raw = base.all_gather_particle_rows(local_jvp, world_size)
    global_jvp_used, jvp_clip_mask, jvp_clip_threshold = (
        _rank_winsorize_rows(global_jvp_raw, jvp_topk, jvp_max_abs)
    )
    global_kernel_raw = global_matrix_raw - global_gate_raw
    global_kernel_after_jvp = (
        global_kernel_raw + global_jvp_used - global_jvp_raw
    )
    (
        global_kernel_used,
        kernel_clip_mask,
        kernel_clip_threshold,
    ) = _rank_winsorize_rows(
        global_kernel_after_jvp,
        kernel_topk,
        kernel_max_abs,
    )
    global_matrix_used = global_gate_used + global_kernel_used
    return global_matrix_used, {
        "raw": global_matrix_raw,
        "gate_raw": global_gate_raw,
        "gate_used": global_gate_used,
        "gate_clip_mask": gate_clip_mask,
        "gate_clip_threshold": gate_clip_threshold,
        "jvp_raw": global_jvp_raw,
        "jvp_used": global_jvp_used,
        "jvp_clip_mask": jvp_clip_mask,
        "jvp_clip_threshold": jvp_clip_threshold,
        "kernel_raw": global_kernel_after_jvp,
        "kernel_used": global_kernel_used,
        "kernel_clip_mask": kernel_clip_mask,
        "kernel_clip_threshold": kernel_clip_threshold,
    }


def _candidate_terms_prop2(
    *,
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    lhat: torch.Tensor,
    theta: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power: float,
    gate_power_new: float,
    lhat_temp: float,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    latents: torch.Tensor,
    mu_a: torch.Tensor,
    mean_fwd_y: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
    zeta_old: torch.Tensor,
    delta0: torch.Tensor,
    jd_delta0: torch.Tensor,
    jvp_shrink_alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Two-field specialization of the retained Proposition-2 expansion.

    The implemented forward correction is ``u=rho1*d+rho2*g``, where
    ``g=zeta*d=grad R``.  With ``q=u-g`` and the matched backward drift, the
    retained local log weight is

        A(rho1,rho2)
        -(rho1+rho2*zeta) Delta0' Jd[Delta0]
        +c_theta (rho2-1/2) (d'Delta0)^2.

    The same fixed displacement Delta0 (rho2=0) is used for every candidate,
    as replacing it by the candidate displacement changes only the discarded
    O_p(h^(3/2)) remainder.  ``A`` is evaluated in finite-step DDPM units, so
    its factors already contain h.
    """
    r1 = rho1.float().reshape(-1, 1)
    r2 = rho2.float().reshape(-1, 1)
    dy = d_y.float().flatten(1)
    gy = g_y.float().flatten(1)
    n = int(dy.shape[0])

    u = r1[:, :, None] * dy[None, :, :] + r2[:, :, None] * gy[None, :, :]
    q = u - gy[None, :, :]
    hmu = (mean_fwd_y.float() - latents.float()).flatten(1)
    nu_a = (latents.float() - mu_a.float()).flatten(1)

    forward_linear = -(u * hmu[None, :, :]).sum(dim=2)
    forward_quadratic = (
        -0.5 * forward_variance.float() * u.square().sum(dim=2)
    )
    reverse_linear = (q * nu_a[None, :, :]).sum(dim=2)
    reverse_quadratic = 0.5 * variance.float() * q.square().sum(dim=2)
    forward_alpha = forward_linear + forward_quadratic
    reverse_alpha = reverse_linear + reverse_quadratic

    gate_nu_a = (latents.float() - gate_mu_a.float()).flatten(1)
    gate_nu_b = (latents.float() - gate_mu_b.float()).flatten(1)
    alpha_l = (
        gate_nu_a.square().sum(dim=1)
        - gate_nu_b.square().sum(dim=1)
    ) / (2.0 * variance.float())
    h_alpha_s = (
        float(eta) * float(lhat_temp) * alpha_l
        + math.log(max(float(c_new), 1e-30) / max(float(c), 1e-30))
        + (float(eta_new) - float(eta)) * lhat.float()
    )
    old_log1m = -torch.nn.functional.softplus(
        math.log(max(float(c), 1e-30)) + float(eta) * lhat.float()
    )
    # Proposition 2 states the fixed-m result.  The second term is the
    # first-order correction for the deterministic m_t schedule used here.
    gate_time_1d = (
        -float(gate_power) * theta.float() * h_alpha_s
        + (float(gate_power_new) - float(gate_power)) * old_log1m
    )
    gate_time = gate_time_1d[None, :].expand(int(r1.shape[0]), -1)

    delta = delta0.float().flatten(1)
    jd = jd_delta0.float().flatten(1)
    jd_quadratic = (delta * jd).sum(dim=1)
    d_projection2 = (dy * delta).sum(dim=1).square()
    curvature_coefficient = (
        float(gate_power)
        * float(eta) ** 2
        * float(lhat_temp) ** 2
        * theta.float()
        * (1.0 - theta.float())
    )

    # The -1/2 term is the target curvature in Proposition 2.  The rho2
    # term is the rank-one component of -rho2 Hessian(R).
    target_curvature_1d = -0.5 * curvature_coefficient * d_projection2
    target_curvature = target_curvature_1d[None, :].expand(
        int(r1.shape[0]), -1
    )
    proposal_curvature = (
        r2 * curvature_coefficient[None, :] * d_projection2[None, :]
    )
    jvp_raw = -(
        r1 + r2 * zeta_old.float()[None, :]
    ) * jd_quadratic[None, :]
    jvp = float(jvp_shrink_alpha) * jvp_raw

    gate = gate_time + target_curvature
    forward_zero = forward_alpha + proposal_curvature
    forward = forward_zero + jvp
    total = gate + reverse_alpha + forward
    assert total.shape == (int(rho1.numel()), n)
    return total, {
        "gate": gate,
        "reverse": reverse_alpha,
        "forward": forward,
        "forward_zero": forward_zero,
        "jvp_correction": jvp,
        "gate_time": gate_time,
        "target_curvature": target_curvature,
        "forward_alpha": forward_alpha,
        "reverse_alpha": reverse_alpha,
        "proposal_curvature": proposal_curvature,
        "jvp_raw": jvp_raw,
        "jd_quadratic": jd_quadratic[None, :].expand(int(r1.shape[0]), -1),
        "d_projection2": d_projection2[None, :].expand(int(r1.shape[0]), -1),
        "curvature_coefficient": curvature_coefficient[None, :].expand(
            int(r1.shape[0]), -1
        ),
    }


def _candidate_terms_fixed_field(
    *,
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    lhat: torch.Tensor,
    theta: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power: float,
    gate_power_new: float,
    lhat_temp: float,
    reward_tanh_scale: float,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    forward_scale: float,
    z: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
    d_forward: torch.Tensor | None = None,
    x_ref: torch.Tensor,
    y: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Candidate exact gate/reverse terms plus frozen-current-score K ratio."""
    r1 = rho1[:, None]
    r2 = rho2[:, None]
    dy = d_y.float().flatten(1)
    gy = g_y.float().flatten(1)
    zf = z.float().flatten(1)
    n = int(dy.shape[0])

    dy2 = dy.square().sum(dim=1)
    gy2 = gy.square().sum(dim=1)
    dyg = (dy * gy).sum(dim=1)
    zdy = (zf * dy).sum(dim=1)
    zgy = (zf * gy).sum(dim=1)

    # w = g-u is the reverse-mean correction relative to L_A.
    wd = -r1
    wg = 1.0 - r2
    zw = wd * zdy[None, :] + wg * zgy[None, :]
    w2 = (
        wd.square() * dy2[None, :]
        + wg.square() * gy2[None, :]
        + 2.0 * wd * wg * dyg[None, :]
    )
    reverse = -zw - 0.5 * variance.float() * w2

    # The equal-covariance A/B reverse-kernel log ratio is affine in x.
    logp_a_ref = base.gaussian_log_prob_isotropic(x_ref, gate_mu_a, variance)
    logp_b_ref = base.gaussian_log_prob_isotropic(x_ref, gate_mu_b, variance)
    kernel_lr_ref = float(lhat_temp) * (logp_b_ref - logp_a_ref)
    mean_delta = (gate_mu_b.float() - gate_mu_a.float()).flatten(1)
    u_dot_mean_delta = (
        r1 * (dy * mean_delta).sum(dim=1)[None, :]
        + r2 * (gy * mean_delta).sum(dim=1)[None, :]
    )
    kernel_lr = kernel_lr_ref[None, :] - (
        float(lhat_temp) * u_dot_mean_delta
    )
    lhat_new = lhat.float()[None, :] + kernel_lr
    reward_new, theta_new, zeta_new = _gate_reward_and_zeta(
        lhat_new,
        c=c_new,
        eta=eta_new,
        gate_power=gate_power_new,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    reward_old, _, _ = _gate_reward_and_zeta(
        lhat.float(),
        c=c,
        eta=eta,
        gate_power=gate_power,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    gate = reward_new - reward_old[None, :]

    # Optimizer-only K ratio: freeze the endpoint score field at a supplied
    # reference (initially y; after one refinement, the first selected x).
    df = (d_y if d_forward is None else d_forward).float().flatten(1)
    df2 = df.square().sum(dim=1)
    dydf = (dy * df).sum(dim=1)
    gydf = (gy * df).sum(dim=1)
    rx = r1 + r2 * zeta_new
    u2 = r1.square() * dy2[None, :] + r2.square() * gy2[None, :] + 2.0 * r1 * r2 * dyg[None, :]
    x_shift_dot_df = -variance.float() * (
        r1 * dydf[None, :] + r2 * gydf[None, :]
    )
    resid_ref = (y.float() - float(forward_scale) * x_ref.float()).flatten(1)
    resid_dot_df = (resid_ref * df).sum(dim=1)[None, :] - float(forward_scale) * x_shift_dot_df
    # u_x=(rho1+rho2*zeta_new)d_forward under the frozen-field approximation.
    forward = rx * resid_dot_df - 0.5 * forward_variance.float() * rx.square() * df2[None, :]
    total = gate + reverse + forward
    assert total.shape == (int(rho1.numel()), n)
    return total, {
        "gate": gate,
        "reverse": reverse,
        "forward": forward,
        "kernel_lr": kernel_lr,
        "lhat_new": lhat_new,
        "theta_new": theta_new,
        "zeta_new": zeta_new,
        "u2": u2,
    }


def _fixed_field_sufficient(
    *,
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power: float,
    gate_power_new: float,
    lhat_temp: float,
    reward_tanh_scale: float,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    forward_scale: float,
    z: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
    d_forward: torch.Tensor,
    x_ref: torch.Tensor,
    y: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Particle-sized sufficient statistics for continuous rho optimization."""
    dy = d_y.float().flatten(1)
    gy = g_y.float().flatten(1)
    df = d_forward.float().flatten(1)
    zf = z.float().flatten(1)
    mean_delta = (gate_mu_b.float() - gate_mu_a.float()).flatten(1)
    resid_ref = (y.float() - float(forward_scale) * x_ref.float()).flatten(1)
    logp_a_ref = base.gaussian_log_prob_isotropic(x_ref, gate_mu_a, variance)
    logp_b_ref = base.gaussian_log_prob_isotropic(x_ref, gate_mu_b, variance)
    old_reward, _, _ = _gate_reward_and_zeta(
        lhat.float(),
        c=c,
        eta=eta,
        gate_power=gate_power,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    return {
        "lhat": lhat.float(),
        "old_reward": old_reward,
        "kernel_lr_ref": float(lhat_temp) * (logp_b_ref - logp_a_ref),
        "dy2": dy.square().sum(dim=1),
        "gy2": gy.square().sum(dim=1),
        "dyg": (dy * gy).sum(dim=1),
        "zdy": (zf * dy).sum(dim=1),
        "zgy": (zf * gy).sum(dim=1),
        "dy_mean_delta": (dy * mean_delta).sum(dim=1),
        "gy_mean_delta": (gy * mean_delta).sum(dim=1),
        "df2": df.square().sum(dim=1),
        "dydf": (dy * df).sum(dim=1),
        "gydf": (gy * df).sum(dim=1),
        "resid_df": (resid_ref * df).sum(dim=1),
        "variance": torch.full_like(lhat.float(), float(variance.item())),
        "forward_variance": torch.full_like(
            lhat.float(), float(forward_variance.item())
        ),
        "c_new": float(c_new),
        "eta_new": float(eta_new),
        "gate_power": float(gate_power),
        "gate_power_new": float(gate_power_new),
        "lhat_temp": float(lhat_temp),
        "reward_tanh_scale": float(reward_tanh_scale),
        "forward_scale": float(forward_scale),
    }


def _predicted_from_sufficient(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    suff: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def v(name: str) -> torch.Tensor:
        value = suff[name]
        assert torch.is_tensor(value)
        return value

    variance = v("variance")
    forward_variance = v("forward_variance")
    wd = -rho1
    wg = 1.0 - rho2
    reverse = -(
        wd * v("zdy") + wg * v("zgy")
    ) - 0.5 * variance * (
        wd.square() * v("dy2")
        + wg.square() * v("gy2")
        + 2.0 * wd * wg * v("dyg")
    )
    kernel_lr = v("kernel_lr_ref") - float(suff["lhat_temp"]) * (
        rho1 * v("dy_mean_delta") + rho2 * v("gy_mean_delta")
    )
    lhat_new = v("lhat") + kernel_lr
    reward_new, theta_new, zeta_new = _gate_reward_and_zeta(
        lhat_new,
        c=float(suff["c_new"]),
        eta=float(suff["eta_new"]),
        gate_power=float(suff["gate_power_new"]),
        lhat_temp=float(suff["lhat_temp"]),
        reward_tanh_scale=float(suff["reward_tanh_scale"]),
    )
    gate = reward_new - v("old_reward")
    rx = rho1 + rho2 * zeta_new
    resid_df = v("resid_df") + float(suff["forward_scale"]) * variance * (
        rho1 * v("dydf") + rho2 * v("gydf")
    )
    forward = rx * resid_df - 0.5 * forward_variance * rx.square() * v("df2")
    total = gate + reverse + forward
    return total, {
        "gate": gate,
        "reverse": reverse,
        "forward": forward,
        "theta_new": theta_new,
    }


def _continuous_variance_optimize(
    *,
    mode: str,
    suff: dict[str, torch.Tensor | float],
    carry: torch.Tensor,
    rho1_start: float,
    rho2_start: float,
    fixed_rho1: float,
    c48_kappa: float,
    ridge: float,
    bounded: bool,
    rho1_min: float,
    rho1_max: float,
    rho2_min: float,
    rho2_max: float,
) -> dict[str, Any]:
    """Unbounded BFGS minimization of predicted cumulative-weight variance."""
    from scipy.optimize import minimize

    cpu_suff: dict[str, torch.Tensor | float] = {
        key: (value.detach().cpu().double() if torch.is_tensor(value) else value)
        for key, value in suff.items()
    }
    carry_cpu = carry.detach().cpu().double()

    if mode == "free_continuous":
        starts = [
            np.array([rho1_start, rho2_start]),
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([-2.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
    else:
        zeta_mean = float(
            float(cpu_suff["gate_power_new"])
            * float(cpu_suff["eta_new"])
            * float(cpu_suff["lhat_temp"])
            * torch.sigmoid(
                math.log(max(float(cpu_suff["c_new"]), 1e-30))
                + float(cpu_suff["eta_new"]) * cpu_suff["lhat"]
            ).mean().item()
        )
        if mode == "c48_ratio_continuous":
            denom = zeta_mean - float(c48_kappa)
            effective_two = 2.0 / denom if abs(denom) > 1e-5 else 0.0
            starts = [
                np.array([rho2_start]), np.array([0.0]),
                np.array([1.0]), np.array([-1.0]),
                np.array([effective_two]),
            ]
        elif mode == "fixed_rho1_continuous":
            cancel = -float(fixed_rho1) / max(abs(zeta_mean), 1e-6)
            starts = [
                np.array([rho2_start]), np.array([0.0]),
                np.array([cancel]), np.array([cancel + 0.5]),
                np.array([cancel - 0.5]),
            ]
        else:
            raise ValueError(mode)

    def unpack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "free_continuous":
            return x[0], x[1]
        if mode == "c48_ratio_continuous":
            return -float(c48_kappa) * x[0], x[0]
        return torch.as_tensor(float(fixed_rho1), dtype=x.dtype), x[0]

    def value_and_grad(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        with torch.enable_grad():
            x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
            r1, r2 = unpack(x)
            predicted, _ = _predicted_from_sufficient(r1, r2, cpu_suff)
            total = carry_cpu + predicted
            centered = total - total.mean()
            objective = centered.square().mean() + float(ridge) * (
                r1.square() + r2.square()
            )
            objective.backward()
            return float(objective.item()), x.grad.detach().numpy().astype(float)

    results = []
    if bool(bounded):
        if mode == "free_continuous":
            bounds = [
                (float(rho1_min), float(rho1_max)),
                (float(rho2_min), float(rho2_max)),
            ]
        else:
            bounds = [(float(rho2_min), float(rho2_max))]
        method = "L-BFGS-B"
    else:
        bounds = None
        method = "BFGS"
    for start in starts:
        if bounds is not None:
            start = np.array(
                [
                    np.clip(value, lower, upper)
                    for value, (lower, upper) in zip(start, bounds)
                ],
                dtype=float,
            )
        result = minimize(
            value_and_grad,
            start.astype(float),
            method=method,
            jac=True,
            bounds=bounds,
            options={"maxiter": 100, "gtol": 1e-8},
        )
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            results.append(result)
    if not results:
        raise RuntimeError("All continuous rho optimizations failed")
    best = min(results, key=lambda result: float(result.fun))
    x_best = torch.tensor(best.x, dtype=torch.float64, requires_grad=True)
    rho1_best, rho2_best = unpack(x_best)

    def unregularized(x: torch.Tensor) -> torch.Tensor:
        rr1, rr2 = unpack(x)
        predicted, _ = _predicted_from_sufficient(rr1, rr2, cpu_suff)
        total = carry_cpu + predicted
        return (total - total.mean()).square().mean()

    with torch.enable_grad():
        hessian = torch.autograd.functional.hessian(
            unregularized, x_best
        ).detach().numpy()
    eigen = np.linalg.eigvalsh(np.atleast_2d(hessian))
    predicted, components = _predicted_from_sufficient(
        rho1_best, rho2_best, cpu_suff
    )
    inc_ess = float(_ess_rows(predicted[None, :].float()).item())
    total_ess = float(_ess_rows((carry_cpu + predicted)[None, :].float()).item())
    return {
        "rho1": float(rho1_best.detach().item()),
        "rho2": float(rho2_best.detach().item()),
        "objective": float(unregularized(x_best).item()),
        "objective_regularized": float(best.fun),
        "iterations": int(best.nit),
        "success": bool(best.success),
        "message": str(best.message),
        "grad_norm": float(np.linalg.norm(best.jac)),
        "hessian_eigen_min": float(eigen.min()),
        "hessian_eigen_max": float(eigen.max()),
        "predicted_inc_ess": inc_ess,
        "predicted_total_ess": total_ess,
        "predicted": predicted.float(),
        "components": {key: value.float() for key, value in components.items()},
    }


def _exact_selected_step(
    *,
    pipe,
    prompt_embeds: torch.Tensor,
    timestep: int,
    rho1: float,
    rho2: float,
    latents: torch.Tensor,
    lhat: torch.Tensor,
    theta: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power: float,
    gate_power_new: float,
    lhat_temp: float,
    reward_tanh_scale: float,
    mu_a: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    variance: torch.Tensor,
    forward_variance: torch.Tensor,
    z: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    u_y = float(rho1) * d_y.float() + float(rho2) * g_y.float()
    w_y = g_y.float() - u_y
    x_new = mu_a.float() + variance.float() * w_y + z.float()

    logp_a = base.gaussian_log_prob_isotropic(x_new, gate_mu_a, variance)
    logp_b = base.gaussian_log_prob_isotropic(x_new, gate_mu_b, variance)
    kernel_lr = float(lhat_temp) * (logp_b - logp_a)
    lhat_new = lhat.float() + kernel_lr
    reward_new, theta_new, zeta_new = _gate_reward_and_zeta(
        lhat_new,
        c=c_new,
        eta=eta_new,
        gate_power=gate_power_new,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    reward_old, _, _ = _gate_reward_and_zeta(
        lhat.float(),
        c=c,
        eta=eta,
        gate_power=gate_power,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    gate = reward_new - reward_old
    logp_prop = base.gaussian_log_prob_isotropic(
        x_new, mu_a.float() + variance.float() * w_y, variance
    )
    reverse = logp_a - logp_prop

    mean_fwd_a, var_fwd = base.forward_kernel_mean_variance(
        pipe.scheduler, int(timestep), x_new
    )
    prev_t = int(pipe.scheduler.previous_timestep(int(timestep)))
    if prev_t >= 0:
        _, _, _, beta_x = base.ab_outputs_and_beta(
            pipe,
            x_new.to(dtype=latents.dtype),
            prev_t,
            prompt_embeds,
            linear_baseline_kappa=0.0,
        )
        d_x = -beta_x.float()
        g_x = zeta_new[:, None, None, None] * d_x
        u_x = float(rho1) * d_x + float(rho2) * g_x
        mean_fwd_prop = mean_fwd_a.float() + var_fwd.float() * u_x
        logk_a = base.gaussian_log_prob_isotropic(latents, mean_fwd_a, var_fwd)
        logk_prop = base.gaussian_log_prob_isotropic(latents, mean_fwd_prop, var_fwd)
        forward = logk_prop - logk_a
    else:
        d_x = torch.zeros_like(d_y.float())
        u_x = torch.zeros_like(d_y.float())
        zeta_new = torch.zeros_like(theta_new)
        forward = torch.zeros_like(gate)
    total = gate + reverse + forward
    return x_new.to(dtype=latents.dtype), lhat_new, total.float(), {
        "gate": gate.float(),
        "reverse": reverse.float(),
        "forward": forward.float(),
        "forward_zero": forward.float(),
        "jvp_correction": torch.zeros_like(forward).float(),
        "kernel_lr": kernel_lr.float(),
        "theta_new": theta_new.float(),
        "zeta_new": zeta_new.float(),
        "d_x": d_x.float(),
        "u_x": u_x.float(),
        "u_y": u_y.float(),
        "w_y": w_y.float(),
    }


def _jvp_selected_step(
    *,
    pipe,
    timestep: int,
    rho1: float,
    rho2: float,
    latents: torch.Tensor,
    lhat: torch.Tensor,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    gate_power: float,
    gate_power_new: float,
    lhat_temp: float,
    reward_tanh_scale: float,
    mu_a: torch.Tensor,
    gate_mu_a: torch.Tensor,
    gate_mu_b: torch.Tensor,
    variance: torch.Tensor,
    z: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
    d_forward: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Selected finite gate/reverse weight with a JVP endpoint-score surrogate."""
    u_y = float(rho1) * d_y.float() + float(rho2) * g_y.float()
    w_y = g_y.float() - u_y
    x_new = mu_a.float() + variance.float() * w_y + z.float()

    logp_a = base.gaussian_log_prob_isotropic(x_new, gate_mu_a, variance)
    logp_b = base.gaussian_log_prob_isotropic(x_new, gate_mu_b, variance)
    kernel_lr = float(lhat_temp) * (logp_b - logp_a)
    lhat_new = lhat.float() + kernel_lr
    reward_new, theta_new, zeta_new = _gate_reward_and_zeta(
        lhat_new,
        c=c_new,
        eta=eta_new,
        gate_power=gate_power_new,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    reward_old, _, _ = _gate_reward_and_zeta(
        lhat.float(),
        c=c,
        eta=eta,
        gate_power=gate_power,
        lhat_temp=lhat_temp,
        reward_tanh_scale=reward_tanh_scale,
    )
    gate = reward_new - reward_old
    logp_prop = base.gaussian_log_prob_isotropic(
        x_new, mu_a.float() + variance.float() * w_y, variance
    )
    reverse = logp_a - logp_prop

    mean_fwd_a, var_fwd = base.forward_kernel_mean_variance(
        pipe.scheduler, int(timestep), x_new
    )
    prev_t = int(pipe.scheduler.previous_timestep(int(timestep)))
    if prev_t >= 0:
        d_x = d_forward.float()
        g_x = zeta_new[:, None, None, None] * d_x
        u_x = float(rho1) * d_x + float(rho2) * g_x
        mean_fwd_prop = mean_fwd_a.float() + var_fwd.float() * u_x
        logk_a = base.gaussian_log_prob_isotropic(latents, mean_fwd_a, var_fwd)
        logk_prop = base.gaussian_log_prob_isotropic(
            latents, mean_fwd_prop, var_fwd
        )
        forward = logk_prop - logk_a

        # The zero-order counterpart isolates the contribution introduced by
        # J d_y[Delta x_0], allowing the same particle-level robustification
        # in proposal selection and propagated weights.
        g_x_zero = zeta_new[:, None, None, None] * d_y.float()
        u_x_zero = float(rho1) * d_y.float() + float(rho2) * g_x_zero
        mean_fwd_zero = mean_fwd_a.float() + var_fwd.float() * u_x_zero
        logk_zero = base.gaussian_log_prob_isotropic(
            latents, mean_fwd_zero, var_fwd
        )
        forward_zero = logk_zero - logk_a
    else:
        d_x = torch.zeros_like(d_y.float())
        u_x = torch.zeros_like(d_y.float())
        zeta_new = torch.zeros_like(theta_new)
        forward = torch.zeros_like(gate)
        forward_zero = torch.zeros_like(gate)
    jvp_correction = forward - forward_zero
    total = gate + reverse + forward
    return x_new.to(dtype=latents.dtype), lhat_new, total.float(), {
        "gate": gate.float(),
        "reverse": reverse.float(),
        "forward": forward.float(),
        "forward_zero": forward_zero.float(),
        "jvp_correction": jvp_correction.float(),
        "kernel_lr": kernel_lr.float(),
        "theta_new": theta_new.float(),
        "zeta_new": zeta_new.float(),
        "d_x": d_x.float(),
        "u_x": u_x.float(),
        "u_y": u_y.float(),
        "w_y": w_y.float(),
    }


def worker(rank: int, args: argparse.Namespace, devices: list[str]) -> None:
    world_size = len(devices)
    device = devices[rank]
    torch.cuda.set_device(torch.device(device))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        prompt, negative, _ = base.resolve_prompt(args)
        local_n = args.n_particles // world_size
        global_start = rank * local_n

        pipe = base.load_sd_pipeline(device, args.model_id)
        pipe.unet.to(dtype=torch.float32)
        pipe.vae.to(dtype=torch.float32)
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(dtype=torch.float32)
        from diffusers.models.attention_processor import AttnProcessor
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.enable_attention_slicing(args.attention_slicing)
        pipe._dng_unet_chunk_size = int(args.unet_chunk_size)
        pipe._pa_gate_cfg_scale = 1.0
        pipe._pa_gate_base_lambda_neg = 0.0
        pipe._pa_gate_linear_baseline_kappa = 0.0
        pipe._pa_gate_gate_ratio_mode = "raw"

        uncond = base.encode_text(pipe, [""] * local_n, device)
        pos = base.encode_text(pipe, [prompt["positive"]] * local_n, device)
        neg = base.encode_text(pipe, [negative] * local_n, device)
        prompt_embeds = torch.cat([uncond, pos, neg], dim=0)

        pipe.scheduler.set_timesteps(args.steps, device=device)
        timesteps = list(pipe.scheduler.timesteps)
        height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        init_generators = [
            torch.Generator(device=device).manual_seed(args.seed + global_start + i)
            for i in range(local_n)
        ]
        noise_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + rank)
        resample_gen = torch.Generator(device=device).manual_seed(args.seed + 200000)
        latents = pipe.prepare_latents(
            local_n, pipe.unet.config.in_channels, height, width,
            pos.dtype, device, init_generators, latents=None,
        )
        lhat = torch.zeros(local_n, device=device, dtype=torch.float32)
        logw_total = torch.zeros_like(lhat)
        origin_ids = torch.arange(
            global_start, global_start + local_n, device=device, dtype=torch.long
        )
        cand_rho1, cand_rho2 = candidate_grid(args, device)
        rho1_current = (
            float(args.fixed_rho1)
            if args.mode in {"fixed_rho1", "fixed_rho1_continuous"}
            else 0.0
        )
        rho2_current = 0.0

        rows: list[dict[str, Any]] = []
        particle_history: dict[str, list[np.ndarray]] = {key: [] for key in PARTICLE_FIELDS}
        optimizer_inc_history: list[np.ndarray] = []
        optimizer_total_history: list[np.ndarray] = []
        optimizer_initial_inc_history: list[np.ndarray] = []
        optimizer_initial_total_history: list[np.ndarray] = []
        optimizer_variance_history: list[np.ndarray] = []
        optimizer_initial_variance_history: list[np.ndarray] = []
        continuous_optimizer_history: list[dict[str, Any]] = []
        start_time = time.time()
        early_stopped = False
        last_resample_step = -10**9

        with torch.no_grad():
            for step, timestep in enumerate(timesteps):
                t0 = time.time()
                t_int = int(timestep)
                c, eta = focused_schedule_at_state(
                    step, len(timesteps), args
                )
                c_new, eta_new = focused_schedule_at_state(
                    step + 1, len(timesteps), args
                )
                # A deterministic power schedule changes only the intermediate
                # target path.  The exact gate increment below retains the
                # resulting change of measure, and every supported schedule
                # reaches ``args.gate_power`` at the terminal state.
                gate_power = base.gate_power_at_state(
                    step, len(timesteps), args
                )
                gate_power_new = base.gate_power_at_state(
                    step + 1, len(timesteps), args
                )
                fixed_rho1_step = fixed_rho1_at_state(
                    step, len(timesteps), args
                )
                if args.mode in {"fixed_rho1", "fixed_rho1_continuous"}:
                    rho1_current = fixed_rho1_step

                out_a, out_gate_a, out_gate_b, beta_l = base.ab_outputs_and_beta(
                    pipe, latents, timestep, prompt_embeds,
                    linear_baseline_kappa=0.0,
                )
                mu_a, variance = base.ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_a, t_int, latents
                )
                gate_mu_a, _ = base.ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_gate_a, t_int, latents
                )
                gate_mu_b, _ = base.ddpm_reverse_mean_and_variance(
                    pipe.scheduler, out_gate_b, t_int, latents
                )
                variance = variance.float()
                mean_fwd_y, forward_variance = base.forward_kernel_mean_variance(
                    pipe.scheduler, t_int, latents
                )
                prev_t = int(pipe.scheduler.previous_timestep(t_int))
                if prev_t >= 0:
                    alpha_t = pipe.scheduler.alphas_cumprod[t_int].float()
                    alpha_prev = pipe.scheduler.alphas_cumprod[prev_t].float()
                    forward_scale = float(torch.sqrt(alpha_t / alpha_prev).item())
                else:
                    forward_scale = 1.0

                d_y = -beta_l.float()
                _, theta, zeta_old = _gate_reward_and_zeta(
                    lhat,
                    c=c,
                    eta=eta,
                    gate_power=gate_power,
                    lhat_temp=args.lhat_temp,
                    reward_tanh_scale=args.reward_tanh_scale,
                )
                g_y = zeta_old[:, None, None, None] * d_y
                noise = torch.randn(
                    latents.shape, device=device, dtype=latents.dtype,
                    generator=noise_gen,
                )
                z = variance.sqrt().to(latents.dtype) * noise
                x_ref = mu_a.float() + variance * g_y + z.float()

                skipped = base.should_skip_smc_weight(
                    pipe.scheduler, t_int, variance, args.min_weight_variance
                )
                optimize = (
                    not skipped
                    and step < len(timesteps) - int(args.no_rho_last_steps)
                )
                d_forward_opt = d_y.float()
                jvp_used = "none"
                jvp_delta0 = torch.zeros_like(d_y.float())
                jvp_noise = torch.zeros_like(d_y.float())
                delta_x0 = (
                    mu_a.float() - latents.float()
                    + variance.float()
                    * (-float(rho1_current) * d_y.float() + g_y.float())
                    + z.float()
                )
                needs_jvp = (
                    not skipped
                    and prev_t >= 0
                    and (
                        (
                            optimize
                            and (
                                args.optimizer_field_mode == "jvp"
                                or args.optimizer_weight_mode == "prop2"
                            )
                        )
                        or args.prop_weight_mode in {"jvp_surrogate", "prop2"}
                    )
                )
                if needs_jvp:
                    # One fixed displacement shared by every rho2 candidate:
                    # x_0-y at rho2=0 and the current fixed rho1.  The score
                    # difference is d=-beta_l, hence Jd[delta]=-Jbeta[delta].
                    jbeta_delta0, jvp_used = base.compute_jbeta_z(
                        pipe,
                        latents,
                        timestep,
                        prompt_embeds,
                        delta_x0,
                        mode=args.jvp_mode,
                        fd_eps=args.jvp_eps,
                        rank=rank,
                    )
                    jvp_delta0 = -jbeta_delta0.float()
                    if args.diagnose_noise_only_jvp:
                        jbeta_noise, _ = base.compute_jbeta_z(
                            pipe,
                            latents,
                            timestep,
                            prompt_embeds,
                            z,
                            mode=args.jvp_mode,
                            fd_eps=args.jvp_eps,
                            rank=rank,
                        )
                        jvp_noise = -jbeta_noise.float()
                    d_forward_opt = (
                        d_y.float()
                        + float(args.jvp_shrink_alpha) * jvp_delta0
                    )
                pred_selected = torch.zeros_like(lhat)
                prop2_selected_components = {
                    key: torch.zeros_like(lhat)
                    for key in (
                        "gate_time", "target_curvature", "forward_alpha",
                        "reverse_alpha", "proposal_curvature", "jvp_raw",
                        "jd_quadratic", "d_projection2",
                        "curvature_coefficient",
                    )
                }
                prop2_selected_raw = torch.zeros_like(lhat)
                prop2_noise_selected_raw = torch.zeros_like(lhat)
                finite_diag = {
                    key: torch.full_like(lhat, float("nan"))
                    for key in (
                        "exact_total", "exact_gate", "exact_reverse",
                        "exact_forward", "no_jvp_total", "jvp_total",
                        "forward_zero", "jvp_forward",
                        "true_jvp_correction", "est_jvp_correction",
                    )
                }
                surface_nan = np.full(
                    int(cand_rho1.numel()), np.nan, dtype=np.float32
                )
                rho1_initial = rho1_current
                rho2_initial = rho2_current
                continuous_mode = args.mode.endswith("_continuous")
                continuous_diag: dict[str, Any] = {}
                if optimize and continuous_mode:
                    local_suff = _fixed_field_sufficient(
                        lhat=lhat,
                        c=c,
                        eta=eta,
                        c_new=c_new,
                        eta_new=eta_new,
                        gate_power=gate_power,
                        gate_power_new=gate_power_new,
                        lhat_temp=args.lhat_temp,
                        reward_tanh_scale=args.reward_tanh_scale,
                        variance=variance,
                        forward_variance=forward_variance,
                        forward_scale=forward_scale,
                        z=z,
                        d_y=d_y,
                        g_y=g_y,
                        d_forward=d_y,
                        x_ref=x_ref,
                        y=latents,
                        gate_mu_a=gate_mu_a,
                        gate_mu_b=gate_mu_b,
                    )
                    global_suff = _gather_sufficient(local_suff, world_size)
                    global_carry = base.all_gather_cat(
                        logw_total.float(), world_size
                    )
                    continuous_diag = _continuous_variance_optimize(
                        mode=args.mode,
                        suff=global_suff,
                        carry=global_carry,
                        rho1_start=rho1_current,
                        rho2_start=rho2_current,
                        fixed_rho1=fixed_rho1_step,
                        c48_kappa=args.c48_kappa,
                        ridge=args.rho_ridge,
                        bounded=args.rho_continuous_bounded,
                        rho1_min=args.rho1_min,
                        rho1_max=args.rho1_max,
                        rho2_min=args.rho2_min,
                        rho2_max=args.rho2_max,
                    )
                    rho1_current = float(continuous_diag["rho1"])
                    rho2_current = float(continuous_diag["rho2"])
                    rho1_initial = rho1_current
                    rho2_initial = rho2_current
                    pred_selected, _ = _predicted_from_sufficient(
                        torch.tensor(rho1_current, device=device),
                        torch.tensor(rho2_current, device=device),
                        local_suff,
                    )
                    if rank == 0:
                        optimizer_initial_inc_history.append(surface_nan.copy())
                        optimizer_initial_total_history.append(surface_nan.copy())
                        optimizer_initial_variance_history.append(
                            surface_nan.copy()
                        )
                elif optimize:
                    if args.optimizer_weight_mode == "prop2":
                        local_matrix, local_candidate_components = (
                            _candidate_terms_prop2(
                                rho1=cand_rho1,
                                rho2=cand_rho2,
                                lhat=lhat,
                                theta=theta,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power=gate_power,
                                gate_power_new=gate_power_new,
                                lhat_temp=args.lhat_temp,
                                variance=variance,
                                forward_variance=forward_variance,
                                latents=latents,
                                mu_a=mu_a,
                                mean_fwd_y=mean_fwd_y,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                                d_y=d_y,
                                g_y=g_y,
                                zeta_old=zeta_old,
                                delta0=jvp_delta0 * 0.0 + delta_x0,
                                jd_delta0=jvp_delta0,
                                jvp_shrink_alpha=args.jvp_shrink_alpha,
                            )
                        )
                        local_zero_components = {
                            **local_candidate_components,
                            "forward": local_candidate_components["forward_zero"],
                        }
                        global_matrix, _ = _candidate_matrix_after_gate_and_jvp_clip(
                            local_matrix,
                            local_candidate_components,
                            local_zero_components,
                            world_size=world_size,
                            gate_topk=args.gate_logw_rank_clip_topk,
                            gate_max_abs=args.gate_logw_clip,
                            jvp_topk=args.jvp_logw_rank_clip_topk,
                            jvp_max_abs=args.jvp_logw_clip,
                            kernel_topk=args.kernel_logw_rank_clip_topk,
                            kernel_max_abs=args.kernel_logw_clip,
                        )
                    else:
                        local_matrix, local_candidate_components = _candidate_terms_fixed_field(
                            rho1=cand_rho1,
                            rho2=cand_rho2,
                            lhat=lhat,
                            theta=theta,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_temp=args.lhat_temp,
                            reward_tanh_scale=args.reward_tanh_scale,
                            variance=variance,
                            forward_variance=forward_variance,
                            forward_scale=forward_scale,
                            z=z,
                            d_y=d_y,
                            g_y=g_y,
                            d_forward=d_forward_opt,
                            x_ref=x_ref,
                            y=latents,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                        )
                    if (
                        args.optimizer_weight_mode != "prop2"
                        and args.optimizer_field_mode == "jvp"
                    ):
                        _, local_zero_components = _candidate_terms_fixed_field(
                            rho1=cand_rho1,
                            rho2=cand_rho2,
                            lhat=lhat,
                            theta=theta,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_temp=args.lhat_temp,
                            reward_tanh_scale=args.reward_tanh_scale,
                            variance=variance,
                            forward_variance=forward_variance,
                            forward_scale=forward_scale,
                            z=z,
                            d_y=d_y,
                            g_y=g_y,
                            d_forward=d_y,
                            x_ref=x_ref,
                            y=latents,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                        )
                        global_matrix, _ = _candidate_matrix_after_gate_and_jvp_clip(
                            local_matrix,
                            local_candidate_components,
                            local_zero_components,
                            world_size=world_size,
                            gate_topk=args.gate_logw_rank_clip_topk,
                            gate_max_abs=args.gate_logw_clip,
                            jvp_topk=args.jvp_logw_rank_clip_topk,
                            jvp_max_abs=args.jvp_logw_clip,
                            kernel_topk=args.kernel_logw_rank_clip_topk,
                            kernel_max_abs=args.kernel_logw_clip,
                        )
                    elif args.optimizer_weight_mode != "prop2":
                        global_matrix, _ = _candidate_matrix_after_gate_clip(
                            local_matrix,
                            local_candidate_components,
                            world_size=world_size,
                            topk=args.gate_logw_rank_clip_topk,
                            max_abs=args.gate_logw_clip,
                        )
                    global_carry = base.all_gather_cat(logw_total.float(), world_size)
                    (
                        inc_ess_candidates,
                        total_ess_candidates,
                        total_logw_variance_candidates,
                    ) = _rho_grid_diagnostics(global_matrix, global_carry)
                    best = _select_rho_grid_index(
                        objective=args.rho2_grid_objective,
                        total_ess=total_ess_candidates,
                        total_logw_variance=total_logw_variance_candidates,
                    )
                    rho1_current = float(cand_rho1[best].item())
                    rho2_current = float(cand_rho2[best].item())
                    rho1_initial = rho1_current
                    rho2_initial = rho2_current
                    pred_selected = global_matrix[
                        best, global_start:global_start + local_n
                    ]
                    if rank == 0:
                        optimizer_initial_inc_history.append(
                            inc_ess_candidates.cpu().numpy()
                        )
                        optimizer_initial_total_history.append(
                            total_ess_candidates.cpu().numpy()
                        )
                        optimizer_initial_variance_history.append(
                            total_logw_variance_candidates.cpu().numpy()
                        )
                else:
                    if rank == 0:
                        optimizer_initial_inc_history.append(surface_nan.copy())
                        optimizer_initial_total_history.append(surface_nan.copy())
                        optimizer_initial_variance_history.append(
                            surface_nan.copy()
                        )

                if (
                    not skipped
                    and (
                        args.optimizer_weight_mode == "prop2"
                        or args.prop_weight_mode == "prop2"
                    )
                ):
                    prop2_selected_matrix, prop2_components_matrix = (
                        _candidate_terms_prop2(
                            rho1=torch.tensor([rho1_current], device=device),
                            rho2=torch.tensor([rho2_current], device=device),
                            lhat=lhat,
                            theta=theta,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_temp=args.lhat_temp,
                            variance=variance,
                            forward_variance=forward_variance,
                            latents=latents,
                            mu_a=mu_a,
                            mean_fwd_y=mean_fwd_y,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            d_y=d_y,
                            g_y=g_y,
                            zeta_old=zeta_old,
                            delta0=jvp_delta0 * 0.0 + delta_x0,
                            jd_delta0=jvp_delta0,
                            jvp_shrink_alpha=args.jvp_shrink_alpha,
                        )
                    )
                    prop2_selected_raw = prop2_selected_matrix[0]
                    prop2_selected_components = {
                        key: value[0]
                        for key, value in prop2_components_matrix.items()
                        if key in prop2_selected_components
                    }
                    if args.diagnose_noise_only_jvp:
                        prop2_noise_matrix, _ = _candidate_terms_prop2(
                            rho1=torch.tensor([rho1_current], device=device),
                            rho2=torch.tensor([rho2_current], device=device),
                            lhat=lhat,
                            theta=theta,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_temp=args.lhat_temp,
                            variance=variance,
                            forward_variance=forward_variance,
                            latents=latents,
                            mu_a=mu_a,
                            mean_fwd_y=mean_fwd_y,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            d_y=d_y,
                            g_y=g_y,
                            zeta_old=zeta_old,
                            delta0=z.float(),
                            jd_delta0=jvp_noise,
                            jvp_shrink_alpha=args.jvp_shrink_alpha,
                        )
                        prop2_noise_selected_raw = prop2_noise_matrix[0]

                if skipped:
                    x_new = mu_a.to(dtype=latents.dtype)
                    lhat_new = lhat
                    logw = torch.zeros_like(lhat)
                    exact = {
                        "gate": torch.zeros_like(lhat),
                        "reverse": torch.zeros_like(lhat),
                        "forward": torch.zeros_like(lhat),
                        "forward_zero": torch.zeros_like(lhat),
                        "jvp_correction": torch.zeros_like(lhat),
                        "kernel_lr": torch.zeros_like(lhat),
                        "theta_new": theta,
                        "zeta_new": torch.zeros_like(theta),
                        "d_x": torch.zeros_like(d_y),
                        "u_x": torch.zeros_like(d_y),
                        "u_y": torch.zeros_like(d_y),
                        "w_y": torch.zeros_like(d_y),
                    }
                elif args.prop_weight_mode in {"jvp_surrogate", "prop2"}:
                    x_new, lhat_new, logw, exact = _jvp_selected_step(
                        pipe=pipe,
                        timestep=t_int,
                        rho1=rho1_current,
                        rho2=rho2_current,
                        latents=latents,
                        lhat=lhat,
                        c=c,
                        eta=eta,
                        c_new=c_new,
                        eta_new=eta_new,
                        gate_power=gate_power,
                        gate_power_new=gate_power_new,
                        lhat_temp=args.lhat_temp,
                        reward_tanh_scale=args.reward_tanh_scale,
                        mu_a=mu_a,
                        gate_mu_a=gate_mu_a,
                        gate_mu_b=gate_mu_b,
                        variance=variance,
                        z=z,
                        d_y=d_y,
                        g_y=g_y,
                        d_forward=d_forward_opt,
                    )
                    # Optional diagnostic only: evaluate the selected endpoint
                    # score once and compare the finite Gaussian forward term
                    # with and without the fixed-displacement JVP.  The
                    # diagnostic reuses the already sampled endpoint and does
                    # not alter the propagated state or weight.
                    if bool(args.diagnose_finite_exact_weight) and prev_t >= 0:
                        jvp_gate = exact["gate"].float()
                        jvp_reverse = exact["reverse"].float()
                        jvp_forward = exact["forward"].float()
                        forward_zero = exact["forward_zero"].float()
                        (
                            x_diag,
                            lhat_diag,
                            exact_diag_total,
                            exact_diag,
                        ) = _exact_selected_step(
                            pipe=pipe,
                            prompt_embeds=prompt_embeds,
                            timestep=t_int,
                            rho1=rho1_current,
                            rho2=rho2_current,
                            latents=latents,
                            lhat=lhat,
                            theta=theta,
                            c=c,
                            eta=eta,
                            c_new=c_new,
                            eta_new=eta_new,
                            gate_power=gate_power,
                            gate_power_new=gate_power_new,
                            lhat_temp=args.lhat_temp,
                            reward_tanh_scale=args.reward_tanh_scale,
                            mu_a=mu_a,
                            gate_mu_a=gate_mu_a,
                            gate_mu_b=gate_mu_b,
                            variance=variance,
                            forward_variance=forward_variance,
                            z=z,
                            d_y=d_y,
                            g_y=g_y,
                        )
                        if not torch.allclose(
                            x_diag.float(), x_new.float(), rtol=1e-5, atol=1e-5
                        ):
                            raise RuntimeError(
                                "Finite diagnostic changed the selected endpoint"
                            )
                        if not torch.allclose(
                            lhat_diag.float(), lhat_new.float(),
                            rtol=1e-5, atol=1e-5,
                        ):
                            raise RuntimeError(
                                "Finite diagnostic changed the likelihood-ratio state"
                            )
                        finite_diag = {
                            "exact_total": exact_diag_total.float(),
                            "exact_gate": exact_diag["gate"].float(),
                            "exact_reverse": exact_diag["reverse"].float(),
                            "exact_forward": exact_diag["forward"].float(),
                            "no_jvp_total": (
                                jvp_gate + jvp_reverse + forward_zero
                            ),
                            "jvp_total": (
                                jvp_gate + jvp_reverse + jvp_forward
                            ),
                            "forward_zero": forward_zero,
                            "jvp_forward": jvp_forward,
                            "true_jvp_correction": (
                                exact_diag["forward"].float() - forward_zero
                            ),
                            "est_jvp_correction": (
                                jvp_forward - forward_zero
                            ),
                        }
                    if args.prop_weight_mode == "prop2":
                        logw = prop2_selected_raw.float()
                        exact["gate"] = prop2_selected_components[
                            "gate_time"
                        ] + prop2_selected_components["target_curvature"]
                        exact["reverse"] = prop2_selected_components[
                            "reverse_alpha"
                        ]
                        exact["forward_zero"] = prop2_selected_components[
                            "forward_alpha"
                        ] + prop2_selected_components["proposal_curvature"]
                        exact["jvp_correction"] = (
                            float(args.jvp_shrink_alpha)
                            * prop2_selected_components["jvp_raw"]
                        )
                        exact["forward"] = (
                            exact["forward_zero"] + exact["jvp_correction"]
                        )
                else:
                    x_new, lhat_new, logw, exact = _exact_selected_step(
                        pipe=pipe,
                        prompt_embeds=prompt_embeds,
                        timestep=t_int,
                        rho1=rho1_current,
                        rho2=rho2_current,
                        latents=latents,
                        lhat=lhat,
                        theta=theta,
                        c=c,
                        eta=eta,
                        c_new=c_new,
                        eta_new=eta_new,
                        gate_power=gate_power,
                        gate_power_new=gate_power_new,
                        lhat_temp=args.lhat_temp,
                        reward_tanh_scale=args.reward_tanh_scale,
                        mu_a=mu_a,
                        gate_mu_a=gate_mu_a,
                        gate_mu_b=gate_mu_b,
                        variance=variance,
                        forward_variance=forward_variance,
                        z=z,
                        d_y=d_y,
                        g_y=g_y,
                    )
                    # One fixed-point refinement substantially reduces the
                    # optimizer/true-weight mismatch: freeze the forward score
                    # at the first selected endpoint, re-optimize, then perform
                    # a fresh endpoint evaluation for the final exact weight.
                    for _ in range(int(args.optimizer_refine_passes) if optimize else 0):
                        if continuous_mode:
                            local_suff = _fixed_field_sufficient(
                                lhat=lhat,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power=gate_power,
                                gate_power_new=gate_power_new,
                                lhat_temp=args.lhat_temp,
                                reward_tanh_scale=args.reward_tanh_scale,
                                variance=variance,
                                forward_variance=forward_variance,
                                forward_scale=forward_scale,
                                z=z,
                                d_y=d_y,
                                g_y=g_y,
                                d_forward=exact["d_x"],
                                x_ref=x_ref,
                                y=latents,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                            )
                            global_suff = _gather_sufficient(
                                local_suff, world_size
                            )
                            global_carry = base.all_gather_cat(
                                logw_total.float(), world_size
                            )
                            continuous_diag = _continuous_variance_optimize(
                                mode=args.mode,
                                suff=global_suff,
                                carry=global_carry,
                                rho1_start=rho1_current,
                                rho2_start=rho2_current,
                                fixed_rho1=fixed_rho1_step,
                                c48_kappa=args.c48_kappa,
                                ridge=args.rho_ridge,
                                bounded=args.rho_continuous_bounded,
                                rho1_min=args.rho1_min,
                                rho1_max=args.rho1_max,
                                rho2_min=args.rho2_min,
                                rho2_max=args.rho2_max,
                            )
                            rho1_refined = float(continuous_diag["rho1"])
                            rho2_refined = float(continuous_diag["rho2"])
                            pred_selected, _ = _predicted_from_sufficient(
                                torch.tensor(rho1_refined, device=device),
                                torch.tensor(rho2_refined, device=device),
                                local_suff,
                            )
                        else:
                            local_matrix, local_candidate_components = _candidate_terms_fixed_field(
                                rho1=cand_rho1,
                                rho2=cand_rho2,
                                lhat=lhat,
                                theta=theta,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power=gate_power,
                                gate_power_new=gate_power_new,
                                lhat_temp=args.lhat_temp,
                                reward_tanh_scale=args.reward_tanh_scale,
                                variance=variance,
                                forward_variance=forward_variance,
                                forward_scale=forward_scale,
                                z=z,
                                d_y=d_y,
                                g_y=g_y,
                                d_forward=exact["d_x"],
                                x_ref=x_ref,
                                y=latents,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                            )
                            global_matrix, _ = _candidate_matrix_after_gate_clip(
                                local_matrix,
                                local_candidate_components,
                                world_size=world_size,
                                topk=args.gate_logw_rank_clip_topk,
                                max_abs=args.gate_logw_clip,
                            )
                            global_carry = base.all_gather_cat(
                                logw_total.float(), world_size
                            )
                            (
                                inc_ess_candidates,
                                total_ess_candidates,
                                total_logw_variance_candidates,
                            ) = _rho_grid_diagnostics(
                                global_matrix, global_carry
                            )
                            best = _select_rho_grid_index(
                                objective=args.rho2_grid_objective,
                                total_ess=total_ess_candidates,
                                total_logw_variance=(
                                    total_logw_variance_candidates
                                ),
                            )
                            rho1_refined = float(cand_rho1[best].item())
                            rho2_refined = float(cand_rho2[best].item())
                            pred_selected = global_matrix[
                                best, global_start:global_start + local_n
                            ]
                        changed = (
                            abs(rho1_refined - rho1_current) > 1e-12
                            or abs(rho2_refined - rho2_current) > 1e-12
                        )
                        rho1_current = rho1_refined
                        rho2_current = rho2_refined
                        if changed:
                            x_new, lhat_new, logw, exact = _exact_selected_step(
                                pipe=pipe,
                                prompt_embeds=prompt_embeds,
                                timestep=t_int,
                                rho1=rho1_current,
                                rho2=rho2_current,
                                latents=latents,
                                lhat=lhat,
                                theta=theta,
                                c=c,
                                eta=eta,
                                c_new=c_new,
                                eta_new=eta_new,
                                gate_power=gate_power,
                                gate_power_new=gate_power_new,
                                lhat_temp=args.lhat_temp,
                                reward_tanh_scale=args.reward_tanh_scale,
                                mu_a=mu_a,
                                gate_mu_a=gate_mu_a,
                                gate_mu_b=gate_mu_b,
                                variance=variance,
                                forward_variance=forward_variance,
                                z=z,
                                d_y=d_y,
                                g_y=g_y,
                            )

                if rank == 0:
                    if optimize and not continuous_mode:
                        optimizer_inc_history.append(
                            inc_ess_candidates.cpu().numpy()
                        )
                        optimizer_total_history.append(
                            total_ess_candidates.cpu().numpy()
                        )
                        optimizer_variance_history.append(
                            total_logw_variance_candidates.cpu().numpy()
                        )
                    else:
                        optimizer_inc_history.append(surface_nan.copy())
                        optimizer_total_history.append(surface_nan.copy())
                        optimizer_variance_history.append(surface_nan.copy())
                    if optimize and continuous_mode:
                        continuous_optimizer_history.append(
                            {
                                key: value
                                for key, value in continuous_diag.items()
                                if key not in {"predicted", "components"}
                            }
                        )
                    else:
                        continuous_optimizer_history.append({})

                global_logw_raw = base.all_gather_cat(logw.float(), world_size)
                global_gate_raw = base.all_gather_cat(
                    exact["gate"].float(), world_size
                )
                global_reverse = base.all_gather_cat(
                    exact["reverse"].float(), world_size
                )
                global_forward = base.all_gather_cat(
                    exact["forward"].float(), world_size
                )
                global_forward_zero = base.all_gather_cat(
                    exact["forward_zero"].float(), world_size
                )
                global_jvp_raw = base.all_gather_cat(
                    exact["jvp_correction"].float(), world_size
                )
                (
                    global_gate_used,
                    gate_clip_mask,
                    gate_clip_threshold,
                ) = _rank_winsorize_rows(
                    global_gate_raw,
                    args.gate_logw_rank_clip_topk,
                    args.gate_logw_clip,
                )
                if args.prop_weight_mode in {"jvp_surrogate", "prop2"}:
                    (
                        global_jvp_used,
                        jvp_clip_mask,
                        jvp_clip_threshold,
                    ) = _rank_winsorize_rows(
                        global_jvp_raw,
                        args.jvp_logw_rank_clip_topk,
                        args.jvp_logw_clip,
                    )
                else:
                    global_jvp_used = global_jvp_raw
                    jvp_clip_mask = torch.zeros_like(
                        global_jvp_raw, dtype=torch.bool
                    )
                    jvp_clip_threshold = torch.tensor(
                        float("inf"), device=global_jvp_raw.device
                    )
                global_kernel_correction = (
                    global_reverse + global_forward_zero + global_jvp_used
                )
                (
                    global_kernel_correction_used,
                    kernel_clip_mask,
                    kernel_clip_threshold,
                ) = _rank_winsorize_rows(
                    global_kernel_correction,
                    args.kernel_logw_rank_clip_topk,
                    args.kernel_logw_clip,
                )
                global_logw = global_gate_used + global_kernel_correction_used
                local_gate_used = global_gate_used[
                    global_start:global_start + local_n
                ]
                local_gate_clip_delta = (
                    local_gate_used - exact["gate"].float()
                )
                local_jvp_used = global_jvp_used[
                    global_start:global_start + local_n
                ]
                local_kernel_correction = global_kernel_correction[
                    global_start:global_start + local_n
                ]
                local_kernel_correction_used = global_kernel_correction_used[
                    global_start:global_start + local_n
                ]
                local_logw_used = global_logw[
                    global_start:global_start + local_n
                ]
                global_carry = base.all_gather_cat(logw_total.float(), world_size)
                global_total = global_carry + global_logw
                inc_ess = base.global_ess_frac_from_logw(global_logw)
                full_cum_ess = base.global_ess_frac_from_logw(global_total)
                raw_inc_ess = base.global_ess_frac_from_logw(global_logw_raw)
                raw_step_on_used_carry_ess = base.global_ess_frac_from_logw(
                    global_carry + global_logw_raw
                )

                # A bounded resampling law may be used without discarding the
                # exact weight.  Ancestors are drawn using the winsorized log
                # weights, while log(W/P) (up to a common constant) is carried
                # by every selected child.  The default clip of zero recovers
                # ordinary exact-weight SMC.
                resample_clip = float(args.resample_logw_clip)
                if resample_clip > 0.0:
                    (
                        global_resample_logw,
                        global_resample_residual,
                        resample_clip_center,
                        resample_clip_mask,
                    ) = base._global_centered_clip(
                        global_total,
                        resample_clip,
                        center_mode=str(args.resample_logw_clip_center),
                    )
                else:
                    global_resample_logw = global_total
                    global_resample_residual = torch.zeros_like(global_total)
                    resample_clip_center = torch.tensor(
                        float("nan"), device=global_total.device
                    )
                    resample_clip_mask = torch.zeros_like(
                        global_total, dtype=torch.bool
                    )
                cum_ess = base.global_ess_frac_from_logw(global_resample_logw)
                resample_trigger_ess = (
                    full_cum_ess
                    if args.resample_trigger_mode == "full"
                    else cum_ess
                )
                do_resample = (
                    resample_trigger_ess < float(args.resample_ess)
                    and step < len(timesteps) - int(args.no_resample_last_steps)
                    and step - last_resample_step >= int(args.resample_min_gap)
                )
                global_weights = base.global_normalized_weights_from_logw(
                    global_resample_logw
                )
                ancestor_idx = None
                if do_resample:
                    if rank == 0:
                        ancestor_idx = base.systematic_resample(global_weights, resample_gen).long()
                    else:
                        ancestor_idx = torch.empty(
                            args.n_particles, device=device, dtype=torch.long
                        )
                    dist.broadcast(ancestor_idx, src=0)

                global_origin_before = base.all_gather_cat(origin_ids, world_size)
                global_origin_after = (
                    global_origin_before[ancestor_idx]
                    if do_resample else global_origin_before
                )
                roots = base.genealogy_stats(global_origin_after, args.n_particles)

                # Local expansion scalars retained for later checks.
                flat_d = d_y.float().flatten(1)
                flat_g = g_y.float().flatten(1)
                flat_u = exact["u_y"].float().flatten(1)
                flat_dx = exact["d_x"].float().flatten(1)
                flat_ux = exact["u_x"].float().flatten(1)
                nu_a = (latents.float() - mu_a.float()).flatten(1)
                hmu = (mean_fwd_y.float() - latents.float()).flatten(1)
                gate_nu_a = (latents.float() - gate_mu_a.float()).flatten(1)
                gate_nu_b = (latents.float() - gate_mu_b.float()).flatten(1)
                alpha_l = (
                    gate_nu_a.square().sum(dim=1)
                    - gate_nu_b.square().sum(dim=1)
                ) / (2.0 * variance)
                h_alpha_s = (
                    float(eta) * float(args.lhat_temp) * alpha_l
                    + math.log(max(c_new, 1e-30) / max(c, 1e-30))
                    + (float(eta_new) - float(eta)) * lhat.float()
                )
                q = flat_u - flat_g
                old_log1m = -torch.nn.functional.softplus(
                    math.log(max(float(c), 1e-30))
                    + float(eta) * lhat.float()
                )
                alpha_gate_time = (
                    -float(gate_power) * theta * h_alpha_s
                    + (float(gate_power_new) - float(gate_power)) * old_log1m
                )
                alpha_forward_linear = -(hmu * flat_u).sum(dim=1)
                alpha_forward_quadratic = -0.5 * forward_variance * flat_u.square().sum(dim=1)
                alpha_reverse_linear = (nu_a * q).sum(dim=1)
                alpha_reverse_quadratic = 0.5 * variance * q.square().sum(dim=1)
                alpha_no_hessian = (
                    alpha_gate_time + alpha_forward_linear
                    + alpha_forward_quadratic + alpha_reverse_linear
                    + alpha_reverse_quadratic
                )
                target_norm = (
                    0.5 * variance * zeta_old * (1.0 + zeta_old)
                    * flat_d.square().sum(dim=1)
                )
                c48_correction = (
                    0.5 * variance * float(args.c48_kappa)
                    * (float(args.c48_kappa) + 1.0)
                    * beta_l.float().flatten(1).square().sum(dim=1)
                )

                local_particle = {
                    "lhat_old": lhat.float(),
                    "lhat_new": lhat_new.float(),
                    "theta_old": theta.float(),
                    "theta_new": exact["theta_new"].float(),
                    "kernel_lr": exact["kernel_lr"].float(),
                    "zeta_old": zeta_old.float(),
                    "zeta_new": exact["zeta_new"].float(),
                    "rho1": torch.full_like(lhat, rho1_current),
                    "rho2": torch.full_like(lhat, rho2_current),
                    "effective_r_old": rho1_current + rho2_current * zeta_old,
                    "effective_r_new": rho1_current + rho2_current * exact["zeta_new"],
                    "d_y_norm2": flat_d.square().sum(dim=1),
                    "g_y_norm2": flat_g.square().sum(dim=1),
                    "u_y_norm2": flat_u.square().sum(dim=1),
                    "d_y_dot_g_y": (flat_d * flat_g).sum(dim=1),
                    "d_x_norm2": flat_dx.square().sum(dim=1),
                    "u_x_norm2": flat_ux.square().sum(dim=1),
                    "nu_a_dot_d": (nu_a * flat_d).sum(dim=1),
                    "forward_displacement_dot_d": (hmu * flat_d).sum(dim=1),
                    "alpha_l": alpha_l,
                    "h_alpha_s": h_alpha_s,
                    "alpha_gate_time": alpha_gate_time,
                    "alpha_forward_linear": alpha_forward_linear,
                    "alpha_forward_quadratic": alpha_forward_quadratic,
                    "alpha_reverse_linear": alpha_reverse_linear,
                    "alpha_reverse_quadratic": alpha_reverse_quadratic,
                    "alpha_no_hessian_total": alpha_no_hessian,
                    "alpha_target_norm_term": target_norm,
                    "c48_center_correction": c48_correction,
                    "prop2_gate_time": prop2_selected_components["gate_time"],
                    "prop2_target_curvature": prop2_selected_components[
                        "target_curvature"
                    ],
                    "prop2_forward_alpha": prop2_selected_components[
                        "forward_alpha"
                    ],
                    "prop2_reverse_alpha": prop2_selected_components[
                        "reverse_alpha"
                    ],
                    "prop2_proposal_curvature": prop2_selected_components[
                        "proposal_curvature"
                    ],
                    "prop2_jvp_raw": prop2_selected_components["jvp_raw"],
                    "prop2_total_raw": prop2_selected_raw,
                    "prop2_noise_total_raw": prop2_noise_selected_raw,
                    "prop2_jd_quadratic": prop2_selected_components[
                        "jd_quadratic"
                    ],
                    "prop2_d_projection2": prop2_selected_components[
                        "d_projection2"
                    ],
                    "prop2_curvature_coefficient": prop2_selected_components[
                        "curvature_coefficient"
                    ],
                    "diag_finite_exact_total": finite_diag["exact_total"],
                    "diag_finite_exact_gate": finite_diag["exact_gate"],
                    "diag_finite_exact_reverse": finite_diag["exact_reverse"],
                    "diag_finite_exact_forward": finite_diag["exact_forward"],
                    "diag_finite_no_jvp_total": finite_diag["no_jvp_total"],
                    "diag_finite_jvp_total": finite_diag["jvp_total"],
                    "diag_finite_forward_zero": finite_diag["forward_zero"],
                    "diag_finite_jvp_forward": finite_diag["jvp_forward"],
                    "diag_finite_true_jvp_correction": finite_diag[
                        "true_jvp_correction"
                    ],
                    "diag_finite_est_jvp_correction": finite_diag[
                        "est_jvp_correction"
                    ],
                    "logw_gate": exact["gate"].float(),
                    "logw_gate_used": local_gate_used.float(),
                    "logw_gate_clip_delta": local_gate_clip_delta.float(),
                    "logw_reverse_kernel": exact["reverse"].float(),
                    "logw_forward_kernel": exact["forward"].float(),
                    "logw_forward_kernel_zero": exact["forward_zero"].float(),
                    "logw_jvp_correction": exact["jvp_correction"].float(),
                    "logw_jvp_correction_used": local_jvp_used.float(),
                    "logw_kernel_correction": local_kernel_correction.float(),
                    "logw_kernel_correction_used": (
                        local_kernel_correction_used.float()
                    ),
                    "logw_kernel_correction_clip_delta": (
                        local_kernel_correction_used.float()
                        - local_kernel_correction.float()
                    ),
                    "logw_total": logw.float(),
                    "logw_total_used": local_logw_used.float(),
                    "logw_predicted_fixed_field": pred_selected.float(),
                    "logw_prediction_error": (
                        local_logw_used.float() - pred_selected.float()
                    ),
                    "logw_carry_before": logw_total.float(),
                    "logw_cumulative_before_resample": (
                        logw_total.float() + local_logw_used.float()
                    ),
                    "logw_resample_selection": global_resample_logw[
                        global_start:global_start + local_n
                    ].float(),
                    "logw_resample_clip_residual": global_resample_residual[
                        global_start:global_start + local_n
                    ].float(),
                    "origin_id_before": origin_ids.float(),
                    "origin_id_after": global_origin_after[
                        global_start:global_start + local_n
                    ].float(),
                }
                global_particle = {
                    key: base.all_gather_cat(value.reshape(-1).contiguous(), world_size)
                    for key, value in local_particle.items()
                }
                if rank == 0:
                    for key in PARTICLE_FIELDS:
                        particle_history[key].append(
                            global_particle[key].detach().cpu().numpy()
                        )
                    row: dict[str, Any] = {
                        "step": step,
                        "timestep": t_int,
                        "rho1": rho1_current,
                        "rho2": rho2_current,
                        "rho1_initial": rho1_initial,
                        "rho2_initial": rho2_initial,
                        "rho1_over_rho2": (
                            rho1_current / rho2_current
                            if abs(rho2_current) > 1e-12 else float("nan")
                        ),
                        "inc_ess_frac": inc_ess,
                        "raw_inc_ess_frac": raw_inc_ess,
                        "raw_step_on_used_carry_ess_frac": raw_step_on_used_carry_ess,
                        "cum_ess_frac": cum_ess,
                        "full_cum_ess_frac": full_cum_ess,
                        "gate_logw_rank_clip_topk": int(
                            args.gate_logw_rank_clip_topk
                        ),
                        "gate_logw_clip": float(args.gate_logw_clip),
                        "gate_logw_clip_threshold": float(
                            gate_clip_threshold.item()
                        ),
                        "gate_logw_clip_frac": float(
                            gate_clip_mask.float().mean().item()
                        ),
                        "optimizer_field_mode": str(args.optimizer_field_mode),
                        "optimizer_weight_mode": str(args.optimizer_weight_mode),
                        "rho2_grid_objective": str(
                            args.rho2_grid_objective
                        ),
                        "prop_weight_mode": str(args.prop_weight_mode),
                        "jvp_used": str(jvp_used),
                        "jvp_shrink_alpha": float(args.jvp_shrink_alpha),
                        "jvp_logw_rank_clip_topk": int(
                            args.jvp_logw_rank_clip_topk
                        ),
                        "jvp_logw_clip": float(args.jvp_logw_clip),
                        "jvp_logw_clip_threshold": float(
                            jvp_clip_threshold.item()
                        ),
                        "jvp_logw_clip_frac": float(
                            jvp_clip_mask.float().mean().item()
                        ),
                        "kernel_logw_rank_clip_topk": int(
                            args.kernel_logw_rank_clip_topk
                        ),
                        "kernel_logw_clip": float(args.kernel_logw_clip),
                        "kernel_logw_clip_threshold": float(
                            kernel_clip_threshold.item()
                        ),
                        "kernel_logw_clip_frac": float(
                            kernel_clip_mask.float().mean().item()
                        ),
                        "resample_logw_clip": resample_clip,
                        "resample_logw_clip_center": (
                            float(resample_clip_center.item())
                            if torch.isfinite(resample_clip_center)
                            else float("nan")
                        ),
                        "resample_logw_clip_frac": float(
                            resample_clip_mask.float().mean().item()
                        ),
                        "resample_logw_clip_residual_std": float(
                            global_resample_residual.std(unbiased=False).item()
                        ),
                        "resample_trigger_mode": str(args.resample_trigger_mode),
                        "resample_trigger_ess_frac": float(resample_trigger_ess),
                        "resampled": int(do_resample),
                        "unique_roots": int(roots["unique_initial_ancestors"]),
                        "effective_roots": float(roots["initial_ancestor_effective_roots"]),
                        "max_root_copies": int(roots["max_initial_ancestor_copies"]),
                        "elapsed_step_sec": time.time() - t0,
                    }
                    for field in (
                        "zeta_old", "effective_r_old", "logw_gate",
                        "logw_gate_used", "logw_gate_clip_delta",
                        "logw_reverse_kernel", "logw_forward_kernel",
                        "logw_forward_kernel_zero",
                        "logw_jvp_correction",
                        "logw_jvp_correction_used",
                        "logw_kernel_correction",
                        "logw_kernel_correction_used",
                        "logw_kernel_correction_clip_delta",
                        "logw_total", "logw_total_used",
                        "logw_prediction_error",
                        "alpha_gate_time", "alpha_forward_linear",
                        "alpha_forward_quadratic", "alpha_reverse_linear",
                        "alpha_reverse_quadratic", "alpha_no_hessian_total",
                        "alpha_target_norm_term", "c48_center_correction",
                        "prop2_gate_time", "prop2_target_curvature",
                        "prop2_forward_alpha", "prop2_reverse_alpha",
                        "prop2_proposal_curvature", "prop2_jvp_raw",
                        "prop2_total_raw", "prop2_jd_quadratic",
                        "prop2_d_projection2",
                        "prop2_curvature_coefficient",
                        "diag_finite_exact_total",
                        "diag_finite_exact_gate",
                        "diag_finite_exact_reverse",
                        "diag_finite_exact_forward",
                        "diag_finite_no_jvp_total",
                        "diag_finite_jvp_total",
                        "diag_finite_forward_zero",
                        "diag_finite_jvp_forward",
                        "diag_finite_true_jvp_correction",
                        "diag_finite_est_jvp_correction",
                    ):
                        row.update(_prefixed_stats(field, global_particle[field]))
                    logw_term_stds = {
                        "gate": row["logw_gate_std"],
                        "reverse_kernel": row["logw_reverse_kernel_std"],
                        "forward_kernel": row["logw_forward_kernel_std"],
                    }
                    alpha_term_stds = {
                        "gate_time": row["alpha_gate_time_std"],
                        "forward_linear": row["alpha_forward_linear_std"],
                        "forward_quadratic": row["alpha_forward_quadratic_std"],
                        "reverse_linear": row["alpha_reverse_linear_std"],
                        "reverse_quadratic": row["alpha_reverse_quadratic_std"],
                    }
                    row["largest_logw_term_by_std"] = max(
                        logw_term_stds, key=logw_term_stds.get
                    )
                    row["largest_logw_term_std"] = max(logw_term_stds.values())
                    row["largest_alpha_term_by_std"] = max(
                        alpha_term_stds, key=alpha_term_stds.get
                    )
                    row["largest_alpha_term_std"] = max(alpha_term_stds.values())
                    if continuous_diag:
                        for key, value in continuous_diag.items():
                            if isinstance(value, (bool, int, float, str)):
                                row[f"optimizer_{key}"] = value
                    rows.append(row)
                    if step % args.log_every == 0 or step == len(timesteps) - 1:
                        print(
                            f"[{args.mode} {step:03d}/{len(timesteps)} t={t_int:3d}] "
                            f"rho1={rho1_current:+.3f} rho2={rho2_current:+.3f} "
                            f"incESS={inc_ess:.3f} selectESS={cum_ess:.3f} "
                            f"fullESS={full_cum_ess:.3f} "
                            f"std(logw)={row['logw_total_std']:.3g} "
                            f"roots={row['unique_roots']} resample={int(do_resample)}",
                            flush=True,
                        )

                if do_resample:
                    last_resample_step = step
                    all_x = base.all_gather_cat(x_new.contiguous(), world_size)
                    all_lhat = base.all_gather_cat(lhat_new.contiguous(), world_size)
                    local_idx = ancestor_idx[global_start:global_start + local_n]
                    latents = all_x[local_idx].to(dtype=latents.dtype)
                    lhat = all_lhat[local_idx].float()
                    origin_ids = global_origin_before[local_idx].long()
                    selected_residual = global_resample_residual[
                        ancestor_idx
                    ].float()
                    # Remove one global common constant without changing
                    # weights; rank-local centering would be incorrect.
                    selected_residual = (
                        selected_residual - selected_residual.max()
                    )
                    logw_total = selected_residual[
                        global_start:global_start + local_n
                    ]
                else:
                    latents = x_new.to(dtype=latents.dtype)
                    lhat = lhat_new.float()
                    # Remove one global additive constant for numerical stability.
                    stabilized = global_total - global_total.max()
                    logw_total = stabilized[global_start:global_start + local_n]

                if (
                    int(args.early_stop_roots) > 0
                    and int(roots["unique_initial_ancestors"])
                    <= int(args.early_stop_roots)
                ):
                    early_stopped = True
                    if rank == 0:
                        print(
                            "[early-stop] genealogy collapsed to "
                            f"{int(roots['unique_initial_ancestors'])} root(s) "
                            f"at step {step}",
                            flush=True,
                        )
                    break

        if args.save_images and not early_stopped:
            print(f"[decode rank {rank}] {local_n} images", flush=True)
            base.decode_and_save_images(
                pipe, latents, args, start_index=global_start,
                make_grid_after=False,
            )
            base.write_decode_done_marker(args.out_dir, rank)
        dist.barrier()
        if rank == 0:
            base.write_rows(args.out_dir / "exact_two_rho_monitor.csv", rows)
            np.savez_compressed(
                args.out_dir / "exact_two_rho_particle_terms.npz",
                **{key: np.stack(values, axis=0) for key, values in particle_history.items()},
            )
            np.savez_compressed(
                args.out_dir / "exact_two_rho_optimizer_surface.npz",
                candidate_rho1=cand_rho1.detach().cpu().numpy(),
                candidate_rho2=cand_rho2.detach().cpu().numpy(),
                predicted_inc_ess_initial=np.stack(
                    optimizer_initial_inc_history, axis=0
                ),
                predicted_total_ess_initial=np.stack(
                    optimizer_initial_total_history, axis=0
                ),
                predicted_total_logw_variance_initial=np.stack(
                    optimizer_initial_variance_history, axis=0
                ),
                predicted_inc_ess=np.stack(optimizer_inc_history, axis=0),
                predicted_total_ess=np.stack(optimizer_total_history, axis=0),
                predicted_total_logw_variance=np.stack(
                    optimizer_variance_history, axis=0
                ),
            )
            with (args.out_dir / "continuous_optimizer_diagnostics.json").open(
                "w"
            ) as handle:
                json.dump(continuous_optimizer_history, handle, indent=2)
            summary = {
                "status": "early_stopped_genealogy" if early_stopped else "complete",
                "mode": args.mode,
                "target": "pA_times_powered_soft_gate",
                "weight": (
                    "prop2_local_expansion_with_gate_and_jvp_winsorization"
                    if args.prop_weight_mode == "prop2"
                    else (
                        "finite_endpoint_field_jvp_weight_with_gate_and_jvp_winsorization"
                        if args.prop_weight_mode == "jvp_surrogate"
                        else "finite_Gaussian_weight_with_gate_winsorization"
                    )
                ),
                "optimizer_surrogate": (
                    "prop2_two_field_local_expansion_with_gate_and_jvp_winsorization"
                    if args.optimizer_weight_mode == "prop2"
                    else (
                        "finite_Gaussian_ratio_with_fixed_displacement_JVP_endpoint_field"
                        if args.optimizer_field_mode == "jvp"
                        else "finite_Gaussian_ratio_with_frozen_endpoint_field"
                    )
                ),
                "optimizer_weight_mode": args.optimizer_weight_mode,
                "optimizer_field_mode": args.optimizer_field_mode,
                "rho2_grid_objective": args.rho2_grid_objective,
                "prop_weight_mode": args.prop_weight_mode,
                "jvp_shrink_alpha": float(args.jvp_shrink_alpha),
                "diagnose_finite_exact_weight": bool(
                    args.diagnose_finite_exact_weight
                ),
                "jvp_logw_rank_clip_topk": int(
                    args.jvp_logw_rank_clip_topk
                ),
                "jvp_logw_clip": float(args.jvp_logw_clip),
                "steps": args.steps,
                "completed_steps": len(rows),
                "particles": args.n_particles,
                "seed": args.seed,
                "resamples": int(sum(row["resampled"] for row in rows)),
                "mean_inc_ess": float(np.mean([row["inc_ess_frac"] for row in rows])),
                "min_inc_ess": float(np.min([row["inc_ess_frac"] for row in rows])),
                "mean_raw_inc_ess": float(np.mean([row["raw_inc_ess_frac"] for row in rows])),
                "min_raw_inc_ess": float(np.min([row["raw_inc_ess_frac"] for row in rows])),
                "gate_logw_rank_clip_topk": int(args.gate_logw_rank_clip_topk),
                "gate_logw_clip": float(args.gate_logw_clip),
                "mean_gate_logw_clip_frac": float(np.mean([row["gate_logw_clip_frac"] for row in rows])),
                "max_gate_logw_clip_frac": float(np.max([row["gate_logw_clip_frac"] for row in rows])),
                "mean_cum_ess": float(np.mean([row["cum_ess_frac"] for row in rows])),
                "min_cum_ess": float(np.min([row["cum_ess_frac"] for row in rows])),
                "final_cum_ess": float(rows[-1]["cum_ess_frac"]),
                "mean_full_cum_ess": float(np.mean([row["full_cum_ess_frac"] for row in rows])),
                "min_full_cum_ess": float(np.min([row["full_cum_ess_frac"] for row in rows])),
                "final_full_cum_ess": float(rows[-1]["full_cum_ess_frac"]),
                "final_unique_roots": int(rows[-1]["unique_roots"]),
                "final_effective_roots": float(rows[-1]["effective_roots"]),
                "mean_rho1": float(np.mean([row["rho1"] for row in rows])),
                "mean_rho2": float(np.mean([row["rho2"] for row in rows])),
                "elapsed_sec": time.time() - start_time,
                "particle_terms_npz": str(args.out_dir / "exact_two_rho_particle_terms.npz"),
                "optimizer_surface_npz": str(args.out_dir / "exact_two_rho_optimizer_surface.npz"),
                "monitor_csv": str(args.out_dir / "exact_two_rho_monitor.csv"),
            }
            with (args.out_dir / "summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2)
            if early_stopped:
                with (args.out_dir / "early_stop.json").open("w") as handle:
                    json.dump(summary, handle, indent=2)
            if args.save_images and not early_stopped:
                base.wait_for_decode_markers(args.out_dir, world_size, args.image_marker_timeout)
                base.assemble_grid_from_saved_images(args, args.n_particles)
            print(f"[saved] {args.out_dir}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "c48_ratio", "free", "fixed_rho1",
            "c48_ratio_continuous", "free_continuous",
            "fixed_rho1_continuous",
        ],
        required=True,
    )
    parser.add_argument("--devices", default="cuda:4,cuda:5,cuda:6,cuda:7")
    parser.add_argument("--dist-port", type=int, default=62401)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompts-file", type=Path, default=SCRIPT_DIR / "dng_prompts.json")
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--negative-kind", choices=["related", "unrelated"], default="related")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--unet-chunk-size", type=int, default=8)
    parser.add_argument("--attention-slicing", default="auto")
    parser.add_argument("--c0", type=float, default=48.0)
    parser.add_argument("--c1", type=float, default=48.0)
    parser.add_argument("--c-schedule", default="constant")
    parser.add_argument("--bs0", type=float, default=0.005)
    parser.add_argument("--bs1", type=float, default=0.005)
    parser.add_argument("--bs-schedule", default="constant")
    parser.add_argument("--gamma-c", type=float, default=1.0)
    parser.add_argument("--gamma-bs", type=float, default=1.0)
    parser.add_argument("--theta-mid-logit-shift", type=float, default=-0.05)
    parser.add_argument(
        "--c-path-shape", choices=["base", "plateau"], default="base"
    )
    parser.add_argument("--c-path-logit-shift", type=float, default=-8.0)
    parser.add_argument("--c-path-down-end-frac", type=float, default=0.25)
    parser.add_argument("--c-path-return-start-frac", type=float, default=0.75)
    parser.add_argument("--gate-power", type=float, default=1769.44444444444)
    parser.add_argument("--gate-power-schedule", default="constant")
    parser.add_argument("--gate-power-start-frac", type=float, default=0.0)
    parser.add_argument("--gate-power-end-frac", type=float, default=1.0)
    parser.add_argument("--gate-power-schedule-gamma", type=float, default=1.0)
    parser.add_argument("--lhat-temp", type=float, default=0.75)
    parser.add_argument(
        "--reward-tanh-scale",
        type=float,
        default=0.0,
        help=(
            "Positive C replaces the centered gate log-potential by "
            "C*tanh(R_centered/C); zero keeps the original target."
        ),
    )
    parser.add_argument("--min-weight-variance", type=float, default=1e-12)
    parser.add_argument(
        "--gate-logw-rank-clip-topk",
        type=int,
        default=0,
        help=(
            "Winsorize at most this many extreme centered gate increments "
            "before rho optimization and in the selected finite weight. "
            "Kernel ratios remain exact; zero disables the approximation."
        ),
    )
    parser.add_argument(
        "--gate-logw-clip",
        type=float,
        default=0.0,
        help=(
            "Optional median-centered absolute cap applied only to the gate "
            "log-weight increment, before rho selection and in the realized "
            "finite weight. Gaussian kernel ratios remain exact; zero "
            "disables the cap."
        ),
    )
    parser.add_argument("--resample-ess", type=float, default=0.825)
    parser.add_argument(
        "--resample-logw-clip",
        type=float,
        default=0.0,
        help=(
            "Winsorize cumulative log weights only for ancestor selection; "
            "the removed residual is carried after resampling. Zero disables."
        ),
    )
    parser.add_argument(
        "--resample-logw-clip-center",
        choices=["median", "mean", "zero"],
        default="median",
    )
    parser.add_argument(
        "--resample-trigger-mode",
        choices=["selection", "full"],
        default="full",
        help=(
            "Trigger resampling from the full cumulative-weight ESS by "
            "default. The selection option is retained only for reproducing "
            "earlier bounded-ancestor ablations."
        ),
    )
    parser.add_argument("--resample-min-gap", type=int, default=1)
    parser.add_argument("--no-resample-last-steps", type=int, default=10)
    parser.add_argument("--no-rho-last-steps", type=int, default=2)
    parser.add_argument("--early-stop-roots", type=int, default=1)
    parser.add_argument("--optimizer-refine-passes", type=int, default=0)
    parser.add_argument(
        "--optimizer-field-mode",
        choices=["frozen", "jvp"],
        default="frozen",
        help=(
            "Endpoint score used during rho search: current-state frozen "
            "field or one fixed-displacement first-order JVP."
        ),
    )
    parser.add_argument(
        "--optimizer-weight-mode",
        choices=["finite_gaussian", "prop2"],
        default="finite_gaussian",
        help=(
            "Weight surrogate used to select rho: the finite Gaussian ratio "
            "or the retained h*alpha+Delta'*Gamma*Delta expansion from "
            "Proposition 2, specialized to the two-field proposal."
        ),
    )
    parser.add_argument(
        "--rho2-grid-objective",
        choices=["full_ess", "log_weight_variance"],
        default="full_ess",
        help=(
            "For grid rho modes, select the source-wise-clipped candidate "
            "by maximum full cumulative ESS or minimum variance of the "
            "full cumulative log weights."
        ),
    )
    parser.add_argument(
        "--prop-weight-mode",
        choices=["exact", "jvp_surrogate", "prop2"],
        default="exact",
        help=(
            "Use a direct selected-endpoint score evaluation or propagate "
            "an endpoint-field JVP surrogate or the retained Proposition-2 "
            "local expansion."
        ),
    )
    parser.add_argument(
        "--jvp-mode",
        choices=["forward-ad", "finite-diff", "auto"],
        default="forward-ad",
    )
    parser.add_argument("--jvp-eps", type=float, default=0.01)
    parser.add_argument("--jvp-shrink-alpha", type=float, default=0.5)
    parser.add_argument("--jvp-logw-rank-clip-topk", type=int, default=4)
    parser.add_argument("--jvp-logw-clip", type=float, default=0.0)
    parser.add_argument(
        "--kernel-logw-rank-clip-topk",
        type=int,
        default=0,
        help=(
            "Winsorize this many extreme centered values of the combined "
            "reverse, forward, and stabilized-JVP local correction."
        ),
    )
    parser.add_argument(
        "--kernel-logw-clip",
        type=float,
        default=0.0,
        help=(
            "Optional median-centered absolute cap on the combined local "
            "kernel correction; zero disables the value cap."
        ),
    )
    parser.add_argument(
        "--diagnose-noise-only-jvp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record a matched noise-only Proposition-2 surrogate using one "
            "additional JVP; this diagnostic does not alter proposal selection."
        ),
    )
    parser.add_argument(
        "--diagnose-finite-exact-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate the selected endpoint score for a diagnostic finite "
            "Gaussian reference. The extra evaluation is recorded but does "
            "not alter the propagated state or weight."
        ),
    )
    parser.add_argument("--rho-ridge", type=float, default=1e-8)
    parser.add_argument("--c48-kappa", type=float, default=6.5)
    parser.add_argument("--fixed-rho1", type=float, default=6.5)
    parser.add_argument("--fixed-rho1-final", type=float, default=0.0)
    parser.add_argument(
        "--fixed-rho1-schedule",
        choices=["constant", "linear", "cosine"],
        default="constant",
    )
    parser.add_argument("--fixed-rho1-start-frac", type=float, default=0.0)
    parser.add_argument("--fixed-rho1-end-frac", type=float, default=0.2)
    parser.add_argument("--rho1-min", type=float, default=-8.0)
    parser.add_argument("--rho1-max", type=float, default=8.0)
    parser.add_argument("--rho1-points", type=int, default=65)
    parser.add_argument("--rho2-min", type=float, default=-3.0)
    parser.add_argument("--rho2-max", type=float, default=3.0)
    parser.add_argument("--rho2-points", type=int, default=121)
    parser.add_argument(
        "--rho-continuous-bounded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-decode-batch-size", type=int, default=1)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--image-marker-timeout", type=float, default=1800.0)
    parser.add_argument("--clip-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-score-device", default="cuda:4")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-text-batch-size", type=int, default=16)
    parser.add_argument("--clip-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-existing-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one CUDA device is required")
    if args.n_particles <= 0 or args.n_particles % len(devices) != 0:
        raise ValueError(
            "--n-particles must be positive and divisible by the number of devices"
        )
    if args.gate_logw_rank_clip_topk < 0:
        raise ValueError("--gate-logw-rank-clip-topk must be nonnegative")
    if args.gate_logw_clip < 0.0:
        raise ValueError("--gate-logw-clip must be nonnegative")
    if args.jvp_logw_rank_clip_topk < 0:
        raise ValueError("--jvp-logw-rank-clip-topk must be nonnegative")
    if args.jvp_logw_clip < 0.0:
        raise ValueError("--jvp-logw-clip must be nonnegative")
    if args.kernel_logw_rank_clip_topk < 0:
        raise ValueError("--kernel-logw-rank-clip-topk must be nonnegative")
    if args.kernel_logw_clip < 0.0:
        raise ValueError("--kernel-logw-clip must be nonnegative")
    if not 0.0 <= float(args.jvp_shrink_alpha) <= 1.0:
        raise ValueError("--jvp-shrink-alpha must lie in [0,1]")
    if (
        args.optimizer_field_mode == "jvp"
        or args.optimizer_weight_mode == "prop2"
    ) and args.mode.endswith("_continuous"):
        raise ValueError("The JVP comparison currently requires a grid rho mode")
    if (
        args.optimizer_field_mode == "jvp"
        or args.optimizer_weight_mode == "prop2"
    ) and args.optimizer_refine_passes != 0:
        raise ValueError(
            "JVP/Proposition-2 selection is a single fixed-displacement pass; "
            "set --optimizer-refine-passes=0"
        )
    if args.prop_weight_mode == "jvp_surrogate" and args.optimizer_field_mode != "jvp":
        raise ValueError("JVP propagated weights require --optimizer-field-mode=jvp")
    if (
        args.optimizer_weight_mode == "prop2"
        or args.prop_weight_mode == "prop2"
    ) and args.reward_tanh_scale != 0.0:
        raise ValueError(
            "The Proposition-2 mode currently implements the unsquashed "
            "powered sigmoid target; set --reward-tanh-scale=0."
        )
    if (
        (
            args.gate_logw_rank_clip_topk > 0
            or args.gate_logw_clip > 0.0
            or args.kernel_logw_rank_clip_topk > 0
            or args.kernel_logw_clip > 0.0
        )
        and args.mode.endswith("_continuous")
    ):
        raise ValueError(
            "Gate rank clipping is non-smooth; use a grid mode so rho is "
            "optimized after clipping."
        )
    # The shared CLIP helper accepts both the multi-device sampler option and
    # the legacy singular fallback used by older runners.
    args.device = devices[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt, negative, _ = base.resolve_prompt(args)
    if args.score_existing_only:
        base.score_final_images_with_clip(args, prompt, negative)
        return
    stale_marker = args.out_dir / "early_stop.json"
    if stale_marker.exists():
        stale_marker.unlink()
    config = {**vars(args), "out_dir": str(args.out_dir), "prompts_file": str(args.prompts_file), "prompt": prompt, "negative_prompt_used": negative}
    with (args.out_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, default=str)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.dist_port)
    mp.spawn(worker, args=(args, devices), nprocs=len(devices), join=True)
    if args.clip_score and not (args.out_dir / "early_stop.json").exists():
        base.score_final_images_with_clip(args, prompt, negative)


if __name__ == "__main__":
    main()
