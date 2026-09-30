# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

import json
from math import sqrt
from math import exp
from scipy.stats import norm
import math
import os
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module
from typing import Any, Dict, Optional, List

from tqdm import tqdm
import boltzgen.model.layers.initialize as init
from boltzgen.data import const
from boltzgen.model.layers.miniformer import MiniformerModule
from boltzgen.model.layers.pairformer import PairformerModule
from boltzgen.model.loss.diffusion import (
    compute_bond_loss,
    smooth_lddt_loss,
    weighted_rigid_align,
    weighted_rigid_centering,
)
from boltzgen.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    CoordinateConditioning,
    FourierEmbedding,
    SingleConditioning,
)
from boltzgen.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltzgen.model.modules.utils import (
    LinearNoBias,
    center,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from scipy.stats import beta


def optionally_tqdm(iterable, use_tqdm=True, **kwargs):
    return tqdm(iterable, **kwargs) if use_tqdm else iterable


def _env_flag(name: str, default: str = "0") -> bool:
    """Read a boolean environment variable."""
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_float(name: str, default: float) -> float:
    """Read a floating-point environment variable."""
    value = os.environ.get(name)
    return default if value is None else float(value)


def _trace_array(value: Any) -> np.ndarray:
    """Convert trace values to NumPy, promoting unsupported BF16 tensors."""
    if torch.is_tensor(value):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def _write_trace_npz(path: Path, **values: Any) -> None:
    """Atomically write one compressed trajectory archive."""
    arrays = {
        key: _trace_array(value)
        for key, value in values.items()
        if value is not None
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_trace_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write human-readable trajectory metadata."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _token_mask_to_atom_mask(
    feats: Dict[str, torch.Tensor],
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Expand a [batch, token] mask to atom slots."""
    return (
        torch.bmm(
            feats["atom_to_token"].float(),
            token_mask.float().unsqueeze(-1),
        )
        .squeeze(-1)
        .bool()
    )


def _binder_atom_mask(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return all atom slots belonging to the de-novo designed chain.

    ``chain_design_mask`` is deliberately used instead of molecule type.
    Protein binders are not NONPOLYMER ligands, and ``design_mask`` alone can
    omit covalently attached designed components.
    """
    if "chain_design_mask" not in feats:
        raise KeyError(
            "Binder negative guidance requires feats['chain_design_mask']. "
            "Use the FromYaml BoltzGen design data module."
        )
    return feats["atom_pad_mask"].bool() & _token_mask_to_atom_mask(
        feats,
        feats["chain_design_mask"].bool(),
    )


def _hla_platform_atom_mask(
    feats: Dict[str, torch.Tensor],
    max_residues: int = 180,
) -> torch.Tensor:
    """Select resolved C-alpha atoms from the longest fixed protein chain.

    For a pMHC target this is the MHC-I heavy chain.  The binder is excluded
    through ``chain_design_mask``; beta-2 microglobulin and the peptide are
    shorter and therefore cannot be selected.  Only the first 180 heavy-chain
    tokens are used so the fit is dominated by the peptide-binding platform.
    """
    token_pad = feats["token_pad_mask"].bool()
    token_protein = torch.eq(
        feats["mol_type"].long(),
        const.chain_type_ids["PROTEIN"],
    )
    token_fixed_protein = (
        token_pad
        & token_protein
        & (~feats["chain_design_mask"].bool())
    )
    selected_tokens = torch.zeros_like(token_fixed_protein)

    for batch_idx in range(token_fixed_protein.shape[0]):
        candidate = token_fixed_protein[batch_idx]
        asym_ids = torch.unique(feats["asym_id"][batch_idx, candidate])
        if asym_ids.numel() == 0:
            raise ValueError(
                "No fixed protein chain is available for pMHC alignment."
            )
        counts = torch.stack(
            [
                (
                    candidate
                    & torch.eq(feats["asym_id"][batch_idx], asym_id)
                ).sum()
                for asym_id in asym_ids
            ]
        )
        heavy_asym_id = asym_ids[torch.argmax(counts)]
        heavy_tokens = torch.nonzero(
            candidate
            & torch.eq(feats["asym_id"][batch_idx], heavy_asym_id),
            as_tuple=False,
        ).flatten()
        selected_tokens[
            batch_idx,
            heavy_tokens[:max_residues],
        ] = True

    atom_mask = _token_mask_to_atom_mask(feats, selected_tokens)
    atom_mask &= feats["atom_pad_mask"].bool()
    atom_mask &= feats["atom_resolved_mask"].bool()
    if "fake_atom_mask" in feats:
        atom_mask &= ~feats["fake_atom_mask"].bool()
    atom_name_codes = feats["ref_atom_name_chars"].int().argmax(dim=-1)
    ca_name_codes = torch.tensor(
        [ord("C") - 32, ord("A") - 32, 0, 0],
        device=atom_name_codes.device,
        dtype=atom_name_codes.dtype,
    )
    atom_mask &= torch.eq(atom_name_codes, ca_name_codes).all(dim=-1)
    return atom_mask


def _row_kabsch(
    mobile: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ``mobile @ rotation + translation`` onto ``target``.

    Coordinates use row-vector convention.  The returned RMSD is measured
    after fitting.  Vectors must be transformed by rotation only.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[-1] != 3:
        raise ValueError(
            f"Kabsch inputs must both be [n, 3], got {mobile.shape} and "
            f"{target.shape}."
        )
    if mobile.shape[0] < 3:
        raise ValueError(
            f"At least three MHC atoms are required for alignment, got "
            f"{mobile.shape[0]}."
        )

    work_dtype = torch.float32
    mobile_f = mobile.to(work_dtype)
    target_f = target.to(work_dtype)
    mobile_centroid = mobile_f.mean(dim=0)
    target_centroid = target_f.mean(dim=0)
    mobile_centered = mobile_f - mobile_centroid
    target_centered = target_f - target_centroid
    covariance = mobile_centered.mT @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    if torch.det(u @ vh) < 0:
        u = u.clone()
        u[:, -1] *= -1
    rotation = u @ vh
    translation = target_centroid - mobile_centroid @ rotation
    fitted = mobile_f @ rotation + translation
    rmsd = torch.sqrt(torch.mean(torch.sum((fitted - target_f) ** 2, dim=-1)))
    return (
        rotation.to(mobile.dtype),
        translation.to(mobile.dtype),
        rmsd.to(mobile.dtype),
    )


def _rotation_angle_degrees(rotation: torch.Tensor) -> float:
    """Return the principal angle of a proper 3D rotation matrix."""
    rotation = rotation.float()
    cosine = torch.clamp(
        (torch.trace(rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    skew = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sine = torch.linalg.vector_norm(skew) / 2.0
    # atan2 is numerically stable near identity, unlike acos(trace), whose
    # derivative diverges as the angle approaches zero.
    angle = torch.atan2(sine, cosine)
    return float(torch.rad2deg(angle).item())


def _semantic_atom_pairs(
    feats: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair wanted/unwanted atom slots by token index and atom name.

    BoltzGen packs valid atoms.  A residue substitution with a different atom
    count therefore shifts every later atom slot, even when token/entity order
    is identical.  Slotwise pairing is invalid for such mutations.
    """
    atom_pad = feats["atom_pad_mask"].detach().cpu().bool()
    atom_to_token = (
        feats["atom_to_token"].detach().cpu().to(torch.int8).argmax(dim=-1)
    )
    atom_name = (
        feats["ref_atom_name_chars"]
        .detach()
        .cpu()
        .to(torch.int8)
        .argmax(dim=-1)
    )
    lookups: list[dict[tuple[int, ...], int]] = []
    for batch_idx in range(2):
        lookup: dict[tuple[int, ...], int] = {}
        for atom_idx in torch.nonzero(
            atom_pad[batch_idx],
            as_tuple=False,
        ).flatten().tolist():
            key = (
                int(atom_to_token[batch_idx, atom_idx]),
                *(
                    int(value)
                    for value in atom_name[batch_idx, atom_idx].tolist()
                ),
            )
            if key in lookup:
                raise ValueError(
                    "Duplicate semantic atom key in condition "
                    f"{batch_idx}: token/name={key}."
                )
            lookup[key] = atom_idx
        lookups.append(lookup)

    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for key, positive_idx in lookups[0].items():
        negative_idx = lookups[1].get(key)
        if negative_idx is not None:
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)
    if not positive_indices:
        raise ValueError(
            "Wanted and unwanted conditions have no semantically matching "
            "atoms."
        )
    device = feats["atom_pad_mask"].device
    return (
        torch.tensor(positive_indices, device=device, dtype=torch.long),
        torch.tensor(negative_indices, device=device, dtype=torch.long),
    )


def _guidance_scale_at_progress(
    base_scale: float,
    progress: float,
    schedule: str,
    start: float,
    end: float,
) -> float:
    """Return a constant or smoothly ramped contrastive scale."""
    if schedule == "constant":
        return base_scale
    if schedule not in {"linear", "cosine"}:
        raise ValueError(
            "BOLTZGEN_NEG_GUIDANCE_SCHEDULE must be constant, linear, or "
            f"cosine, got {schedule!r}."
        )
    if not 0 <= start < end <= 1:
        raise ValueError(
            "Negative-guidance ramp must satisfy 0 <= start < end <= 1."
        )
    fraction = min(max((progress - start) / (end - start), 0.0), 1.0)
    if schedule == "cosine":
        fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
    return base_scale * fraction


def _compose_binder_denoised(
    positive_denoised: torch.Tensor,
    negative_denoised: torch.Tensor,
    positive_noisy: torch.Tensor,
    negative_noisy: torch.Tensor,
    t_hat: float,
    positive_binder_indices: torch.Tensor,
    negative_binder_indices: torch.Tensor,
    negative_to_positive_rotation: torch.Tensor,
    guidance_scale: float,
    max_delta_ratio: float,
) -> tuple[torch.Tensor, float, float, float]:
    """Contrast frame-consistent EDM velocities on binder atoms only.

    The unwanted displacement vector is rotated into the wanted HLA frame.
    Translation is deliberately absent: applying the same rigid transform to
    both a noisy origin and its denoised endpoint cancels translation.  The
    returned context is copied exactly from the wanted/positive condition.
    """
    if positive_denoised.ndim != 2 or positive_denoised.shape[-1] != 3:
        raise ValueError("Denoised coordinate tensors must have shape [m, 3].")
    if positive_denoised.shape != positive_noisy.shape:
        raise ValueError("Wanted denoised/noisy coordinate tensors must match.")
    if negative_denoised.shape != negative_noisy.shape:
        raise ValueError(
            "Unwanted denoised/noisy coordinate tensors must match."
        )
    if not math.isfinite(float(t_hat)) or float(t_hat) <= 0:
        raise ValueError(f"t_hat must be finite and positive, got {t_hat}.")
    if positive_binder_indices.numel() == 0:
        raise ValueError("The designed protein binder atom index is empty.")
    if positive_binder_indices.shape != negative_binder_indices.shape:
        raise ValueError("Wanted/unwanted binder atom pairing is incomplete.")

    positive_velocity = (
        positive_noisy[positive_binder_indices]
        - positive_denoised[positive_binder_indices]
    ) / float(t_hat)
    negative_velocity_aligned = (
        (
            negative_noisy[negative_binder_indices]
            - negative_denoised[negative_binder_indices]
        )
        / float(t_hat)
    ) @ negative_to_positive_rotation
    velocity_delta = positive_velocity - negative_velocity_aligned
    # x0_guided = x_noisy - t * (v_A + lambda * (v_A - R(v_B))).
    endpoint_delta = -float(t_hat) * velocity_delta
    positive_binder = positive_denoised[positive_binder_indices]
    positive_motion = (
        positive_binder - positive_noisy[positive_binder_indices]
    )
    positive_rms = float(
        torch.sqrt(torch.mean(torch.square(positive_motion.float()))).item()
    )
    delta_rms = float(
        torch.sqrt(torch.mean(torch.square(endpoint_delta.float()))).item()
    )

    effective_scale = guidance_scale
    if (
        max_delta_ratio > 0
        and abs(guidance_scale) > 0
        and delta_rms > 0
    ):
        max_addition_rms = max_delta_ratio * max(positive_rms, 1e-8)
        requested_addition_rms = abs(guidance_scale) * delta_rms
        if requested_addition_rms > max_addition_rms:
            effective_scale *= max_addition_rms / requested_addition_rms

    guided = positive_denoised.clone()
    guided[positive_binder_indices] = (
        positive_binder + effective_scale * endpoint_delta
    )
    return guided, effective_scale, positive_rms, delta_rms


def _dng_update_log_posterior(
    log_posterior: torch.Tensor,
    sampled_next: torch.Tensor,
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
    variance: float,
    temperature: float,
    offset: float,
    p_min: float,
    p_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the DNG reverse-kernel likelihood-ratio posterior update.

    ``mean_a`` and ``mean_b`` are branch-specific next-step means in one
    common frame.  The likelihood is evaluated only on the coordinates passed
    by the caller (binder atoms in the pMHC sampler).
    """
    if sampled_next.shape != mean_a.shape or sampled_next.shape != mean_b.shape:
        raise ValueError("DNG sample and branch means must have equal shapes.")
    if sampled_next.numel() == 0:
        raise ValueError("DNG posterior requires at least one coordinate.")
    if not math.isfinite(float(variance)) or float(variance) <= 0:
        raise ValueError(f"DNG variance must be positive, got {variance}.")
    if not 0 < p_min < p_max < 1:
        raise ValueError(
            "DNG posterior clamps must satisfy 0 < p_min < p_max < 1."
        )

    sample_f = sampled_next.float().reshape(-1)
    mean_a_f = mean_a.float().reshape(-1)
    mean_b_f = mean_b.float().reshape(-1)
    distance_a = torch.sum(torch.square(sample_f - mean_a_f))
    distance_b = torch.sum(torch.square(sample_f - mean_b_f))
    distance_difference = distance_b - distance_a
    kernel_log_ratio = -0.5 * distance_difference / float(variance)
    offset_term = 0.5 * float(offset) / float(variance)
    increment = float(temperature) * kernel_log_ratio + offset_term
    updated = torch.clamp(
        log_posterior.float() + increment,
        min=math.log(p_min),
        max=math.log(p_max),
    )
    diagnostics = {
        "variance_angstrom2": float(variance),
        "distance_a_squared_angstrom": float(distance_a.item()),
        "distance_b_squared_angstrom": float(distance_b.item()),
        "distance_b_minus_a_squared_angstrom": float(
            distance_difference.item()
        ),
        "kernel_log_ratio": float(kernel_log_ratio.item()),
        "posterior_increment": float(increment.item()),
    }
    return updated.to(log_posterior), diagnostics


def _pa_gate_theta_and_scale(
    lhat: torch.Tensor,
    *,
    c: float,
    eta: float,
    gate_power: float,
    lhat_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pure-positive-A gate and its target guidance coefficient.

    For ``gamma_t = p_t^A (1 - theta_t)^gate_power`` and
    ``theta_t = sigmoid(log(c) + eta * lhat_t)``, the A-minus-B score
    coefficient is ``gate_power * theta_t * eta * lhat_temperature``.
    No unconditional/CFG branch enters this construction.
    """
    if c <= 0:
        raise ValueError(f"PA-gate c must be positive, got {c}.")
    for name, value in (
        ("eta", eta),
        ("gate_power", gate_power),
        ("lhat_temperature", lhat_temperature),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"PA-gate {name} must be finite, got {value}.")
    theta = torch.sigmoid(
        torch.as_tensor(
            math.log(float(c)),
            device=lhat.device,
            dtype=torch.float32,
        )
        + float(eta) * lhat.float()
    )
    scale = (
        float(gate_power)
        * theta
        * float(eta)
        * float(lhat_temperature)
    )
    return theta.to(lhat), scale.to(lhat)


def _ess_fraction_from_log_weights(log_weights: torch.Tensor) -> float:
    """Return ESS divided by particle count for finite log weights."""
    flat = log_weights.float().reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("ESS requires a nonempty finite log-weight vector.")
    weights = torch.softmax(flat, dim=0)
    return float(
        (1.0 / (flat.numel() * torch.sum(torch.square(weights)))).item()
    )


def _select_pa_rho_candidate(
    *,
    objective: str,
    total_ess: list[float],
    total_logw_variance: list[float],
    zero_index: int,
    min_ess_gain: float = 0.0,
    min_variance_gain: float = 0.0,
) -> int:
    """Select a post-clip rho candidate while retaining rho=0 fallback."""
    if len(total_ess) != len(total_logw_variance) or not total_ess:
        raise ValueError("Rho objective arrays must be nonempty and aligned.")
    if not 0 <= int(zero_index) < len(total_ess):
        raise ValueError(f"Invalid zero candidate index: {zero_index}.")
    if objective == "full_ess":
        selected = int(np.argmax(np.asarray(total_ess, dtype=float)))
        if total_ess[selected] < total_ess[zero_index] + float(min_ess_gain):
            return int(zero_index)
        return selected
    if objective == "log_weight_variance":
        selected = int(
            np.argmin(np.asarray(total_logw_variance, dtype=float))
        )
        if (
            total_logw_variance[selected]
            > total_logw_variance[zero_index] - float(min_variance_gain)
        ):
            return int(zero_index)
        return selected
    raise ValueError(f"Unknown PA rho objective: {objective}")


def _systematic_resample_indices(
    log_weights: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Systematically resample one ancestor index per particle."""
    flat = log_weights.float().reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError(
            "Systematic resampling requires nonempty finite log weights."
        )
    weights = torch.softmax(flat, dim=0)
    count = flat.numel()
    start = torch.rand(
        (),
        device=flat.device,
        generator=generator,
        dtype=torch.float32,
    ) / count
    positions = start + torch.arange(
        count,
        device=flat.device,
        dtype=torch.float32,
    ) / count
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False)


def _temper_log_weights_to_ess(
    log_weights: torch.Tensor,
    target_ess_fraction: float,
    *,
    iterations: int = 40,
) -> tuple[torch.Tensor, float]:
    """Temper log weights to a requested minimum ESS fraction.

    Returns ``(alpha * log_weights, alpha)`` with the largest alpha in [0, 1]
    whose ESS/N is at least the target. Alpha=1 leaves weights unchanged.
    """
    target = float(target_ess_fraction)
    if not 0 < target <= 1:
        raise ValueError(
            "Target tempered ESS fraction must lie in (0, 1], "
            f"got {target}."
        )
    flat = log_weights.float()
    if _ess_fraction_from_log_weights(flat) >= target:
        return log_weights, 1.0
    low = 0.0
    high = 1.0
    for _ in range(max(1, int(iterations))):
        middle = 0.5 * (low + high)
        if _ess_fraction_from_log_weights(middle * flat) >= target:
            low = middle
        else:
            high = middle
    return (low * log_weights), low


def _winsorize_centered_log_weights(
    log_weights: torch.Tensor,
    *,
    topk: int = 0,
    max_abs: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Winsorize only extreme deviations from the particle median.

    ``topk`` replaces the largest absolute deviations by the next-largest
    deviation.  ``max_abs`` optionally imposes a second absolute cap.  The
    common median is restored after clipping, so only relative weights change.
    """
    values = log_weights.float().reshape(-1)
    count = int(values.numel())
    if count == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("Log-weight clipping requires finite values.")
    if topk < 0 or topk >= count:
        raise ValueError(
            f"topk must lie in [0, {count - 1}], got {topk}."
        )
    if max_abs < 0:
        raise ValueError(f"max_abs must be nonnegative, got {max_abs}.")
    center_value = torch.median(values)
    centered = values - center_value
    threshold = float("inf")
    if topk > 0:
        ordered = torch.sort(torch.abs(centered), descending=True).values
        threshold = float(ordered[topk].item())
    if max_abs > 0:
        threshold = min(threshold, float(max_abs))
    if not math.isfinite(threshold):
        return log_weights, torch.zeros_like(values, dtype=torch.bool), threshold
    clipped_centered = torch.clamp(centered, min=-threshold, max=threshold)
    clipped = clipped_centered + center_value
    mask = torch.ne(clipped_centered, centered)
    return clipped.to(log_weights), mask, threshold


def _winsorize_centered_log_weight_rows(
    log_weights: torch.Tensor,
    *,
    topk: int = 0,
    max_abs: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply particle-wise winsorization independently to candidate rows."""
    if log_weights.ndim != 2:
        raise ValueError(
            "Candidate log weights must have shape [candidate, particle]."
        )
    clipped_rows = []
    mask_rows = []
    thresholds = []
    for row in log_weights:
        clipped, mask, threshold = _winsorize_centered_log_weights(
            row,
            topk=topk,
            max_abs=max_abs,
        )
        clipped_rows.append(clipped)
        mask_rows.append(mask)
        thresholds.append(threshold)
    return (
        torch.stack(clipped_rows),
        torch.stack(mask_rows),
        torch.tensor(
            thresholds,
            device=log_weights.device,
            dtype=torch.float32,
        ),
    )


def _pa_prop2_candidate_terms(
    *,
    rho1: float,
    rho2: torch.Tensor,
    gate_time: torch.Tensor,
    theta: torch.Tensor,
    gate_power: float,
    eta: float,
    lhat_temperature: float,
    variance: float,
    nu_a: torch.Tensor,
    d_y: torch.Tensor,
    g_y: torch.Tensor,
    zeta: torch.Tensor,
    delta0: torch.Tensor,
    jd_delta0: torch.Tensor,
    jvp_shrink_alpha: float,
) -> dict[str, torch.Tensor]:
    """Evaluate the retained two-field Proposition-2 expansion.

    The tensors after the particle dimension are flattened.  ``nu_a`` is
    the reference reverse displacement ``y - mean_A`` in local Gaussian
    units, while ``variance`` is the common ``h s`` scale.  The auxiliary
    forward process for BoltzGen's variance-exploding EDM is zero drift, so
    its scalar contribution is ``-variance * ||u||^2 / 2``.

    A single displacement and JVP are reused for every rho2 candidate, as in
    the fixed-displacement approximation following Proposition 2.
    """
    if rho2.ndim != 1:
        raise ValueError("rho2 candidates must be a one-dimensional tensor.")
    if not math.isfinite(float(variance)) or float(variance) <= 0:
        raise ValueError(f"Prop-2 variance must be positive, got {variance}.")
    particle_count = int(d_y.shape[0])
    for name, value in (
        ("nu_a", nu_a),
        ("g_y", g_y),
        ("delta0", delta0),
        ("jd_delta0", jd_delta0),
    ):
        if value.shape != d_y.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match d_y {d_y.shape}."
            )
    for name, value in (
        ("gate_time", gate_time),
        ("theta", theta),
        ("zeta", zeta),
    ):
        if value.shape != (particle_count,):
            raise ValueError(
                f"{name} must have shape [{particle_count}], got {value.shape}."
            )

    r1 = torch.as_tensor(
        float(rho1), device=d_y.device, dtype=torch.float32
    )
    r2 = rho2.float().reshape(-1, 1)
    d = d_y.float().flatten(1)
    g = g_y.float().flatten(1)
    nu = nu_a.float().flatten(1)
    delta = delta0.float().flatten(1)
    jd = jd_delta0.float().flatten(1)

    u = r1 * d[None, :, :] + r2[:, :, None] * g[None, :, :]
    q = u - g[None, :, :]
    forward_alpha = -0.5 * float(variance) * u.square().sum(dim=2)
    reverse_alpha = (
        (q * nu[None, :, :]).sum(dim=2)
        + 0.5 * float(variance) * q.square().sum(dim=2)
    )

    jd_quadratic = (delta * jd).sum(dim=1)
    d_projection2 = (d * delta).sum(dim=1).square()
    curvature_coefficient = (
        float(gate_power)
        * float(eta) ** 2
        * float(lhat_temperature) ** 2
        * theta.float()
        * (1.0 - theta.float())
    )
    target_curvature = (
        -0.5 * curvature_coefficient * d_projection2
    )[None, :].expand(int(r2.shape[0]), -1)
    proposal_curvature = (
        r2 * curvature_coefficient[None, :] * d_projection2[None, :]
    )
    jvp_raw = -(
        r1 + r2 * zeta.float()[None, :]
    ) * jd_quadratic[None, :]
    jvp = float(jvp_shrink_alpha) * jvp_raw
    gate = gate_time.float()[None, :] + target_curvature
    forward_zero = forward_alpha + proposal_curvature
    forward = forward_zero + jvp
    reverse = reverse_alpha
    total = gate + reverse + forward
    return {
        "total": total,
        "gate": gate,
        "reverse": reverse,
        "forward": forward,
        "forward_zero": forward_zero,
        "jvp": jvp,
        "jvp_raw": jvp_raw,
        "gate_time": gate_time.float()[None, :].expand(int(r2.shape[0]), -1),
        "target_curvature": target_curvature,
        "forward_alpha": forward_alpha,
        "reverse_alpha": reverse_alpha,
        "proposal_curvature": proposal_curvature,
        "jd_quadratic": jd_quadratic[None, :].expand(int(r2.shape[0]), -1),
        "d_projection2": d_projection2[None, :].expand(int(r2.shape[0]), -1),
        "curvature_coefficient": curvature_coefficient[None, :].expand(
            int(r2.shape[0]), -1
        ),
    }


def _pa_prop2_robustify_candidate_terms(
    terms: dict[str, torch.Tensor],
    *,
    gate_topk: int,
    gate_max_abs: float,
    jvp_topk: int,
    jvp_max_abs: float,
    kernel_topk: int,
    kernel_max_abs: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clip Prop-2 terms by source while preserving their decomposition."""
    gate_used, gate_mask, gate_threshold = (
        _winsorize_centered_log_weight_rows(
            terms["gate"], topk=gate_topk, max_abs=gate_max_abs
        )
    )
    jvp_used, jvp_mask, jvp_threshold = (
        _winsorize_centered_log_weight_rows(
            terms["jvp"], topk=jvp_topk, max_abs=jvp_max_abs
        )
    )
    kernel_raw = terms["reverse"] + terms["forward_zero"] + jvp_used
    kernel_used, kernel_mask, kernel_threshold = (
        _winsorize_centered_log_weight_rows(
            kernel_raw, topk=kernel_topk, max_abs=kernel_max_abs
        )
    )
    total_used = gate_used + kernel_used
    diagnostics = {
        "gate_used": gate_used,
        "gate_mask": gate_mask,
        "gate_threshold": gate_threshold,
        "jvp_used": jvp_used,
        "jvp_mask": jvp_mask,
        "jvp_threshold": jvp_threshold,
        "kernel_raw": kernel_raw,
        "kernel_used": kernel_used,
        "kernel_mask": kernel_mask,
        "kernel_threshold": kernel_threshold,
    }
    return total_used, diagnostics


"""
b - batch
h - heads
n - residue sequence length
m - atom sequence length
nw - windowed sequence length
ts - feature dimension (single)
tz - feature dimension (pairwise)
as - feature dimension (atompair)
az - feature dimension (atompair input)
"""


class DiffusionModule(Module):
    """Algorithm 20."""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_layers: int = 1,
        token_transformer_depth: int = 6,
        token_transformer_heads: int = 8,
        use_miniformer: bool = False,
        diffusion_pairformer_args: Dict[str, Any] = None,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        gaussian_random_3d_encoding_dim: int = 0,
        transformer_post_ln: bool = False,
        tfmr_s: Optional[int] = None,
        predict_res_type: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing
        if tfmr_s is None:
            tfmr_s = 2 * token_s
        self.tfmr_s = tfmr_s

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            tfmr_s=tfmr_s,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            gaussian_random_3d_encoding_dim=gaussian_random_3d_encoding_dim,
            transformer_post_layer_norm=transformer_post_ln,
            tfmr_s=tfmr_s,
            use_qk_norm=use_qk_norm,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(tfmr_s), LinearNoBias(tfmr_s, tfmr_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer_layers = nn.ModuleList()
        self.token_pairformer_layers = nn.ModuleList()

        self.token_transformer = DiffusionTransformer(
            dim=tfmr_s,
            dim_single_cond=tfmr_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            use_qk_norm=use_qk_norm,
        )

        self.a_norm = nn.LayerNorm(tfmr_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            tfmr_s=tfmr_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            predict_res_type=predict_res_type,
            use_qk_norm=use_qk_norm,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        # run token level transformations
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update, res_type = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return {
            "r_update": r_update,
            "token_a": a.detach(),
            "res_type": res_type,
        }


class OutTokenFeatUpdate(Module):
    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        mse_rotational_alignment: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        second_order_correction: bool = False,
        pass_resolved_mask_diff_train: bool = False,
        sampling_schedule: str = "af3",
        noise_scale_function: str = "constant",
        step_scale_function: str = "constant",
        min_noise_scale: float = 1.0,
        max_noise_scale: float = 1.0,
        noise_scale_alpha: float = 1.0,
        noise_scale_beta: float = 1.0,
        min_step_scale: float = 1.0,
        max_step_scale: float = 1.0,
        step_scale_alpha: float = 1.0,
        step_scale_beta: float = 1.0,
        time_dilation: float = 1.0,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
        pred_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

        if pred_threshold is None:
            # disable nucleation mask
            self.pred_sigma_thresh = float("inf")
        else:
            q = norm.ppf(pred_threshold)
            self.pred_sigma_thresh = self.sigma_data * exp(self.P_mean + self.P_std * q)

        self.num_sampling_steps = num_sampling_steps
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.noise_scale_function = noise_scale_function
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.noise_scale_alpha = noise_scale_alpha
        self.noise_scale_beta = noise_scale_beta
        self.step_scale = step_scale
        self.step_scale_function = step_scale_function
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale
        self.step_scale_alpha = step_scale_alpha
        self.step_scale_beta = step_scale_beta
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.mse_rotational_alignment = mse_rotational_alignment
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.second_order_correction = second_order_correction
        self.pass_resolved_mask_diff_train = pass_resolved_mask_diff_train
        self.token_s = score_model_args["token_s"]

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return (
            log(sigma / self.sigma_data) * 0.25
        )  # note here the AF3 authors divide by sigma_data but not EDM

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        if training and self.pass_resolved_mask_diff_train:
            res_mask = (
                network_condition_kwargs["feats"]["atom_resolved_mask"]
                .unsqueeze(-1)
                .float()
            )
            noised_atom_coords = noised_atom_coords * res_mask.repeat_interleave(
                network_condition_kwargs["multiplicity"], 0
            )

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )

        return denoised_coords, net_out

    def sample_schedule_af3(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample_schedule_dilated(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        ts = steps / (num_sampling_steps - 1)

        # remap to dilate a particular interval
        def dilate(ts, start, end, dilation):
            x = end - start
            l = start
            u = 1 - end
            assert (dilation - 1) * x <= l + u, "dilation too large"

            inv_dilation = 1 / dilation
            ratio = (l + u + (1 - dilation) * x) / (l + u)
            inv_ratio = 1 / ratio
            lprime = l * ratio
            uprime = u * ratio
            xprime = x * dilation

            lower_third = ts * inv_ratio
            middle_third = (ts - lprime) * inv_dilation + l
            upper_third = (ts - (lprime + xprime)) * inv_ratio + l + x
            return (
                (ts < lprime) * lower_third
                + ((ts >= lprime) & (ts < lprime + xprime)) * middle_third
                + (ts >= lprime + xprime) * upper_third
            )

        dilated_ts = dilate(
            ts, self.time_dilation_start, self.time_dilation_end, self.time_dilation
        )
        sigmas = (
            self.sigma_max**inv_rho
            + dilated_ts * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def beta_noise_scale_schedule(self, num_sampling_steps):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(1 - t, self.noise_scale_alpha, self.noise_scale_beta)
        )
        return (
            self.max_noise_scale
            + (self.min_noise_scale - self.max_noise_scale) * beta_cdf_weights
        )

    def beta_step_scale_schedule(self, num_sampling_steps=None):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(t, self.step_scale_alpha, self.step_scale_beta)
        )
        return (
            self.min_step_scale
            + (self.max_step_scale - self.min_step_scale) * beta_cdf_weights
        )

    def sample(
        self,
        atom_mask,  #: Bool['b m'] | None = None,
        num_sampling_steps=None,
        multiplicity=1,
        step_scale=None,
        noise_scale=None,
        inference_logging=False,
        **network_condition_kwargs,
    ):
        if _env_flag("BOLTZGEN_BINDER_NEG_GUIDANCE", "0"):
            if (
                os.environ.get(
                    "BOLTZGEN_NEG_GUIDANCE_MODE",
                    "fixed",
                ).strip().lower()
                == "pa_gate_smc"
            ):
                return self._sample_binder_pa_gate_smc(
                    atom_mask=atom_mask,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity,
                    step_scale=step_scale,
                    noise_scale=noise_scale,
                    inference_logging=inference_logging,
                    **network_condition_kwargs,
                )
            return self._sample_binder_negative_guidance(
                atom_mask=atom_mask,
                num_sampling_steps=num_sampling_steps,
                multiplicity=multiplicity,
                step_scale=step_scale,
                noise_scale=noise_scale,
                inference_logging=inference_logging,
                **network_condition_kwargs,
            )

        if self.training and self.step_scale_random is not None:
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale schedule: {self.noise_scale_function}"
            )
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)

        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_gammas_ss_ns = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        feats = network_condition_kwargs["feats"]

        # gradually denoise
        coords_traj = [atom_coords]
        x0_coords_traj = []
        for step_idx, (
            sigma_tm,
            sigma_t,
            gamma,
            step_scale,
            noise_scale,
        ) in optionally_tqdm(
            enumerate(sigmas_gammas_ss_ns),
            use_tqdm=inference_logging,
            desc="Denoising steps.",
        ):
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            # sigma_tm is sigma_t-1 and sigma_t is sigma_t
            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)

            atom_coords = center(atom_coords, atom_mask)

            if self.coordinate_augmentation_inference:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )

            eps = noise_scale * sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, net_out = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        **network_condition_kwargs,
                    ),
                )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # note here I believe there is a mistake in the AF3 paper where they use atom_coords instead of atom_coords_noisy
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            coords_traj.append(atom_coords_next)
            x0_coords_traj.append(atom_coords_denoised)
            atom_coords = atom_coords_next
        coords_traj.append(atom_coords)

        result = dict(
            sample_atom_coords=atom_coords,
            coords_traj=coords_traj,
            x0_coords_traj=x0_coords_traj,
        )

        return result

    def _sample_binder_negative_guidance(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        step_scale=None,
        noise_scale=None,
        inference_logging=False,
        **network_condition_kwargs,
    ):
        """Sample one binder with wanted-minus-unwanted pMHC guidance.

        The batch must contain exactly ``[wanted, unwanted]`` with the same
        tensor dimensions and token/entity order.  Per-residue atom occupancy
        may differ at the mutated peptide position (for example Pro versus
        Phe).  Semantically common atoms share one noisy state, while
        condition-only mutation atoms retain a native auxiliary trajectory.
        Each condition predicts its own denoised endpoint.  Directions are
        aligned on the MHC-I heavy-chain platform and contrasted only on the
        designed binder atoms; returned target/context atoms follow the
        wanted-condition update unchanged.
        """
        if self.training:
            raise ValueError(
                "Binder negative guidance is an inference-only sampler."
            )
        if multiplicity != 1:
            raise ValueError(
                "Binder negative guidance requires diffusion_batch_size=1. "
                "Parallelize independent wanted/unwanted pairs across GPUs."
            )

        feats = network_condition_kwargs["feats"]
        batch_size = int(feats["atom_pad_mask"].shape[0])
        if batch_size != 2:
            raise ValueError(
                "Binder negative guidance requires a two-item batch ordered "
                f"[wanted, unwanted], got batch size {batch_size}."
            )
        if atom_mask.ndim != 2 or atom_mask.shape[0] != 2:
            raise ValueError(
                "Binder negative guidance expects atom masks with shape "
                f"[2, atoms], got {tuple(atom_mask.shape)}."
            )
        # A peptide substitution can legitimately change atom14 occupancy.
        # What must remain identical is the token/entity schema and tensor
        # dimensions; the wanted atom mask controls the returned structure.
        for schema_key in (
            "token_pad_mask",
            "asym_id",
            "residue_index",
            "mol_type",
            "chain_design_mask",
        ):
            schema_value = feats[schema_key]
            if (
                schema_value.shape[0] != 2
                or not torch.equal(schema_value[0], schema_value[1])
            ):
                raise ValueError(
                    "Wanted and unwanted conditions have different token/"
                    f"entity schema at feats[{schema_key!r}]. Use identical "
                    "entity order and chain lengths."
                )

        semantic_positive, semantic_negative = _semantic_atom_pairs(feats)
        binder_masks = _binder_atom_mask(feats)
        binder_pair_mask = (
            binder_masks[0, semantic_positive]
            & binder_masks[1, semantic_negative]
        )
        binder_positive_indices = semantic_positive[binder_pair_mask]
        binder_negative_indices = semantic_negative[binder_pair_mask]
        if (
            binder_positive_indices.numel() == 0
            or binder_positive_indices.numel() != int(binder_masks[0].sum())
            or binder_negative_indices.numel() != int(binder_masks[1].sum())
        ):
            raise ValueError(
                "Wanted/unwanted designed-binder atoms do not have a complete "
                "semantic correspondence. Use the same binder placeholder "
                "chain in both specifications."
            )

        hla_masks = _hla_platform_atom_mask(
            feats,
            max_residues=int(
                _env_float("BOLTZGEN_NEG_GUIDANCE_HLA_RESIDUES", 180)
            ),
        )
        hla_pair_mask = (
            hla_masks[0, semantic_positive]
            & hla_masks[1, semantic_negative]
        )
        hla_positive_indices = semantic_positive[hla_pair_mask]
        hla_negative_indices = semantic_negative[hla_pair_mask]
        if hla_positive_indices.numel() < 3:
            raise ValueError(
                "Wanted/unwanted inputs do not share at least three resolved "
                "semantically paired MHC-I platform atoms."
            )

        conditioning_coords = feats["coords"]
        if conditioning_coords.ndim == 4:
            if conditioning_coords.shape[1] != 1:
                raise ValueError(
                    "Only one conditioning target conformation is supported."
                )
            conditioning_coords = conditioning_coords[:, 0]
        if conditioning_coords.ndim != 3:
            raise ValueError(
                "Expected conditioning coordinates with shape [2, atoms, 3]."
            )
        _, _, conditioning_fit_rmsd = _row_kabsch(
            conditioning_coords[1, hla_negative_indices],
            conditioning_coords[0, hla_positive_indices],
        )
        conditioning_raw_rmsd = torch.sqrt(
            torch.mean(
                torch.sum(
                    (
                        conditioning_coords[1, hla_negative_indices]
                        - conditioning_coords[0, hla_positive_indices]
                    )
                    ** 2,
                    dim=-1,
                )
            )
        )
        max_raw_rmsd = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_MAX_INPUT_HLA_RMSD",
            2.0,
        )
        if float(conditioning_raw_rmsd) > max_raw_rmsd:
            raise ValueError(
                "Wanted/unwanted pMHC files are not already in a common MHC "
                f"frame: raw HLA RMSD={float(conditioning_raw_rmsd):.3f} A "
                f"(limit {max_raw_rmsd:.3f} A). Pre-align on HLA residues "
                "1..180 before sampling."
            )

        base_guidance_scale = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_SCALE",
            0.5,
        )
        guidance_mode = os.environ.get(
            "BOLTZGEN_NEG_GUIDANCE_MODE",
            "fixed",
        ).strip().lower()
        if guidance_mode not in {"fixed", "dng", "pa_gate"}:
            raise ValueError(
                "BOLTZGEN_NEG_GUIDANCE_MODE must be fixed, dng, or pa_gate, got "
                f"{guidance_mode!r}."
            )
        schedule = os.environ.get(
            "BOLTZGEN_NEG_GUIDANCE_SCHEDULE",
            "constant",
        ).strip().lower()
        ramp_start = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_RAMP_START",
            0.2,
        )
        ramp_end = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_RAMP_END",
            0.65,
        )
        max_delta_ratio = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_MAX_DELTA_RATIO",
            1.0,
        )
        dng_prior = _env_float("BOLTZGEN_DNG_PRIOR", 0.01)
        dng_temperature = _env_float("BOLTZGEN_DNG_TEMPERATURE", 0.2)
        dng_offset = _env_float("BOLTZGEN_DNG_OFFSET", 0.0)
        dng_p_min = _env_float("BOLTZGEN_DNG_P_MIN", 1e-6)
        dng_p_max = _env_float("BOLTZGEN_DNG_P_MAX", 0.8)
        dng_posterior_eps = _env_float(
            "BOLTZGEN_DNG_POSTERIOR_EPS",
            1e-6,
        )
        dng_variance_floor = _env_float(
            "BOLTZGEN_DNG_VARIANCE_FLOOR",
            1e-4,
        )
        dng_variance_scale = _env_float(
            "BOLTZGEN_DNG_VARIANCE_SCALE",
            1.0,
        )
        pa_gate_c = _env_float("BOLTZGEN_PA_GATE_C", 9.0)
        pa_gate_eta = _env_float("BOLTZGEN_PA_GATE_ETA", 0.005)
        pa_gate_power = _env_float(
            "BOLTZGEN_PA_GATE_POWER",
            1925.92592593,
        )
        pa_lhat_temperature = _env_float(
            "BOLTZGEN_PA_LHAT_TEMPERATURE",
            0.75,
        )
        pa_lhat_clip = _env_float("BOLTZGEN_PA_LHAT_CLIP", 0.0)
        if not 0 < dng_prior < 1:
            raise ValueError("BOLTZGEN_DNG_PRIOR must be between zero and one.")
        if not 0 < dng_p_min < dng_p_max < 1:
            raise ValueError(
                "DNG clamps must satisfy 0 < p_min < p_max < 1."
            )
        if dng_variance_floor <= 0 or dng_variance_scale <= 0:
            raise ValueError(
                "DNG variance floor and scale must both be positive."
            )
        if pa_gate_c <= 0:
            raise ValueError("BOLTZGEN_PA_GATE_C must be positive.")
        if pa_gate_eta < 0 or pa_gate_power < 0:
            raise ValueError(
                "PA-gate eta and gate power must be nonnegative."
            )
        if pa_lhat_temperature <= 0 or pa_lhat_clip < 0:
            raise ValueError(
                "PA-gate lhat temperature must be positive and clip "
                "must be nonnegative."
            )
        dng_log_posterior = torch.tensor(
            math.log(dng_prior),
            device=self.device,
            dtype=torch.float32,
        )
        pa_lhat = torch.tensor(
            0.0,
            device=self.device,
            dtype=torch.float32,
        )
        log_stride = max(
            1,
            int(_env_float("BOLTZGEN_NEG_GUIDANCE_LOG_STRIDE", 50)),
        )
        seed_value = os.environ.get("BOLTZGEN_NEG_GUIDANCE_SEED")
        if seed_value is not None:
            seed = int(seed_value)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        diagnostics_dir_value = os.environ.get(
            "BOLTZGEN_NEG_GUIDANCE_DIAGNOSTICS_DIR",
            "",
        ).strip()
        diagnostics_dir: Path | None = None
        diagnostics_call_index = int(
            getattr(self, "_binder_negative_guidance_call_index", 0)
        )
        self._binder_negative_guidance_call_index = (
            diagnostics_call_index + 1
        )
        guidance_step_diagnostics: list[dict[str, Any]] = []
        if diagnostics_dir_value:
            diagnostics_dir = Path(
                diagnostics_dir_value
            ).expanduser().resolve()
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

        trace_dir_value = os.environ.get(
            "BOLTZGEN_NEG_GUIDANCE_TRACE_DIR",
            "",
        ).strip()
        trace_dir: Path | None = None
        if trace_dir_value:
            trace_dir = Path(trace_dir_value).expanduser().resolve()
            if trace_dir.exists() and any(trace_dir.iterdir()):
                raise FileExistsError(
                    "Negative-guidance trace directory is not empty: "
                    f"{trace_dir}. Use a new directory to avoid mixing runs."
                )
            trace_dir.mkdir(parents=True, exist_ok=True)

        num_sampling_steps = default(
            num_sampling_steps,
            self.num_sampling_steps,
        )
        if self.training and self.step_scale_random is not None:
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale function {self.noise_scale_function!r}."
            )

        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid sampling schedule {self.sampling_schedule!r}."
            )
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        schedule_rows = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        # Correlate noise only through semantic atom correspondence.  The two
        # native packed layouts can differ after the mutated peptide residue.
        atom_coords[1, semantic_negative] = atom_coords[
            0,
            semantic_positive,
        ]
        coords_traj = [atom_coords[0:1].clone()]
        x0_coords_traj = []

        if trace_dir is not None:
            atom_to_token_index = (
                feats["atom_to_token"].detach().to(torch.int8).argmax(dim=-1)
            )
            atom_name_codes = (
                feats["ref_atom_name_chars"]
                .detach()
                .to(torch.int8)
                .argmax(dim=-1)
            )
            metadata_arrays: dict[str, Any] = {
                "atom_mask_ab": atom_mask,
                "atom_pad_mask_ab": feats["atom_pad_mask"],
                "atom_resolved_mask_ab": feats["atom_resolved_mask"],
                "binder_mask_ab": binder_masks,
                "hla_platform_mask_ab": hla_masks,
                "semantic_a_indices": semantic_positive,
                "semantic_b_indices": semantic_negative,
                "binder_a_indices": binder_positive_indices,
                "binder_b_indices": binder_negative_indices,
                "hla_a_indices": hla_positive_indices,
                "hla_b_indices": hla_negative_indices,
                "atom_to_token_index_ab": atom_to_token_index,
                "atom_name_codes_ab": atom_name_codes,
                "token_pad_mask_ab": feats["token_pad_mask"],
                "token_asym_id_ab": feats["asym_id"],
                "token_residue_index_ab": feats["residue_index"],
                "token_mol_type_ab": feats["mol_type"],
                "token_chain_design_mask_ab": feats["chain_design_mask"],
                "conditioning_coords_ab": conditioning_coords,
                "initial_state_ab": atom_coords,
                "sigmas": sigmas,
                "gammas": gammas,
                "step_scales": step_scales,
                "noise_scales": noise_scales,
            }
            for optional_key in (
                "fake_atom_mask",
                "binding_type",
                "design_mask",
                "token_resolved_mask",
            ):
                if optional_key in feats:
                    metadata_arrays[f"{optional_key}_ab"] = feats[optional_key]
            _write_trace_npz(
                trace_dir / "metadata_arrays.npz",
                **metadata_arrays,
            )
            _write_trace_json(
                trace_dir / "metadata.json",
                {
                    "format": "boltzgen_binder_negative_guidance_trace",
                    "format_version": 2,
                    "condition_a": "wanted/positive pMHC",
                    "condition_b": "unwanted/negative pMHC",
                    "num_steps": len(schedule_rows),
                    "guidance_mode": guidance_mode,
                    "base_guidance_scale": base_guidance_scale,
                    "guidance_schedule": schedule,
                    "guidance_ramp_start": ramp_start,
                    "guidance_ramp_end": ramp_end,
                    "max_delta_ratio": max_delta_ratio,
                    "dng": {
                        "prior": dng_prior,
                        "temperature": dng_temperature,
                        "offset": dng_offset,
                        "p_min": dng_p_min,
                        "p_max": dng_p_max,
                        "posterior_eps": dng_posterior_eps,
                        "variance_floor_angstrom2": dng_variance_floor,
                        "variance_scale": dng_variance_scale,
                        "posterior_coordinates": "binder atoms only",
                        "branch_mean_frame": (
                            "each A/B branch mean rigidly aligned to the "
                            "sampled next A state on the HLA platform"
                        ),
                    },
                    "pa_gate": {
                        "c": pa_gate_c,
                        "eta": pa_gate_eta,
                        "gate_power": pa_gate_power,
                        "lhat_temperature": pa_lhat_temperature,
                        "lhat_clip": pa_lhat_clip,
                        "target": "p_A * (1 - theta)^gate_power",
                        "theta": "sigmoid(log(c) + eta*lhat)",
                        "base_branch": "condition A",
                        "negative_branch": "condition B",
                        "unconditional_or_cfg_base_branch": False,
                        "transfer_scope": (
                            "single-particle target-equivalent PA gate; "
                            "does not include SMC resampling, centered proposal, "
                            "or JVP/rho optimization"
                        ),
                    },
                    "seed": (
                        int(seed_value) if seed_value is not None else None
                    ),
                    "hla_alignment_residues": int(
                        _env_float(
                            "BOLTZGEN_NEG_GUIDANCE_HLA_RESIDUES",
                            180,
                        )
                    ),
                    "input_hla_raw_rmsd_angstrom": float(
                        conditioning_raw_rmsd
                    ),
                    "input_hla_fit_rmsd_angstrom": float(
                        conditioning_fit_rmsd
                    ),
                    "guidance_equation": (
                        "v_A=(noisy_A-denoised_A)/t; "
                        "v_B_A=((noisy_B-denoised_B)/t)@R_HLA; "
                        "v_guided_A[binder]=v_A+effective_scale*(v_A-v_B_A); "
                        "for DNG requested_scale=lambda0*p_t/(1-p_t), "
                        "for PA gate requested_scale="
                        "gate_power*theta_t*eta*lhat_temperature, "
                        "with A-conditioned v_A as the unguided base; "
                        "guided_A[binder]=noisy_A-t*v_guided_A; "
                        "guided_A[context] = denoised_A[context]"
                    ),
                    "step_file_pattern": "step_0000.npz",
                    "notes": [
                        "A and B are outputs of one shared model checkpoint "
                        "on a two-condition batch.",
                        "All coordinate arrays retain native padded atom-slot "
                        "layouts; use metadata semantic index maps to compare "
                        "A and B.",
                        "B_aligned coordinate arrays apply the same HLA "
                        "rotation and translation to noisy origin and endpoint.",
                        "direction arrays are EDM solver directions "
                        "(noisy - denoised) / t_hat, not forces or affinity "
                        "gradients.",
                        "cfg_delta_a is the endpoint-equivalent of the "
                        "frame-consistent velocity contrast; "
                        "cfg_delta_raw_frame_a + "
                        "cfg_alignment_component_a = cfg_delta_a.",
                        "noisy_for_solver_raw_endpoint_ab is the "
                        "counterfactual reverse-diffusion rigid alignment "
                        "against raw A/B denoiser endpoints instead of the "
                        "guided A endpoint.",
                    ],
                },
            )
            print(
                f"[BINDER_NEG_GUIDANCE_TRACE] writing {len(schedule_rows)} "
                f"steps to {trace_dir}",
                flush=True,
            )

        print(
            "[BINDER_NEG_GUIDANCE] "
            f"mode={guidance_mode} scale={base_guidance_scale:g} "
            f"schedule={schedule} "
            f"binder_atoms={binder_positive_indices.numel()} "
            f"semantic_atom_pairs={semantic_positive.numel()} "
            f"hla_fit_atoms={hla_positive_indices.numel()} "
            f"input_hla_raw_rmsd={float(conditioning_raw_rmsd):.3f}A "
            f"input_hla_fit_rmsd={float(conditioning_fit_rmsd):.3f}A "
            f"dng_prior={dng_prior:g} dng_temp={dng_temperature:g} "
            f"dng_offset={dng_offset:g} "
            f"pa_c={pa_gate_c:g} pa_eta={pa_gate_eta:g} "
            f"pa_power={pa_gate_power:g} pa_temp={pa_lhat_temperature:g} "
            f"seed={seed_value if seed_value is not None else 'framework'}",
            flush=True,
        )

        for step_idx, (
            sigma_tm,
            sigma_t,
            gamma,
            step_scale_value,
            noise_scale_value,
        ) in optionally_tqdm(
            enumerate(schedule_rows),
            use_tqdm=inference_logging,
            desc="Binder negative-guidance denoising.",
        ):
            sigma_tm = float(sigma_tm.item())
            sigma_t = float(sigma_t.item())
            gamma = float(gamma.item())
            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scale_value**2 * (t_hat**2 - sigma_tm**2)
            state_input_ab = (
                atom_coords.clone() if trace_dir is not None else None
            )

            # Native packed layouts remain condition-specific.  Keep both in
            # one HLA frame and tie every semantically common noisy atom;
            # condition-only mutation atoms retain an auxiliary trajectory.
            atom_coords = center(atom_coords, atom_mask)
            (
                state_rotation,
                state_translation,
                state_hla_fit_rmsd,
            ) = _row_kabsch(
                atom_coords[1, hla_negative_indices],
                atom_coords[0, hla_positive_indices],
            )
            atom_coords[1] = (
                atom_coords[1] @ state_rotation + state_translation
            )
            atom_coords[1, semantic_negative] = atom_coords[
                0,
                semantic_positive,
            ]
            augmentation_rotation = torch.eye(
                3,
                device=atom_coords.device,
                dtype=atom_coords.dtype,
            )
            augmentation_translation = torch.zeros(
                3,
                device=atom_coords.device,
                dtype=atom_coords.dtype,
            )
            if self.coordinate_augmentation_inference:
                random_rotation, random_translation = compute_random_augmentation(
                    1,
                    device=atom_coords.device,
                    dtype=atom_coords.dtype,
                )
                augmentation_rotation = random_rotation[0]
                augmentation_translation = random_translation[0]
                atom_coords = (
                    atom_coords @ augmentation_rotation
                    + augmentation_translation
                )
            state_for_model_ab = (
                atom_coords.clone() if trace_dir is not None else None
            )

            eps = (
                noise_scale_value
                * sqrt(float(noise_var))
                * torch.randn(shape, device=self.device)
            )
            eps[1, semantic_negative] = eps[0, semantic_positive]
            atom_coords_noisy = atom_coords + eps
            if not torch.allclose(
                atom_coords_noisy[0, binder_positive_indices],
                atom_coords_noisy[1, binder_negative_indices],
                atol=1e-5,
                rtol=0,
            ):
                raise RuntimeError(
                    "Wanted/unwanted denoisers did not receive the same noisy "
                    "binder state."
                )

            with torch.no_grad():
                denoised, net_out = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=1,
                        **network_condition_kwargs,
                    ),
                )

            positive_denoised = denoised[0]
            negative_denoised = denoised[1]
            (
                negative_to_positive_rotation,
                negative_to_positive_translation,
                hla_fit_rmsd,
            ) = _row_kabsch(
                negative_denoised[hla_negative_indices],
                positive_denoised[hla_positive_indices],
            )
            negative_denoised_aligned = (
                negative_denoised @ negative_to_positive_rotation
                + negative_to_positive_translation
            )
            guidance_rotation_degrees = _rotation_angle_degrees(
                negative_to_positive_rotation
            )
            guidance_translation_norm = float(
                torch.linalg.vector_norm(
                    negative_to_positive_translation.float()
                ).item()
            )
            progress = step_idx / max(len(schedule_rows) - 1, 1)
            dng_posterior_before = float(dng_log_posterior.exp().item())
            pa_lhat_before = float(pa_lhat.item())
            pa_theta_before: float | None = None
            if guidance_mode == "dng":
                requested_scale = (
                    base_guidance_scale
                    * dng_posterior_before
                    / max(1.0 - dng_posterior_before, dng_posterior_eps)
                )
            elif guidance_mode == "pa_gate":
                pa_theta_tensor, pa_scale_tensor = (
                    _pa_gate_theta_and_scale(
                        pa_lhat,
                        c=pa_gate_c,
                        eta=pa_gate_eta,
                        gate_power=pa_gate_power,
                        lhat_temperature=pa_lhat_temperature,
                    )
                )
                pa_theta_before = float(pa_theta_tensor.item())
                requested_scale = float(pa_scale_tensor.item())
            else:
                requested_scale = _guidance_scale_at_progress(
                    base_scale=base_guidance_scale,
                    progress=progress,
                    schedule=schedule,
                    start=ramp_start,
                    end=ramp_end,
                )
            guided_denoised, effective_scale, positive_rms, delta_rms = (
                _compose_binder_denoised(
                    positive_denoised=positive_denoised,
                    negative_denoised=negative_denoised,
                    positive_noisy=atom_coords_noisy[0],
                    negative_noisy=atom_coords_noisy[1],
                    t_hat=t_hat,
                    positive_binder_indices=binder_positive_indices,
                    negative_binder_indices=binder_negative_indices,
                    negative_to_positive_rotation=negative_to_positive_rotation,
                    guidance_scale=requested_scale,
                    max_delta_ratio=max_delta_ratio,
                )
            )
            positive_binder_velocity = (
                atom_coords_noisy[0, binder_positive_indices]
                - positive_denoised[binder_positive_indices]
            ) / t_hat
            negative_binder_velocity_aligned = (
                (
                    atom_coords_noisy[1, binder_negative_indices]
                    - negative_denoised[binder_negative_indices]
                )
                / t_hat
            ) @ negative_to_positive_rotation
            binder_velocity_delta = (
                positive_binder_velocity
                - negative_binder_velocity_aligned
            )
            cfg_delta_a = torch.zeros_like(positive_denoised)
            cfg_delta_a[binder_positive_indices] = (
                -t_hat * binder_velocity_delta
            )
            cfg_delta_raw_frame_a = torch.zeros_like(positive_denoised)
            cfg_delta_raw_frame_a[binder_positive_indices] = (
                positive_denoised[binder_positive_indices]
                - negative_denoised[binder_negative_indices]
            )
            cfg_alignment_component_a = torch.zeros_like(
                positive_denoised
            )
            cfg_alignment_component_a[binder_positive_indices] = (
                cfg_delta_a[binder_positive_indices]
                - cfg_delta_raw_frame_a[binder_positive_indices]
            )
            cfg_velocity_delta_a = torch.zeros_like(positive_denoised)
            cfg_velocity_delta_a[binder_positive_indices] = (
                binder_velocity_delta
            )
            guidance_step_diagnostics.append(
                {
                    "step": step_idx,
                    "progress": progress,
                    "requested_guidance_scale": requested_scale,
                    "effective_guidance_scale": effective_scale,
                    "guidance_mode": guidance_mode,
                    "dng_posterior_before": (
                        dng_posterior_before
                        if guidance_mode == "dng"
                        else None
                    ),
                    "pa_lhat_before": (
                        pa_lhat_before
                        if guidance_mode == "pa_gate"
                        else None
                    ),
                    "pa_theta_before": (
                        pa_theta_before
                        if guidance_mode == "pa_gate"
                        else None
                    ),
                    "guidance_rotation_angle_degrees": (
                        guidance_rotation_degrees
                    ),
                    "guidance_translation_norm_angstrom": (
                        guidance_translation_norm
                    ),
                    "rotation_angle_degrees": guidance_rotation_degrees,
                    "translation_norm_angstrom": guidance_translation_norm,
                    "HLA_fit_rmsd_angstrom": float(hla_fit_rmsd),
                    "rotation_matrix_row_vector": _trace_array(
                        negative_to_positive_rotation
                    ).tolist(),
                    "translation_row_vector_angstrom": _trace_array(
                        negative_to_positive_translation
                    ).tolist(),
                }
            )
            negative_noisy_aligned = (
                atom_coords_noisy[1] @ negative_to_positive_rotation
                + negative_to_positive_translation
            )
            if not torch.equal(
                guided_denoised[~binder_masks[0]],
                positive_denoised[~binder_masks[0]],
            ):
                raise RuntimeError(
                    "Negative guidance modified wanted pMHC context atoms."
                )

            denoised_pair = torch.stack(
                [guided_denoised, negative_denoised],
                dim=0,
            )
            need_raw_endpoint = (
                trace_dir is not None
                or guidance_mode in {"dng", "pa_gate"}
            )
            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    noisy_for_step = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        denoised_pair.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )
                    if need_raw_endpoint:
                        noisy_for_raw_endpoint = weighted_rigid_align(
                            atom_coords_noisy.float(),
                            denoised.float(),
                            atom_mask.float(),
                            atom_mask.float(),
                        )
                noisy_for_step = noisy_for_step.to(denoised_pair)
                if need_raw_endpoint:
                    noisy_for_raw_endpoint = noisy_for_raw_endpoint.to(
                        denoised_pair
                    )
            else:
                noisy_for_step = atom_coords_noisy
                if need_raw_endpoint:
                    noisy_for_raw_endpoint = atom_coords_noisy

            denoised_over_sigma = (
                noisy_for_step - denoised_pair
            ) / t_hat
            raw_endpoint_direction_ab = (
                (noisy_for_raw_endpoint - denoised) / t_hat
                if need_raw_endpoint
                else None
            )
            raw_direction_ab = (atom_coords_noisy - denoised) / t_hat
            atom_coords_next = (
                noisy_for_step
                + step_scale_value
                * (sigma_t - t_hat)
                * denoised_over_sigma
            )
            if guidance_mode in {"dng", "pa_gate"}:
                branch_step = float(step_scale_value) * (sigma_t - t_hat)
                mean_a = (
                    noisy_for_raw_endpoint[0]
                    + branch_step * raw_endpoint_direction_ab[0]
                )
                mean_b = (
                    noisy_for_raw_endpoint[1]
                    + branch_step * raw_endpoint_direction_ab[1]
                )
                sampled_next_a = atom_coords_next[0]
                mean_a_rotation, mean_a_translation, _ = _row_kabsch(
                    mean_a[hla_positive_indices],
                    sampled_next_a[hla_positive_indices],
                )
                mean_b_rotation, mean_b_translation, _ = _row_kabsch(
                    mean_b[hla_negative_indices],
                    sampled_next_a[hla_positive_indices],
                )
                mean_a_aligned = (
                    mean_a @ mean_a_rotation + mean_a_translation
                )
                mean_b_aligned = (
                    mean_b @ mean_b_rotation + mean_b_translation
                )
                dng_variance = max(
                    dng_variance_floor,
                    branch_step**2 * dng_variance_scale,
                )
                if guidance_mode == "dng":
                    dng_log_posterior, dng_terms = (
                        _dng_update_log_posterior(
                            log_posterior=dng_log_posterior,
                            sampled_next=sampled_next_a[
                                binder_positive_indices
                            ],
                            mean_a=mean_a_aligned[binder_positive_indices],
                            mean_b=mean_b_aligned[binder_negative_indices],
                            variance=dng_variance,
                            temperature=dng_temperature,
                            offset=dng_offset,
                            p_min=dng_p_min,
                            p_max=dng_p_max,
                        )
                    )
                    guidance_step_diagnostics[-1].update(
                        {
                            "dng_posterior_after": float(
                                dng_log_posterior.exp().item()
                            ),
                            "dng_dynamic_scale_requested": requested_scale,
                            **dng_terms,
                        }
                    )
                else:
                    _, pa_terms = _dng_update_log_posterior(
                        log_posterior=torch.zeros_like(pa_lhat),
                        sampled_next=sampled_next_a[
                            binder_positive_indices
                        ],
                        mean_a=mean_a_aligned[binder_positive_indices],
                        mean_b=mean_b_aligned[binder_negative_indices],
                        variance=dng_variance,
                        temperature=1.0,
                        offset=0.0,
                        p_min=1e-30,
                        p_max=1.0 - 1e-7,
                    )
                    pa_lhat = pa_lhat + (
                        float(pa_lhat_temperature)
                        * float(pa_terms["kernel_log_ratio"])
                    )
                    if pa_lhat_clip > 0:
                        pa_lhat = torch.clamp(
                            pa_lhat,
                            min=-pa_lhat_clip,
                            max=pa_lhat_clip,
                        )
                    guidance_step_diagnostics[-1].update(
                        {
                            "pa_lhat_after": float(pa_lhat.item()),
                            "pa_dynamic_scale_requested": requested_scale,
                            "pa_transition_variance_angstrom2": dng_variance,
                            "pa_kernel_log_ratio_untempered": pa_terms[
                                "kernel_log_ratio"
                            ],
                        }
                    )
            state_next_before_b_frame_ab = (
                atom_coords_next.clone() if trace_dir is not None else None
            )
            # Re-express the auxiliary unwanted trajectory in the wanted HLA
            # frame, then restore one shared state for every semantically
            # common atom.  Unwanted-only mutation atoms retain their own
            # solver trajectory.  No unwanted context update enters branch 0.
            (
                next_rotation,
                next_translation,
                next_hla_fit_rmsd,
            ) = _row_kabsch(
                atom_coords_next[1, hla_negative_indices],
                atom_coords_next[0, hla_positive_indices],
            )
            atom_coords_next[1] = (
                atom_coords_next[1] @ next_rotation + next_translation
            )
            atom_coords_next[1, semantic_negative] = atom_coords_next[
                0,
                semantic_positive,
            ]

            if trace_dir is not None:
                trace_values: dict[str, Any] = {
                    "step_index": step_idx,
                    "progress": progress,
                    "sigma_tm": sigma_tm,
                    "sigma_t": sigma_t,
                    "gamma": gamma,
                    "t_hat": t_hat,
                    "noise_variance": noise_var,
                    "step_scale": step_scale_value,
                    "noise_scale": noise_scale_value,
                    "requested_guidance_scale": requested_scale,
                    "effective_guidance_scale": effective_scale,
                    "positive_motion_rms": positive_rms,
                    "cfg_delta_rms": delta_rms,
                    "state_hla_fit_rmsd": state_hla_fit_rmsd,
                    "denoised_hla_fit_rmsd": hla_fit_rmsd,
                    "next_hla_fit_rmsd": next_hla_fit_rmsd,
                    "state_input_ab": state_input_ab,
                    "state_for_model_ab": state_for_model_ab,
                    "added_noise_ab": eps,
                    "noisy_model_input_ab": atom_coords_noisy,
                    "denoised_raw_ab": denoised,
                    "denoised_b_aligned_to_a": (
                        negative_denoised_aligned
                    ),
                    "cfg_delta_a": cfg_delta_a,
                    "cfg_velocity_delta_a": cfg_velocity_delta_a,
                    "cfg_delta_raw_frame_a": cfg_delta_raw_frame_a,
                    "cfg_alignment_component_a": (
                        cfg_alignment_component_a
                    ),
                    "denoised_cfg_a": guided_denoised,
                    "noisy_b_aligned_by_denoised_transform": (
                        negative_noisy_aligned
                    ),
                    "noisy_for_solver_ab": noisy_for_step,
                    "noisy_for_solver_raw_endpoint_ab": (
                        noisy_for_raw_endpoint
                    ),
                    "raw_direction_ab": raw_direction_ab,
                    "solver_direction_ab": denoised_over_sigma,
                    "solver_direction_raw_endpoint_ab": (
                        raw_endpoint_direction_ab
                    ),
                    "state_next_before_b_frame_ab": (
                        state_next_before_b_frame_ab
                    ),
                    "state_next_ab": atom_coords_next,
                    "state_b_to_a_rotation": state_rotation,
                    "state_b_to_a_translation": state_translation,
                    "augmentation_rotation": augmentation_rotation,
                    "augmentation_translation": augmentation_translation,
                    "denoised_b_to_a_rotation": (
                        negative_to_positive_rotation
                    ),
                    "denoised_b_to_a_translation": (
                        negative_to_positive_translation
                    ),
                    "next_b_to_a_rotation": next_rotation,
                    "next_b_to_a_translation": next_translation,
                    "c_in": self.c_in(
                        torch.tensor(t_hat, device=self.device)
                    ),
                    "c_skip": self.c_skip(
                        torch.tensor(t_hat, device=self.device)
                    ),
                    "c_out": self.c_out(
                        torch.tensor(t_hat, device=self.device)
                    ),
                    "c_noise": self.c_noise(
                        torch.tensor(t_hat, device=self.device)
                    ),
                }
                for output_name, output_value in net_out.items():
                    trace_values[f"network_{output_name}_ab"] = output_value
                _write_trace_npz(
                    trace_dir / f"step_{step_idx:04d}.npz",
                    **trace_values,
                )

            atom_coords = atom_coords_next
            coords_traj.append(atom_coords[0:1].clone())
            x0_coords_traj.append(guided_denoised[None].clone())

            if (
                step_idx % log_stride == 0
                or step_idx == len(schedule_rows) - 1
            ):
                print(
                    "[BINDER_NEG_GUIDANCE_STEP] "
                    f"step={step_idx}/{len(schedule_rows) - 1} "
                    f"sigma={t_hat:.5g} requested_scale={requested_scale:.4g} "
                    f"effective_scale={effective_scale:.4g} "
                    + (
                        f"dng_p={dng_posterior_before:.4g}->"
                        f"{float(dng_log_posterior.exp().item()):.4g} "
                        if guidance_mode == "dng"
                        else ""
                    )
                    + (
                        f"pa_theta={pa_theta_before:.4g} "
                        f"pa_lhat={pa_lhat_before:.4g}->"
                        f"{float(pa_lhat.item()):.4g} "
                        if guidance_mode == "pa_gate"
                        else ""
                    )
                    +
                    f"rotation={guidance_rotation_degrees:.5f}deg "
                    f"translation={guidance_translation_norm:.5f}A "
                    f"hla_fit_rmsd={float(hla_fit_rmsd):.3f}A "
                    f"wanted_motion_rms={positive_rms:.4g} "
                    f"contrast_rms={delta_rms:.4g}",
                    flush=True,
                )

        # The second branch exists only to evaluate the unwanted condition.
        # Returning wanted branch 0 avoids invalid paired-writer reconstruction
        # and ensures inverse folding sees exactly one guided backbone.
        if diagnostics_dir is not None:
            rotation_angles = np.asarray(
                [
                    item["rotation_angle_degrees"]
                    for item in guidance_step_diagnostics
                ],
                dtype=float,
            )
            active_angles = np.asarray(
                [
                    item["rotation_angle_degrees"]
                    for item in guidance_step_diagnostics
                    if abs(float(item["requested_guidance_scale"])) > 0
                ],
                dtype=float,
            )
            full_angles = np.asarray(
                [
                    item["rotation_angle_degrees"]
                    for item in guidance_step_diagnostics
                    if math.isclose(
                        float(item["requested_guidance_scale"]),
                        base_guidance_scale,
                        rel_tol=1e-6,
                        abs_tol=1e-8,
                    )
                ],
                dtype=float,
            )

            def angle_summary(values: np.ndarray) -> dict[str, Any]:
                if values.size == 0:
                    return {"count": 0}
                return {
                    "count": int(values.size),
                    "minimum_degrees": float(np.min(values)),
                    "median_degrees": float(np.median(values)),
                    "mean_degrees": float(np.mean(values)),
                    "maximum_degrees": float(np.max(values)),
                }

            diagnostics_path = (
                diagnostics_dir
                / f"sample_{diagnostics_call_index:03d}.json"
            )
            _write_trace_json(
                diagnostics_path,
                {
                    "format": (
                        "boltzgen_frame_consistent_negative_guidance_"
                        "diagnostics"
                    ),
                    "format_version": 1,
                    "sample_call_index": diagnostics_call_index,
                    "guidance_mode": guidance_mode,
                    "seed": (
                        int(seed_value) if seed_value is not None else None
                    ),
                    "base_guidance_scale": base_guidance_scale,
                    "dng": (
                        {
                            "lambda0": base_guidance_scale,
                            "prior": dng_prior,
                            "temperature": dng_temperature,
                            "offset": dng_offset,
                            "p_min": dng_p_min,
                            "p_max": dng_p_max,
                            "posterior_eps": dng_posterior_eps,
                            "variance_floor_angstrom2": dng_variance_floor,
                            "variance_scale": dng_variance_scale,
                            "posterior_coordinates": "binder atoms only",
                            "base_branch": "condition A",
                            "negative_branch": "condition B",
                            "unconditional_or_cfg_base_branch": False,
                        }
                        if guidance_mode == "dng"
                        else None
                    ),
                    "pa_gate": (
                        {
                            "c": pa_gate_c,
                            "eta": pa_gate_eta,
                            "gate_power": pa_gate_power,
                            "lhat_temperature": pa_lhat_temperature,
                            "lhat_clip": pa_lhat_clip,
                            "final_lhat": float(pa_lhat.item()),
                            "variance_floor_angstrom2": dng_variance_floor,
                            "variance_scale": dng_variance_scale,
                            "posterior_coordinates": "binder atoms only",
                            "base_branch": "condition A",
                            "negative_branch": "condition B",
                            "unconditional_or_cfg_base_branch": False,
                            "transfer_scope": (
                                "single-particle target-equivalent PA gate; "
                                "SMC/centered-kappa/JVP-rho disabled"
                            ),
                        }
                        if guidance_mode == "pa_gate"
                        else None
                    ),
                    "input_HLA_raw_rmsd_angstrom": float(
                        conditioning_raw_rmsd
                    ),
                    "input_HLA_fit_rmsd_angstrom": float(
                        conditioning_fit_rmsd
                    ),
                    "rotation_summary_all_steps": angle_summary(
                        rotation_angles
                    ),
                    "rotation_summary_active_guidance_steps": angle_summary(
                        active_angles
                    ),
                    "rotation_summary_full_guidance_steps": angle_summary(
                        full_angles
                    ),
                    "steps": guidance_step_diagnostics,
                },
            )
            print(
                "[BINDER_NEG_GUIDANCE_DIAGNOSTICS] "
                f"{diagnostics_path}",
                flush=True,
            )
        if trace_dir is not None:
            _write_trace_npz(
                trace_dir / "final_state.npz",
                final_state_ab=atom_coords,
                final_sample_a=atom_coords[0:1],
            )
            _write_trace_json(
                trace_dir / "complete.json",
                {
                    "complete": True,
                    "num_steps_written": len(schedule_rows),
                    "final_step_file": (
                        f"step_{len(schedule_rows) - 1:04d}.npz"
                    ),
                },
            )
            print(
                f"[BINDER_NEG_GUIDANCE_TRACE] complete: {trace_dir}",
                flush=True,
            )
        return {
            "sample_atom_coords": atom_coords[0:1],
            "coords_traj": coords_traj,
            "x0_coords_traj": x0_coords_traj,
        }

    def _sample_binder_pa_gate_smc(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        step_scale=None,
        noise_scale=None,
        inference_logging=False,
        **network_condition_kwargs,
    ):
        """Two-field Proposition-2 PA-gate SMC for BoltzGen's EDM sampler.

        The auxiliary field is ``u = rho1 * d + rho2 * g`` with
        ``g = zeta * d``.  Its matched backward proposal has mean
        ``mean_A + hs * (g - u)``.  Candidate rho2 values and the propagated
        incremental weights both use the retained
        ``h*alpha + Delta' Gamma Delta`` expansion from Proposition 2.  One
        finite-difference JVP at the rho2=0 displacement is shared by all
        rho2 candidates.  For the variance-exploding EDM forward process the
        reference forward drift is zero and ``hs`` is represented by the
        configured local sigma-step variance.
        """
        if self.training:
            raise ValueError("PA-gate SMC is inference only.")
        particle_count = int(multiplicity)
        if particle_count < 2:
            raise ValueError(
                "PA-gate SMC requires diffusion_samples >= 2."
            )

        feats = network_condition_kwargs["feats"]
        if int(feats["atom_pad_mask"].shape[0]) != 2:
            raise ValueError(
                "PA-gate SMC requires [wanted, unwanted] input conditions."
            )
        for schema_key in (
            "token_pad_mask",
            "asym_id",
            "residue_index",
            "mol_type",
            "chain_design_mask",
        ):
            if not torch.equal(feats[schema_key][0], feats[schema_key][1]):
                raise ValueError(
                    "Wanted/unwanted conditions have different schemas at "
                    f"feats[{schema_key!r}]."
                )

        semantic_a, semantic_b = _semantic_atom_pairs(feats)
        binder_masks = _binder_atom_mask(feats)
        binder_pair_mask = (
            binder_masks[0, semantic_a] & binder_masks[1, semantic_b]
        )
        binder_a = semantic_a[binder_pair_mask]
        binder_b = semantic_b[binder_pair_mask]
        if (
            binder_a.numel() == 0
            or binder_a.numel() != int(binder_masks[0].sum())
            or binder_b.numel() != int(binder_masks[1].sum())
        ):
            raise ValueError("PA-gate SMC binder atom pairing is incomplete.")
        hla_masks = _hla_platform_atom_mask(
            feats,
            max_residues=int(
                _env_float("BOLTZGEN_NEG_GUIDANCE_HLA_RESIDUES", 180)
            ),
        )
        hla_pair_mask = (
            hla_masks[0, semantic_a] & hla_masks[1, semantic_b]
        )
        hla_a = semantic_a[hla_pair_mask]
        hla_b = semantic_b[hla_pair_mask]
        if hla_a.numel() < 3:
            raise ValueError("PA-gate SMC HLA platform pairing is incomplete.")

        conditioning_coords = feats["coords"]
        if conditioning_coords.ndim == 4:
            conditioning_coords = conditioning_coords[:, 0]
        _, _, conditioning_fit_rmsd = _row_kabsch(
            conditioning_coords[1, hla_b],
            conditioning_coords[0, hla_a],
        )
        conditioning_raw_rmsd = torch.sqrt(
            torch.mean(
                torch.sum(
                    (
                        conditioning_coords[1, hla_b]
                        - conditioning_coords[0, hla_a]
                    )
                    ** 2,
                    dim=-1,
                )
            )
        )
        max_raw_rmsd = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_MAX_INPUT_HLA_RMSD",
            2.0,
        )
        if float(conditioning_raw_rmsd) > max_raw_rmsd:
            raise ValueError(
                "PA-gate SMC inputs are not in a common HLA frame: "
                f"raw RMSD={float(conditioning_raw_rmsd):.3f} A."
            )

        c = _env_float("BOLTZGEN_PA_GATE_C", 9.0)
        eta = _env_float("BOLTZGEN_PA_GATE_ETA", 0.005)
        gate_power = _env_float(
            "BOLTZGEN_PA_GATE_POWER",
            1925.92592593,
        )
        lhat_temp = _env_float(
            "BOLTZGEN_PA_LHAT_TEMPERATURE",
            0.75,
        )
        lhat_clip = _env_float("BOLTZGEN_PA_LHAT_CLIP", 0.0)
        rho1 = _env_float("BOLTZGEN_PA_RHO1", -2.0)
        rho2_min = _env_float("BOLTZGEN_PA_RHO2_MIN", 0.0)
        rho2_max = _env_float("BOLTZGEN_PA_RHO2_MAX", 1.2)
        rho2_candidates = int(
            _env_float("BOLTZGEN_PA_RHO2_CANDIDATES", 25)
        )
        min_rho_ess_gain = _env_float(
            "BOLTZGEN_PA_RHO_ESS_MIN_GAIN",
            1e-4,
        )
        rho_objective = os.environ.get(
            "BOLTZGEN_PA_RHO_OBJECTIVE", "full_ess"
        ).strip()
        min_rho_variance_gain = _env_float(
            "BOLTZGEN_PA_RHO_VARIANCE_MIN_GAIN",
            0.0,
        )
        no_rho_last = int(
            _env_float("BOLTZGEN_PA_NO_RHO_LAST_STEPS", 2)
        )
        jvp_eps = _env_float("BOLTZGEN_PA_JVP_EPS", 0.01)
        jvp_shrink_alpha = _env_float(
            "BOLTZGEN_PA_JVP_SHRINK_ALPHA",
            0.75,
        )
        gate_logw_rank_clip_topk = int(
            _env_float("BOLTZGEN_PA_GATE_LOGW_RANK_CLIP_TOPK", 0)
        )
        gate_logw_clip = _env_float(
            "BOLTZGEN_PA_GATE_LOGW_CLIP",
            0.0,
        )
        logw_rank_clip_topk = int(
            _env_float("BOLTZGEN_PA_LOGW_RANK_CLIP_TOPK", 0)
        )
        logw_clip = _env_float("BOLTZGEN_PA_LOGW_CLIP", 0.0)
        jvp_logw_rank_clip_topk = int(
            _env_float("BOLTZGEN_PA_JVP_LOGW_RANK_CLIP_TOPK", 0)
        )
        jvp_logw_clip = _env_float(
            "BOLTZGEN_PA_JVP_LOGW_CLIP", 0.0
        )
        kernel_logw_rank_clip_topk = int(
            _env_float("BOLTZGEN_PA_KERNEL_LOGW_RANK_CLIP_TOPK", 0)
        )
        kernel_logw_clip = _env_float(
            "BOLTZGEN_PA_KERNEL_LOGW_CLIP", 0.0
        )
        resample_carry_correction = _env_flag(
            "BOLTZGEN_PA_RESAMPLE_CARRY_CORRECTION",
            "1",
        )
        resample_logw_rank_clip_topk = int(
            _env_float("BOLTZGEN_PA_RESAMPLE_LOGW_RANK_CLIP_TOPK", 0)
        )
        resample_logw_clip = _env_float(
            "BOLTZGEN_PA_RESAMPLE_LOGW_CLIP", 0.0
        )
        max_resamples = int(
            _env_float("BOLTZGEN_PA_MAX_RESAMPLES", 10)
        )
        resample_ess = _env_float("BOLTZGEN_PA_RESAMPLE_ESS", 0.85)
        no_resample_last = int(
            _env_float("BOLTZGEN_PA_NO_RESAMPLE_LAST_STEPS", 10)
        )
        resample_cooldown = int(
            _env_float("BOLTZGEN_PA_RESAMPLE_COOLDOWN_STEPS", 0)
        )
        resample_temper_ess = _env_float(
            "BOLTZGEN_PA_RESAMPLE_TEMPER_ESS",
            0.0,
        )
        min_unique_roots = int(
            _env_float("BOLTZGEN_PA_MIN_UNIQUE_ROOTS", 11)
        )
        max_delta_ratio = _env_float(
            "BOLTZGEN_NEG_GUIDANCE_MAX_DELTA_RATIO",
            1.0,
        )
        variance_floor = _env_float(
            "BOLTZGEN_DNG_VARIANCE_FLOOR",
            1e-4,
        )
        variance_scale = _env_float(
            "BOLTZGEN_DNG_VARIANCE_SCALE",
            1.0,
        )
        network_chunk = max(
            1,
            int(_env_float("BOLTZGEN_PA_NETWORK_CHUNK_PARTICLES", 1)),
        )
        if (
            c <= 0
            or eta < 0
            or gate_power < 0
            or lhat_temp <= 0
            or lhat_clip < 0
            or not math.isfinite(rho1)
            or not math.isfinite(rho2_min)
            or not math.isfinite(rho2_max)
            or rho2_min > rho2_max
            or rho2_candidates < 2
            or rho_objective not in {"full_ess", "log_weight_variance"}
            or min_rho_ess_gain < 0
            or min_rho_variance_gain < 0
            or no_rho_last < 0
            or jvp_eps <= 0
            or not 0 <= jvp_shrink_alpha <= 1
            or gate_logw_rank_clip_topk < 0
            or gate_logw_rank_clip_topk >= particle_count
            or gate_logw_clip < 0
            or logw_rank_clip_topk < 0
            or logw_rank_clip_topk >= particle_count
            or logw_clip < 0
            or jvp_logw_rank_clip_topk < 0
            or jvp_logw_rank_clip_topk >= particle_count
            or jvp_logw_clip < 0
            or kernel_logw_rank_clip_topk < 0
            or kernel_logw_rank_clip_topk >= particle_count
            or kernel_logw_clip < 0
            or resample_logw_rank_clip_topk < 0
            or resample_logw_rank_clip_topk >= particle_count
            or not 0 <= resample_logw_clip <= 10
            or max_resamples < 0
            or max_resamples > 10
            or not 0 < resample_ess <= 1
            or resample_cooldown < 0
            or not 0 <= resample_temper_ess <= 1
        ):
            raise ValueError("Invalid PA-gate SMC hyperparameters.")

        seed_value = os.environ.get("BOLTZGEN_NEG_GUIDANCE_SEED")
        seed = int(seed_value) if seed_value is not None else 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        proposal_generator = torch.Generator(device=self.device).manual_seed(
            seed + 100000
        )
        resample_generator = torch.Generator(device=self.device).manual_seed(
            seed + 200000
        )

        diagnostics_dir_value = os.environ.get(
            "BOLTZGEN_NEG_GUIDANCE_DIAGNOSTICS_DIR",
            "",
        ).strip()
        diagnostics_dir = (
            Path(diagnostics_dir_value).expanduser().resolve()
            if diagnostics_dir_value
            else None
        )
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
        call_index = int(
            getattr(self, "_binder_negative_guidance_call_index", 0)
        )
        self._binder_negative_guidance_call_index = call_index + 1

        num_sampling_steps = default(
            num_sampling_steps,
            self.num_sampling_steps,
        )
        if self.training and self.step_scale_random is not None:
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps,
                device=self.device,
                dtype=torch.float32,
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale function {self.noise_scale_function!r}."
            )
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid sampling schedule {self.sampling_schedule!r}."
            )
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        schedule_rows = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        atom_mask = atom_mask.repeat_interleave(particle_count, 0)
        shape = (*atom_mask.shape, 3)
        atom_coords = sigmas[0] * torch.randn(shape, device=self.device)
        for particle in range(particle_count):
            atom_coords[particle_count + particle, semantic_b] = atom_coords[
                particle,
                semantic_a,
            ]
        initial_a = atom_coords[:particle_count].clone()
        lhat = torch.zeros(
            particle_count,
            device=self.device,
            dtype=torch.float32,
        )
        cumulative_logw = torch.zeros_like(lhat)
        origin_ids = torch.arange(
            particle_count,
            device=self.device,
            dtype=torch.long,
        )
        steps: list[dict[str, Any]] = []
        resample_count = 0
        last_resample_step = -max(1, resample_cooldown)
        rho2_current = 0.0

        def run_denoisers(noisy: torch.Tensor, t_hat: float) -> torch.Tensor:
            result = torch.empty_like(noisy)
            for start in range(0, particle_count, network_chunk):
                end = min(start + network_chunk, particle_count)
                count = end - start
                chunk = torch.cat(
                    [
                        noisy[start:end],
                        noisy[
                            particle_count + start : particle_count + end
                        ],
                    ],
                    dim=0,
                )
                denoised_chunk, _ = self.preconditioned_network_forward(
                    chunk,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=count,
                        **network_condition_kwargs,
                    ),
                )
                result[start:end] = denoised_chunk[:count]
                result[
                    particle_count + start : particle_count + end
                ] = denoised_chunk[count:]
            return result

        def solver_next(
            noisy: torch.Tensor,
            endpoint: torch.Tensor,
            mask: torch.Tensor,
            t_hat: float,
            branch_step: float,
        ) -> torch.Tensor:
            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    origin = weighted_rigid_align(
                        noisy.float(),
                        endpoint.float(),
                        mask.float(),
                        mask.float(),
                    ).to(endpoint)
            else:
                origin = noisy
            return origin + branch_step * (origin - endpoint) / t_hat

        def guided_endpoints(
            denoised_a: torch.Tensor,
            noisy_a: torch.Tensor,
            noisy_b: torch.Tensor,
            denoised_b: torch.Tensor,
            rotations: list[torch.Tensor],
            scales: torch.Tensor,
        ) -> tuple[torch.Tensor, list[float]]:
            endpoints = denoised_a.clone()
            effective: list[float] = []
            for particle in range(particle_count):
                endpoint, used, _, _ = _compose_binder_denoised(
                    positive_denoised=denoised_a[particle],
                    negative_denoised=denoised_b[particle],
                    positive_noisy=noisy_a[particle],
                    negative_noisy=noisy_b[particle],
                    t_hat=t_hat_value,
                    positive_binder_indices=binder_a,
                    negative_binder_indices=binder_b,
                    negative_to_positive_rotation=rotations[particle],
                    guidance_scale=float(scales[particle]),
                    # Proposition 2 requires a linear score field. Stability
                    # is supplied by term-wise log-weight clipping instead.
                    max_delta_ratio=0.0,
                )
                endpoints[particle] = endpoint
                effective.append(used)
            return endpoints, effective

        print(
            "[BINDER_PA_GATE_SMC] "
            f"particles={particle_count} chunk={network_chunk} "
            f"prop2=1 rho1={rho1:g} "
            f"rho2=[{rho2_min:g},{rho2_max:g}] "
            f"rho2_candidates={rho2_candidates} "
            f"rho_objective={rho_objective} "
            f"c={c:g} eta={eta:g} "
            f"gate_power={gate_power:g} lhat_temp={lhat_temp:g} "
            f"jvp_eps={jvp_eps:g} jvp_alpha={jvp_shrink_alpha:g} "
            "ess_mode=full_cumulative "
            f"gate_clip_topk={gate_logw_rank_clip_topk} "
            f"gate_clip={gate_logw_clip:g} "
            f"logw_clip_topk={logw_rank_clip_topk} "
            f"logw_clip={logw_clip:g} "
            f"resample_ess={resample_ess:g} seed={seed}",
            flush=True,
        )

        for step_index, (
            sigma_tm,
            sigma_t,
            gamma,
            step_scale_value,
            noise_scale_value,
        ) in optionally_tqdm(
            enumerate(schedule_rows),
            use_tqdm=inference_logging,
            desc="Binder PA-gate SMC denoising.",
        ):
            sigma_tm_value = float(sigma_tm)
            sigma_t_value = float(sigma_t)
            gamma_value = float(gamma)
            t_hat_value = sigma_tm_value * (1 + gamma_value)
            noise_variance = float(noise_scale_value) ** 2 * (
                t_hat_value**2 - sigma_tm_value**2
            )
            atom_coords = center(atom_coords, atom_mask)
            state_rotations: list[float] = []
            for particle in range(particle_count):
                b_index = particle_count + particle
                rotation, translation, _ = _row_kabsch(
                    atom_coords[b_index, hla_b],
                    atom_coords[particle, hla_a],
                )
                atom_coords[b_index] = (
                    atom_coords[b_index] @ rotation + translation
                )
                atom_coords[b_index, semantic_b] = atom_coords[
                    particle,
                    semantic_a,
                ]

            if self.coordinate_augmentation_inference:
                random_rotations, random_translations = (
                    compute_random_augmentation(
                        particle_count,
                        device=atom_coords.device,
                        dtype=atom_coords.dtype,
                    )
                )
                for particle in range(particle_count):
                    for branch_index in (
                        particle,
                        particle_count + particle,
                    ):
                        atom_coords[branch_index] = (
                            atom_coords[branch_index]
                            @ random_rotations[particle]
                            + random_translations[particle]
                        )

            added_noise = (
                float(noise_scale_value)
                * sqrt(noise_variance)
                * torch.randn(shape, device=self.device)
            )
            for particle in range(particle_count):
                added_noise[
                    particle_count + particle,
                    semantic_b,
                ] = added_noise[particle, semantic_a]
            noisy = atom_coords + added_noise
            with torch.no_grad():
                denoised = run_denoisers(noisy, t_hat_value)
            denoised_a = denoised[:particle_count]
            denoised_b = denoised[particle_count:]
            noisy_a = noisy[:particle_count]
            noisy_b = noisy[particle_count:]
            rotations: list[torch.Tensor] = []
            rotation_degrees: list[float] = []
            for particle in range(particle_count):
                rotation, _, _ = _row_kabsch(
                    denoised_b[particle, hla_b],
                    denoised_a[particle, hla_a],
                )
                rotations.append(rotation)
                rotation_degrees.append(_rotation_angle_degrees(rotation))

            branch_step = float(step_scale_value) * (
                sigma_t_value - t_hat_value
            )
            raw_mean_a = solver_next(
                noisy_a,
                denoised_a,
                atom_mask[:particle_count],
                t_hat_value,
                branch_step,
            )
            raw_mean_b = solver_next(
                noisy_b,
                denoised_b,
                atom_mask[particle_count:],
                t_hat_value,
                branch_step,
            )
            theta, target_scale = _pa_gate_theta_and_scale(
                lhat,
                c=c,
                eta=eta,
                gate_power=gate_power,
                lhat_temperature=lhat_temp,
            )
            transition_variance = max(
                variance_floor,
                branch_step**2 * variance_scale,
            )

            # A one-unit guidance displacement defines d in the same local
            # Gaussian units as hs=transition_variance.  This avoids assuming
            # a DDPM parameterization for BoltzGen's EDM velocity output.
            unit_endpoints, _ = guided_endpoints(
                denoised_a,
                noisy_a,
                noisy_b,
                denoised_b,
                rotations,
                torch.ones_like(target_scale),
            )
            unit_mean = solver_next(
                noisy_a,
                unit_endpoints,
                atom_mask[:particle_count],
                t_hat_value,
                branch_step,
            )
            d_y_full = (
                unit_mean.float() - raw_mean_a.float()
            ) / float(transition_variance)
            g_y_full = target_scale.float()[:, None, None] * d_y_full
            proposal_noise = (
                math.sqrt(transition_variance)
                * torch.randn(
                    raw_mean_a.shape,
                    device=self.device,
                    dtype=torch.float32,
                    generator=proposal_generator,
                )
                * atom_mask[:particle_count, :, None].float()
            )
            delta0_full = (
                raw_mean_a.float()
                - noisy_a.float()
                + float(transition_variance)
                * (g_y_full - float(rho1) * d_y_full)
                + proposal_noise
            )

            # Fixed-displacement finite-difference JVP: all rho2 candidates
            # share Jd[Delta0], exactly as in the Prop-2 surrogate.
            perturbed_a = noisy_a.float() + float(jvp_eps) * delta0_full
            perturbed_b = noisy_b.float().clone()
            perturbed_b[:, semantic_b] = perturbed_a[:, semantic_a]
            perturbed = torch.cat([perturbed_a, perturbed_b], dim=0).to(noisy)
            with torch.no_grad():
                denoised_perturbed = run_denoisers(
                    perturbed,
                    t_hat_value,
                )
            denoised_perturbed_a = denoised_perturbed[:particle_count]
            denoised_perturbed_b = denoised_perturbed[particle_count:]
            perturbed_rotations: list[torch.Tensor] = []
            for particle in range(particle_count):
                rotation, _, _ = _row_kabsch(
                    denoised_perturbed_b[particle, hla_b],
                    denoised_perturbed_a[particle, hla_a],
                )
                perturbed_rotations.append(rotation)
            raw_mean_perturbed = solver_next(
                perturbed_a.to(noisy_a),
                denoised_perturbed_a,
                atom_mask[:particle_count],
                t_hat_value,
                branch_step,
            )
            unit_endpoints_perturbed, _ = guided_endpoints(
                denoised_perturbed_a,
                perturbed_a.to(noisy_a),
                perturbed_b.to(noisy_b),
                denoised_perturbed_b,
                perturbed_rotations,
                torch.ones_like(target_scale),
            )
            unit_mean_perturbed = solver_next(
                perturbed_a.to(noisy_a),
                unit_endpoints_perturbed,
                atom_mask[:particle_count],
                t_hat_value,
                branch_step,
            )
            d_perturbed_full = (
                unit_mean_perturbed.float() - raw_mean_perturbed.float()
            ) / float(transition_variance)
            jd_delta0_full = (
                d_perturbed_full - d_y_full
            ) / float(jvp_eps)

            # The gate's scalar time coefficient is estimated from the
            # reference A/B local Gaussian likelihood ratio at y.  The B mean
            # is fitted once into A's frame; only binder coordinates enter.
            alpha_l_rows = []
            for particle in range(particle_count):
                fit_b_r, fit_b_t, _ = _row_kabsch(
                    raw_mean_b[particle, hla_b],
                    raw_mean_a[particle, hla_a],
                )
                mean_b_in_a = raw_mean_b[particle] @ fit_b_r + fit_b_t
                y_binder = noisy_a[particle, binder_a].float()
                distance_a = torch.sum(
                    torch.square(
                        y_binder
                        - raw_mean_a[particle, binder_a].float()
                    )
                )
                distance_b = torch.sum(
                    torch.square(
                        y_binder - mean_b_in_a[binder_b].float()
                    )
                )
                alpha_l_rows.append(
                    (distance_a - distance_b)
                    / (2.0 * float(transition_variance))
                )
            alpha_l = torch.stack(alpha_l_rows)
            gate_time = (
                -float(gate_power)
                * theta.float()
                * float(eta)
                * float(lhat_temp)
                * alpha_l
            )

            rho2_grid = torch.linspace(
                rho2_min,
                rho2_max,
                rho2_candidates,
                device=self.device,
                dtype=torch.float32,
            )
            if not bool(
                torch.isclose(rho2_grid, torch.zeros_like(rho2_grid)).any()
            ):
                rho2_grid = torch.cat(
                    [rho2_grid, torch.zeros(1, device=self.device)]
                ).sort().values
            prop2_terms = _pa_prop2_candidate_terms(
                rho1=rho1,
                rho2=rho2_grid,
                gate_time=gate_time,
                theta=theta.float(),
                gate_power=gate_power,
                eta=eta,
                lhat_temperature=lhat_temp,
                variance=transition_variance,
                nu_a=(
                    noisy_a[:, binder_a].float()
                    - raw_mean_a[:, binder_a].float()
                ),
                d_y=d_y_full[:, binder_a],
                g_y=g_y_full[:, binder_a],
                zeta=target_scale.float(),
                delta0=delta0_full[:, binder_a],
                jd_delta0=jd_delta0_full[:, binder_a],
                jvp_shrink_alpha=jvp_shrink_alpha,
            )
            candidate_logw, robust_terms = (
                _pa_prop2_robustify_candidate_terms(
                    prop2_terms,
                    gate_topk=gate_logw_rank_clip_topk,
                    gate_max_abs=gate_logw_clip,
                    jvp_topk=jvp_logw_rank_clip_topk,
                    jvp_max_abs=jvp_logw_clip,
                    kernel_topk=kernel_logw_rank_clip_topk,
                    kernel_max_abs=kernel_logw_clip,
                )
            )
            candidate_logw, total_clip_mask, total_clip_thresholds = (
                _winsorize_centered_log_weight_rows(
                    candidate_logw,
                    topk=logw_rank_clip_topk,
                    max_abs=logw_clip,
                )
            )
            incremental_ess_candidates = [
                _ess_fraction_from_log_weights(row)
                for row in candidate_logw
            ]
            total_ess_candidates = [
                _ess_fraction_from_log_weights(cumulative_logw + row)
                for row in candidate_logw
            ]
            total_logw_variance_candidates = [
                float(
                    torch.var(
                        cumulative_logw.float() + row.float(),
                        unbiased=False,
                    ).item()
                )
                for row in candidate_logw
            ]
            zero_index = int(torch.argmin(torch.abs(rho2_grid)).item())
            if step_index >= len(schedule_rows) - max(0, no_rho_last):
                selected_index = int(
                    torch.argmin(
                        torch.abs(rho2_grid - float(rho2_current))
                    ).item()
                )
            else:
                selected_index = _select_pa_rho_candidate(
                    objective=rho_objective,
                    total_ess=total_ess_candidates,
                    total_logw_variance=total_logw_variance_candidates,
                    zero_index=zero_index,
                    min_ess_gain=min_rho_ess_gain,
                    min_variance_gain=min_rho_variance_gain,
                )
            rho2_current = float(rho2_grid[selected_index].item())
            selected_rho = rho2_current
            incremental_ess = incremental_ess_candidates[selected_index]
            cumulative_ess = total_ess_candidates[selected_index]
            objective_ess = cumulative_ess
            logw_increment = candidate_logw[selected_index]
            proposal_scale = (
                target_scale.float()
                - float(rho1)
                - float(rho2_current) * target_scale.float()
            )
            effective_scales = [float(value) for value in proposal_scale]
            next_a = (
                raw_mean_a.float()
                + float(transition_variance)
                * (
                    g_y_full
                    - float(rho1) * d_y_full
                    - float(rho2_current) * g_y_full
                )
                + proposal_noise
            ).to(raw_mean_a)

            # Advance the likelihood-ratio state with the selected realized
            # proposal.  This state update is separate from the Prop-2 weight,
            # which remains the retained local RN expansion above.
            kernel_ratios: list[torch.Tensor] = []
            for particle in range(particle_count):
                fit_a_r, fit_a_t, _ = _row_kabsch(
                    raw_mean_a[particle, hla_a],
                    next_a[particle, hla_a],
                )
                fit_b_r, fit_b_t, _ = _row_kabsch(
                    raw_mean_b[particle, hla_b],
                    next_a[particle, hla_a],
                )
                mean_a_in_sample = raw_mean_a[particle] @ fit_a_r + fit_a_t
                mean_b_in_sample = raw_mean_b[particle] @ fit_b_r + fit_b_t
                sample_binder = next_a[particle, binder_a].float()
                distance_a = torch.sum(
                    torch.square(
                        sample_binder
                        - mean_a_in_sample[binder_a].float()
                    )
                )
                distance_b = torch.sum(
                    torch.square(
                        sample_binder
                        - mean_b_in_sample[binder_b].float()
                    )
                )
                kernel_ratios.append(
                    -0.5
                    * (distance_b - distance_a)
                    / float(transition_variance)
                )
            kernel_ratio = torch.stack(kernel_ratios)
            lhat_new = lhat.float() + float(lhat_temp) * kernel_ratio
            if lhat_clip > 0:
                lhat_new = torch.clamp(
                    lhat_new,
                    min=-lhat_clip,
                    max=lhat_clip,
                )
            gate_clip_fraction = float(
                robust_terms["gate_mask"][selected_index].float().mean()
            )
            gate_clip_threshold = float(
                robust_terms["gate_threshold"][selected_index]
            )
            jvp_clip_fraction = float(
                robust_terms["jvp_mask"][selected_index].float().mean()
            )
            jvp_clip_threshold = float(
                robust_terms["jvp_threshold"][selected_index]
            )
            kernel_clip_fraction = float(
                robust_terms["kernel_mask"][selected_index].float().mean()
            )
            kernel_clip_threshold = float(
                robust_terms["kernel_threshold"][selected_index]
            )
            logw_clip_fraction = float(
                total_clip_mask[selected_index].float().mean()
            )
            logw_clip_threshold = float(
                total_clip_thresholds[selected_index]
            )
            cumulative_logw = cumulative_logw + logw_increment

            next_b = raw_mean_b.clone()
            for particle in range(particle_count):
                rotation, translation, _ = _row_kabsch(
                    next_b[particle, hla_b],
                    next_a[particle, hla_a],
                )
                next_b[particle] = (
                    next_b[particle] @ rotation + translation
                )
                next_b[particle, semantic_b] = next_a[
                    particle,
                    semantic_a,
                ]
            atom_coords_next = torch.cat([next_a, next_b], dim=0)

            allow_resample = (
                step_index
                < len(schedule_rows) - max(0, no_resample_last)
            )
            cooldown_active = (
                resample_cooldown > 0
                and step_index - last_resample_step < resample_cooldown
            )
            do_resample = (
                allow_resample
                and not cooldown_active
                and resample_count < max_resamples
                and objective_ess < resample_ess
            )
            resample_temper_alpha = 1.0
            resample_weight_ess = objective_ess
            ancestor_clip_fraction = 0.0
            ancestor_clip_threshold = float("inf")
            if do_resample:
                full_resample_logw = cumulative_logw
                (
                    resample_logw,
                    ancestor_clip_mask,
                    ancestor_clip_threshold,
                ) = _winsorize_centered_log_weights(
                    full_resample_logw,
                    topk=resample_logw_rank_clip_topk,
                    max_abs=resample_logw_clip,
                )
                ancestor_clip_fraction = float(
                    ancestor_clip_mask.float().mean().item()
                )
                resample_weight_ess = _ess_fraction_from_log_weights(
                    resample_logw
                )
                if (
                    resample_temper_ess > 0
                    and _ess_fraction_from_log_weights(resample_logw)
                    < resample_temper_ess
                ):
                    (
                        resample_logw,
                        resample_temper_alpha,
                    ) = _temper_log_weights_to_ess(
                        resample_logw,
                        resample_temper_ess,
                    )
                    resample_weight_ess = _ess_fraction_from_log_weights(
                        resample_logw
                    )
                indices = _systematic_resample_indices(
                    resample_logw,
                    generator=resample_generator,
                )
                next_a = next_a[indices]
                next_b = next_b[indices]
                lhat_new = lhat_new[indices]
                origin_ids = origin_ids[indices]
                if resample_carry_correction:
                    # Clipping and tempering alter only ancestor selection.
                    # Carry log(W/P) so the full Prop-2 target weight is
                    # retained after resampling, up to a common constant.
                    cumulative_logw = (
                        full_resample_logw - resample_logw
                    )[indices]
                else:
                    cumulative_logw = torch.zeros_like(cumulative_logw)
                atom_coords_next = torch.cat([next_a, next_b], dim=0)
                resample_count += 1
                last_resample_step = step_index
                unique_roots = int(torch.unique(origin_ids).numel())
                if min_unique_roots > 0 and unique_roots < min_unique_roots:
                    if diagnostics_dir is not None:
                        _write_trace_json(
                            diagnostics_dir
                            / f"sample_{call_index:03d}_early_stop.json",
                            {
                                "format": (
                                    "boltzgen_pa_gate_smc_edm_early_stop"
                                ),
                                "format_version": 1,
                                "guidance_mode": "pa_gate_smc",
                                "status": (
                                    "early_stopped_infeasible_roots"
                                ),
                                "seed": seed,
                                "particle_count": particle_count,
                                "early_stop_step": step_index,
                                "observed_unique_initial_roots": (
                                    unique_roots
                                ),
                                "minimum_required_unique_roots": (
                                    min_unique_roots
                                ),
                                "resample_count": resample_count,
                                "final_origin_ids": _trace_array(
                                    origin_ids
                                ).tolist(),
                                "last_incremental_ess_fraction": (
                                    incremental_ess
                                ),
                                "last_resample_weight_ess_fraction": (
                                    resample_weight_ess
                                ),
                                "last_resample_temper_alpha": (
                                    resample_temper_alpha
                                ),
                                "last_selected_rho1": rho1,
                                "last_selected_rho2": selected_rho,
                                "last_target_scale_mean": float(
                                    target_scale.mean()
                                ),
                                "last_proposal_scale_mean": float(
                                    proposal_scale.mean()
                                ),
                                "completed_steps": steps,
                            },
                        )
                    raise RuntimeError(
                        "PA-gate SMC genealogy collapsed below the requested "
                        f"{min_unique_roots} roots at step {step_index}: "
                        f"{unique_roots} remain."
                    )
            unique_roots = int(torch.unique(origin_ids).numel())
            steps.append(
                {
                    "step": step_index,
                    "sigma": t_hat_value,
                    "requested_guidance_scale": float(target_scale.mean()),
                    "effective_guidance_scale": float(
                        statistics.fmean(effective_scales)
                    ),
                    "target_scale_min": float(target_scale.min()),
                    "target_scale_mean": float(target_scale.mean()),
                    "target_scale_max": float(target_scale.max()),
                    "proposal_scale_mean": float(proposal_scale.mean()),
                    "rho1": rho1,
                    "rho2": selected_rho,
                    "rho2_selection_objective": rho_objective,
                    "rho2_zero_total_ess_fraction": (
                        total_ess_candidates[zero_index]
                    ),
                    "rho2_candidate_total_ess_min": min(
                        total_ess_candidates
                    ),
                    "rho2_candidate_total_ess_max": max(
                        total_ess_candidates
                    ),
                    "rho2_candidate_total_ess_range": (
                        max(total_ess_candidates)
                        - min(total_ess_candidates)
                    ),
                    "rho2_zero_total_logw_variance": (
                        total_logw_variance_candidates[zero_index]
                    ),
                    "rho2_candidate_total_logw_variance_min": min(
                        total_logw_variance_candidates
                    ),
                    "rho2_candidate_total_logw_variance_max": max(
                        total_logw_variance_candidates
                    ),
                    "rho2_candidate_total_logw_variance_range": (
                        max(total_logw_variance_candidates)
                        - min(total_logw_variance_candidates)
                    ),
                    "incremental_ess_fraction": incremental_ess,
                    "cumulative_ess_fraction": cumulative_ess,
                    "ess_objective_fraction": objective_ess,
                    "ess_objective": "full_cumulative_after_clip",
                    "gate_logw_clip_fraction": gate_clip_fraction,
                    "gate_logw_clip_threshold": gate_clip_threshold,
                    "jvp_logw_clip_fraction": jvp_clip_fraction,
                    "jvp_logw_clip_threshold": jvp_clip_threshold,
                    "kernel_logw_clip_fraction": kernel_clip_fraction,
                    "kernel_logw_clip_threshold": kernel_clip_threshold,
                    "logw_clip_fraction": logw_clip_fraction,
                    "logw_clip_threshold": logw_clip_threshold,
                    "ancestor_logw_clip_fraction": ancestor_clip_fraction,
                    "ancestor_logw_clip_threshold": ancestor_clip_threshold,
                    "resample_weight_ess_fraction": resample_weight_ess,
                    "resample_temper_alpha": resample_temper_alpha,
                    "resample_cooldown_active": cooldown_active,
                    "resampled": do_resample,
                    "unique_initial_roots": unique_roots,
                    "theta_mean": float(theta.mean()),
                    "lhat_mean_before": float(lhat.mean()),
                    "lhat_mean_after": float(lhat_new.mean()),
                    "rotation_angle_degrees": float(
                        statistics.median(rotation_degrees)
                    ),
                    "rotation_angle_median_degrees": float(
                        statistics.median(rotation_degrees)
                    ),
                    "rotation_angle_max_degrees": max(rotation_degrees),
                }
            )
            atom_coords = atom_coords_next
            lhat = lhat_new

            if step_index % 25 == 0 or step_index == len(schedule_rows) - 1:
                print(
                    "[BINDER_PA_GATE_SMC_STEP] "
                    f"step={step_index}/{len(schedule_rows) - 1} "
                    f"sigma={t_hat_value:.5g} "
                            f"target={float(target_scale.mean()):.4g} "
                            f"proposal={steps[-1]['proposal_scale_mean']:.4g} "
                            f"rho1={rho1:+.3g} rho2={selected_rho:+.3g} "
                            f"incESS={incremental_ess:.3f} "
                            f"cumESS={cumulative_ess:.3f} "
                            f"resample={int(do_resample)} roots={unique_roots}",
                    flush=True,
                )

        final_a = atom_coords[:particle_count]
        if diagnostics_dir is not None:
            diagnostics_path = (
                diagnostics_dir / f"sample_{call_index:03d}.json"
            )
            _write_trace_json(
                diagnostics_path,
                {
                    "format": "boltzgen_pa_gate_smc_prop2_edm_diagnostics",
                    "format_version": 2,
                    "guidance_mode": "pa_gate_smc",
                    "seed": seed,
                    "particle_count": particle_count,
                    "input_HLA_raw_rmsd_angstrom": float(
                        conditioning_raw_rmsd
                    ),
                    "input_HLA_fit_rmsd_angstrom": float(
                        conditioning_fit_rmsd
                    ),
                    "pa_gate": {
                        "c": c,
                        "eta": eta,
                        "gate_power": gate_power,
                        "lhat_temperature": lhat_temp,
                        "lhat_clip": lhat_clip,
                        "proposal_family": "u=rho1*d+rho2*g",
                        "weight_mode": "proposition_2_retained_expansion",
                        "rho1": rho1,
                        "rho2_min": rho2_min,
                        "rho2_max": rho2_max,
                        "rho2_candidates": rho2_candidates,
                        "rho2_selection_objective": rho_objective,
                        "rho2_variance_min_gain": (
                            min_rho_variance_gain
                        ),
                        "no_rho_last_steps": no_rho_last,
                        "jvp_mode": "fixed_displacement_finite_difference",
                        "jvp_eps": jvp_eps,
                        "jvp_shrink_alpha": jvp_shrink_alpha,
                        "ess_objective": "full_cumulative_after_clip",
                        "gate_logw_rank_clip_topk": (
                            gate_logw_rank_clip_topk
                        ),
                        "gate_logw_clip": gate_logw_clip,
                        "jvp_logw_rank_clip_topk": (
                            jvp_logw_rank_clip_topk
                        ),
                        "jvp_logw_clip": jvp_logw_clip,
                        "kernel_logw_rank_clip_topk": (
                            kernel_logw_rank_clip_topk
                        ),
                        "kernel_logw_clip": kernel_logw_clip,
                        "logw_rank_clip_topk": logw_rank_clip_topk,
                        "logw_clip": logw_clip,
                        "resample_logw_rank_clip_topk": (
                            resample_logw_rank_clip_topk
                        ),
                        "resample_logw_clip": resample_logw_clip,
                        "max_resamples": max_resamples,
                        "resample_carry_correction": (
                            resample_carry_correction
                        ),
                        "resample_ess": resample_ess,
                        "resample_temper_ess": resample_temper_ess,
                        "resample_cooldown_steps": resample_cooldown,
                        "no_resample_last_steps": no_resample_last,
                        "min_unique_roots": min_unique_roots,
                    },
                    "resample_count": resample_count,
                    "final_unique_initial_roots": int(
                        torch.unique(origin_ids).numel()
                    ),
                    "final_origin_ids": _trace_array(origin_ids).tolist(),
                    "edm_transfer": {
                        "retained_prop2_components": [
                            "pure positive-A nonlinear gate",
                            "two-field u=rho1*d+rho2*g forward correction",
                            "matched backward mean_A+hs*(g-u)",
                            "h*alpha scalar term",
                            "target and proposal curvature terms",
                            "fixed-displacement Jd[Delta0] term",
                            "global interacting-particle ESS",
                            "systematic resampling",
                        ],
                        "approximated_components": [
                            "EDM sigma-step local Gaussian variance",
                            "finite-difference rather than forward-AD JVP",
                            "finite rho2 grid",
                            "optional term-wise finite-step winsorization",
                        ],
                        "edm_specialization": [
                            "variance-exploding reference forward drift is zero",
                            "d is calibrated from one-unit EDM guidance displacement",
                        ],
                    },
                    "steps": steps,
                },
            )
            print(
                f"[BINDER_PA_GATE_SMC_DIAGNOSTICS] {diagnostics_path}",
                flush=True,
            )
        return {
            "sample_atom_coords": final_a,
            "coords_traj": [initial_a, final_a],
            "x0_coords_traj": [final_a],
        }

    # training
    def loss_weight(self, sigma):
        # note: in AF3 there is a + at denominator while in EDM a *, we think this is a mistake in the paper
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        # note: in AF3 the sample is scaled by sigma_data while in EDM it is not
        # in practice this just means scaling P_mean by the log

        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity
        atom_coords = feats["coords"]
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)

        padded_sigmas = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise
        # alphas=1. in paper

        denoised_atom_coords, net_out = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        out_dict = {
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }
        out_dict.update(net_out)

        return out_dict

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        add_bond_loss=False,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        fake_atom_weight=1.0,
        residue_type_weight=0.0,
        multiplicity=1,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            noised_atom_coords = out_dict["noised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            # fake atom weighting
            fake_atom_mask = feats["fake_atom_mask"]
            fake_atom_weight = (1 - fake_atom_mask) + fake_atom_mask * fake_atom_weight

            # residue type weighting.
            if residue_type_weight > 0.0:
                design_atom_mask = torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["design_mask"].float().unsqueeze(-1),
                ).squeeze(-1)
                _res_type_weight = torch.tensor(
                    const.res_type_weight, device=denoised_atom_coords.device
                )
                _res_type_weight = torch.bmm(
                    feats["atom_to_token"].float(),
                    (feats["res_type"].float() @ _res_type_weight)
                    .unsqueeze(-1)
                    .float(),
                ).squeeze(-1)
                res_type_weight = (
                    1.0 - design_atom_mask
                ) + design_atom_mask * _res_type_weight
                res_type_weight = res_type_weight**residue_type_weight
            else:
                res_type_weight = 1.0

            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            if self.mse_rotational_alignment:
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach(),
                    denoised_atom_coords.detach(),
                    align_weights.detach(),
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0)
                    .detach(),
                )
            else:
                atom_coords_aligned_ground_truth = weighted_rigid_centering(
                    atom_coords,
                    denoised_atom_coords,
                    align_weights,
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0),
                )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss
                * align_weights
                * fake_atom_weight
                * res_type_weight
                * resolved_atom_mask,
                dim=-1,
            ) / (
                torch.sum(
                    3
                    * align_weights
                    * fake_atom_weight
                    * res_type_weight
                    * resolved_atom_mask,
                    dim=-1,
                )
                + 1e-5
            )
            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            if add_bond_loss:
                bond_loss, num_bonds = compute_bond_loss(
                    pred_atom_coords=out_dict["denoised_atom_coords"].float(),
                    true_coords=atom_coords_aligned_ground_truth,
                    feats=feats,
                )
                total_loss += bond_loss
            else:
                bond_loss = self.zero

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "bond_loss": bond_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
