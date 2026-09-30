#!/usr/bin/env python3
"""Monitor SMC negative guidance on one Stable Diffusion DNG prompt.

This is the text-to-image analogue of the project's CIFAR SMC monitor.
The important difference is the incremental weight: by default this script
uses the finite-step Gaussian SMC weight

    log w =
        log(1 - theta_new) - log(1 - theta_old)
        + log N(x_new; mu_base, sigma_t^2 I)
        - log N(x_new; mu_guided, sigma_t^2 I),

where the means and variance are computed from the active Diffusers
DDPMScheduler.  It also records the local C*h approximation and the quadratic
approximation term, so the exact-style weight can be compared to the monitor
used in the DDPM multiclass experiments.

Default task:
    prompt 1, "Medieval feast", related negative prompt "Chalices, candles".

Example:
    python run_smc_gaussian_monitor.py --device cuda:0 --steps 20 --n-particles 32
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_np_grid_eval import HF_CACHE, SD_MODEL_ID  # noqa: E402
from run_dng_eval import (  # noqa: E402
    DNG_HYPERPARAMS_BY_NAME,
    PROMPTS_FILE,
    ddpm_reverse_mean_and_variance,
)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    return payload["prompts"] if isinstance(payload, dict) and "prompts" in payload else payload


def load_sd_pipeline(device: str, model_id: str):
    from diffusers import DDPMScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_text(pipe, texts: list[str], device: str):
    embeds, _ = pipe.encode_prompt(
        texts,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return embeds


def log1m_theta_from_lhat(lhat: torch.Tensor, c: float, bs: float) -> torch.Tensor:
    z = math.log(max(float(c), 1e-30)) + float(bs) * lhat
    return -torch.logaddexp(torch.zeros_like(z), z.clamp(-80.0, 80.0))


def theta_from_lhat(lhat: torch.Tensor, c: float, bs: float, theta_eps: float) -> torch.Tensor:
    z = math.log(max(float(c), 1e-30)) + float(bs) * lhat
    theta = torch.sigmoid(z.clamp(-80.0, 80.0))
    if theta_eps > 0.0:
        theta = theta.clamp(min=float(theta_eps), max=1.0 - float(theta_eps))
    return theta


def gaussian_log_prob_isotropic(
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    x_flat = x.float().flatten(1)
    mean_flat = mean.float().flatten(1)
    var = variance.to(device=x.device, dtype=torch.float32).clamp_min(1e-20)
    d = x_flat.shape[1]
    dist = (x_flat - mean_flat).pow(2).sum(dim=1)
    return -0.5 * d * torch.log(2.0 * torch.pi * var) - 0.5 * dist / var


def model_output_to_epsilon(scheduler, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
    t = int(timestep)
    alpha_prod_t = scheduler.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
    beta_prod_t = 1.0 - alpha_prod_t
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return model_output
    if prediction_type == "v_prediction":
        return alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    if prediction_type == "sample":
        return (sample - alpha_prod_t.sqrt() * model_output) / beta_prod_t.sqrt().clamp_min(1e-12)
    raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")


def schedule_interp(progress: float, low_noise: float, high_noise: float, mode: str, gamma: float) -> float:
    """Interpolate from high_noise at progress=0 to low_noise at progress=1."""
    progress = min(1.0, max(0.0, float(progress)))
    if mode == "constant":
        return float(high_noise)
    if mode == "linear" or gamma < 1e-8:
        return float(high_noise + (low_noise - high_noise) * progress)
    if mode == "log":
        weight = math.log(1.0 + gamma * progress) / math.log(1.0 + gamma)
        return float(high_noise + (low_noise - high_noise) * weight)
    if mode == "exp":
        base = 1.0 + gamma
        weight = (base**progress - 1.0) / gamma
        return float(high_noise + (low_noise - high_noise) * weight)
    raise ValueError(f"Unknown schedule mode {mode!r}")


def schedule_at_state(k: int, n_steps: int, args: argparse.Namespace) -> tuple[float, float]:
    # Match Linrui's _smc_softening_schedules convention:
    # progress/s = 0 at the high-noise side and 1 at the data side.
    progress = float(k) / max(int(n_steps) - 1, 1)
    c = schedule_interp(progress, args.c0, args.c1, args.c_schedule, args.gamma_c)
    bs = schedule_interp(progress, args.bs0, args.bs1, args.bs_schedule, args.gamma_bs)
    return c, bs


def normalized_weights_from_logw(logw: torch.Tensor) -> torch.Tensor:
    lw = logw - torch.logsumexp(logw, dim=0)
    return torch.exp(lw)


def ess_frac_from_logw(logw: torch.Tensor) -> float:
    w = normalized_weights_from_logw(logw)
    return float(1.0 / (w.shape[0] * torch.sum(w * w)).item())


def systematic_resample(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    n = w.shape[0]
    u0 = torch.rand((), generator=gen, device=w.device, dtype=torch.float32)
    positions = (u0 + torch.arange(n, device=w.device, dtype=torch.float32)) / n
    cdf = torch.cumsum(w.float(), dim=0)
    return torch.searchsorted(cdf.contiguous(), positions.contiguous()).clamp(0, n - 1)


def finite_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float()
    y = y.detach().float()
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).mean() * (y * y).mean()).clamp_min(1e-20)
    return float(((x * y).mean() / denom).item())


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"mean_{prefix}": float(x.mean().item()),
        f"std_{prefix}": float(x.std(unbiased=False).item()),
        f"min_{prefix}": float(x.min().item()),
        f"max_{prefix}": float(x.max().item()),
    }


def make_image_grid(images, cols: int = 8):
    from PIL import Image

    if not images:
        return None
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    return grid


def plot_monitor(rows: list[dict[str, Any]], out_dir: Path, resample_ess: float) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows_sorted])

    fig, axes = plt.subplots(5, 2, figsize=(15, 18), dpi=140)

    ax = axes[0, 0]
    ax.plot(step, [r["guidance_mean"] for r in rows_sorted], label="mean theta*bs")
    ax.plot(step, [r["guidance_max"] for r in rows_sorted], ls="--", alpha=0.7, label="max theta*bs")
    ax.set_title("Effective Guidance")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("theta * bs")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(step, [r["mean_theta"] for r in rows_sorted], label="mean theta")
    ax.plot(step, [r["min_theta"] for r in rows_sorted], ls="--", label="min theta")
    ax.plot(step, [r["max_theta"] for r in rows_sorted], ls="--", label="max theta")
    ax.set_title("Theta Statistics")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(step, [r["inc_ess_frac_used"] for r in rows_sorted], label="incremental")
    ax.plot(step, [r["cum_ess_frac_used"] for r in rows_sorted], label="cumulative")
    if resample_ess <= 1.0:
        ax.axhline(resample_ess, color="tab:red", ls=":", label="resample threshold")
    rs = [r for r in rows_sorted if int(r["resampled"]) == 1]
    if rs:
        ax.scatter([r["step"] for r in rs], [r["cum_ess_frac_used"] for r in rs], s=20, color="black", label="resampled")
    ax.set_title("ESS/N From Used Weights")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("ESS/N")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step, [r["std_logw_gaussian"] for r in rows_sorted], label="gaussian")
    ax.plot(step, [r["std_logw_local"] for r in rows_sorted], label="local C")
    ax.plot(step, [r["std_logw_quad"] for r in rows_sorted], label="local + quad")
    ax.set_title("Per-Step Log-Weight Spread")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("std(log w)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(step, [r["mean_exact_minus_local"] for r in rows_sorted], label="mean")
    ax.fill_between(
        step,
        [r["mean_exact_minus_local"] - r["std_exact_minus_local"] for r in rows_sorted],
        [r["mean_exact_minus_local"] + r["std_exact_minus_local"] for r in rows_sorted],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Gaussian Logw Minus Local C")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("difference")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(step, [r["mean_exact_minus_quad"] for r in rows_sorted], label="mean")
    ax.fill_between(
        step,
        [r["mean_exact_minus_quad"] - r["std_exact_minus_quad"] for r in rows_sorted],
        [r["mean_exact_minus_quad"] + r["std_exact_minus_quad"] for r in rows_sorted],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Gaussian Logw Minus Local+Quadratic")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("difference")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 0]
    ax.plot(step, [r["mean_lhat"] for r in rows_sorted], label="mean lhat")
    ax.fill_between(
        step,
        [r["mean_lhat"] - r["std_lhat"] for r in rows_sorted],
        [r["mean_lhat"] + r["std_lhat"] for r in rows_sorted],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("lhat Mean +/- Std")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("lhat")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 1]
    ax.plot(step, [r["min_lhat"] for r in rows_sorted], label="min")
    ax.plot(step, [r["max_lhat"] for r in rows_sorted], label="max")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("lhat Range")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("lhat")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4, 0]
    ax.semilogy(step, [max(r["mean_gl2_per_dim"], 1e-30) for r in rows_sorted], label="mean ||g_l||^2 / dim")
    ax.set_title("Negative Direction Score Norm")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("mean ||g_l||^2 / dim")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4, 1]
    ax.plot(step, [r["c_t"] for r in rows_sorted], label="c")
    ax.plot(step, [r["bs_t"] for r in rows_sorted], label="bs")
    ax.set_title("Soft Gate Schedule")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / "smc_gaussian_monitor.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_prompt(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, float]]:
    prompts = load_prompts(args.prompts_file)
    if args.prompt_index < 1 or args.prompt_index > len(prompts):
        raise ValueError(f"--prompt-index must be in [1, {len(prompts)}]")
    prompt = prompts[args.prompt_index - 1]
    negative_key = "related_negative" if args.negative_kind == "related" else "unrelated_negative"
    negative = prompt[negative_key]
    prompt_hp = prompt.get("dng_hyperparams", {})
    table_hp = DNG_HYPERPARAMS_BY_NAME.get(prompt["name"], {"prior": 0.01, "temp": 0.2, "offset": 0.0})
    hp = {
        "prior": float(prompt_hp.get("prior", table_hp["prior"])),
        "temp": float(prompt_hp.get("temperature", prompt_hp.get("temp", table_hp["temp"]))),
        "offset": float(prompt_hp.get("offset", table_hp["offset"])),
    }
    return prompt, negative, hp


def fill_hyperparameter_defaults(args: argparse.Namespace, hp: dict[str, float]) -> None:
    prior = float(args.prior if args.prior is not None else hp["prior"])
    temp = float(args.temp if args.temp is not None else hp["temp"])
    odds = prior / max(1.0 - prior, 1e-12)
    bs = 1.0 / max(temp, 1e-12)

    if args.c0 is None:
        args.c0 = odds
    if args.c1 is None:
        args.c1 = odds
    if args.bs0 is None:
        args.bs0 = bs
    if args.bs1 is None:
        args.bs1 = bs
    if args.lhat_offset is None:
        args.lhat_offset = 0.0

    args.prior = prior
    args.temp = temp


def run_monitor(args: argparse.Namespace) -> None:
    prompt, negative, hp = resolve_prompt(args)
    fill_hyperparameter_defaults(args, hp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "prompts_file": str(args.prompts_file),
        "prompt": prompt,
        "negative_prompt_used": negative,
        "dng_table_hyperparams": hp,
    }
    with (args.out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    print("[task]", prompt["name"], flush=True)
    print("[positive]", prompt["positive"], flush=True)
    print(f"[negative:{args.negative_kind}]", negative, flush=True)
    print(
        f"[gate] c0={args.c0:.6g} c1={args.c1:.6g} bs0={args.bs0:.6g} bs1={args.bs1:.6g} "
        f"theta_eps={args.theta_eps:g}",
        flush=True,
    )
    print(
        f"[weights] mode={args.weight_mode} lhat_temp={args.lhat_temp:g} "
        f"lhat_offset={args.lhat_offset:g} lhat_clip={args.lhat_clip:g}",
        flush=True,
    )

    if args.dry_run:
        print("[dry-run] configuration written; model not loaded.", flush=True)
        return

    pipe = load_sd_pipeline(args.device, args.model_id)
    height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor
    n = args.n_particles

    pos_embeds = encode_text(pipe, [prompt["positive"]] * n, args.device)
    uncond_embeds = encode_text(pipe, [""] * n, args.device)
    neg_embeds = encode_text(pipe, [negative] * n, args.device)
    prompt_embeds = torch.cat([uncond_embeds, pos_embeds, neg_embeds], dim=0)

    pipe.scheduler.set_timesteps(args.steps, device=args.device)
    timesteps = list(pipe.scheduler.timesteps)

    init_generators = [torch.Generator(device=args.device).manual_seed(args.seed + i) for i in range(n)]
    step_generators = [torch.Generator(device=args.device).manual_seed(args.seed + 100000 + i) for i in range(n)]
    resample_gen = torch.Generator(device=args.device).manual_seed(args.seed + 200000)

    latents = pipe.prepare_latents(
        n,
        pipe.unet.config.in_channels,
        height,
        width,
        pos_embeds.dtype,
        args.device,
        init_generators,
        latents=None,
    )
    lhat = torch.zeros(n, device=args.device, dtype=torch.float32)
    logw_total = torch.zeros(n, device=args.device, dtype=torch.float32)

    rows: list[dict[str, Any]] = []
    start = time.time()
    print(f"[run] {len(timesteps)} reverse steps, {n} particles on {args.device}", flush=True)

    with torch.inference_mode():
        for step, t in enumerate(timesteps):
            t_int = int(t)
            c, bs = schedule_at_state(step, len(timesteps), args)
            c_new, bs_new = schedule_at_state(step + 1, len(timesteps), args)

            latent_model_input = torch.cat([latents, latents, latents], dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            model_out = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            out_uncond, out_pos, out_neg = model_out.chunk(3)
            if args.base_mode == "cfg":
                out_base = out_uncond + args.cfg_scale * (out_pos - out_uncond)
            elif args.base_mode == "positive":
                out_base = out_pos
            else:
                raise ValueError(f"Unknown base mode {args.base_mode!r}")

            theta = theta_from_lhat(lhat, c, bs, args.theta_eps).to(dtype=latents.dtype)
            guidance = theta[:, None, None, None] * float(bs)
            guided_out = out_base - guidance * (out_neg - out_base)

            x_new = pipe.scheduler.step(
                guided_out,
                t_int,
                latents,
                generator=step_generators,
                return_dict=False,
            )[0]

            mu_base, variance = ddpm_reverse_mean_and_variance(pipe.scheduler, out_base, t_int, latents)
            mu_neg, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, out_neg, t_int, latents)
            mu_guided, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, guided_out, t_int, latents)

            logp_base = gaussian_log_prob_isotropic(x_new, mu_base, variance)
            logp_neg = gaussian_log_prob_isotropic(x_new, mu_neg, variance)
            logp_guided = gaussian_log_prob_isotropic(x_new, mu_guided, variance)

            kernel_lr = logp_neg - logp_base
            offset_term = 0.5 * float(args.lhat_offset) / variance
            lhat_new = lhat + float(args.lhat_temp) * kernel_lr + offset_term
            if args.lhat_clip is not None and args.lhat_clip > 0:
                lhat_new = lhat_new.clamp(-float(args.lhat_clip), float(args.lhat_clip))

            theta_term = log1m_theta_from_lhat(lhat_new, c_new, bs_new) - log1m_theta_from_lhat(lhat, c, bs)
            logw_gaussian = theta_term + logp_base - logp_guided

            eps_base = model_output_to_epsilon(pipe.scheduler, out_base, t_int, latents)
            eps_neg = model_output_to_epsilon(pipe.scheduler, out_neg, t_int, latents)
            alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)
            beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
            g_l = -(eps_neg - eps_base) / beta_prod_t.sqrt()
            g_l_flat = g_l.float().flatten(1)
            gl2 = g_l_flat.pow(2).sum(dim=1)

            schedule_term = log1m_theta_from_lhat(lhat, c_new, bs_new) - log1m_theta_from_lhat(lhat, c, bs)
            theta_f = theta.float()
            var_f = variance.to(device=latents.device, dtype=torch.float32)
            bs_f = float(bs)
            logw_local = schedule_term + 0.5 * var_f * theta_f * bs_f * (1.0 - bs_f + 2.0 * theta_f * bs_f) * gl2

            std_noise = (x_new.float() - mu_guided.float()).flatten(1) / var_f.sqrt()
            gl_dot_noise = (g_l_flat * std_noise).sum(dim=1)
            quad_centered = -0.5 * var_f * theta_f * (1.0 - theta_f) * (bs_f**2) * (
                gl_dot_noise.pow(2) - gl2
            )
            logw_quad = logw_local + quad_centered

            if args.weight_mode == "gaussian":
                logw_used = logw_gaussian
            elif args.weight_mode == "local":
                logw_used = logw_local
            elif args.weight_mode == "quad":
                logw_used = logw_quad
            else:
                raise ValueError(f"Unknown weight mode {args.weight_mode!r}")

            inc_ess = ess_frac_from_logw(logw_used)
            logw_total_new = logw_total + logw_used.float()
            cum_w = normalized_weights_from_logw(logw_total_new)
            cum_ess = float(1.0 / (n * torch.sum(cum_w * cum_w)).item())
            do_resample = bool(args.resample_ess <= 1.0 and cum_ess < args.resample_ess)

            exact_minus_local = logw_gaussian - logw_local
            exact_minus_quad = logw_gaussian - logw_quad
            dim = int(latents[0].numel())
            row: dict[str, Any] = {
                "step": step,
                "timestep": t_int,
                "variance": float(var_f.item()),
                "c_t": c,
                "bs_t": bs,
                "c_next": c_new,
                "bs_next": bs_new,
                "base_mode": args.base_mode,
                "weight_mode": args.weight_mode,
                "inc_ess_frac_gaussian": ess_frac_from_logw(logw_gaussian),
                "inc_ess_frac_local": ess_frac_from_logw(logw_local),
                "inc_ess_frac_quad": ess_frac_from_logw(logw_quad),
                "inc_ess_frac_used": inc_ess,
                "cum_ess_frac_used": cum_ess,
                "resampled": int(do_resample),
                "guidance_mean": float((theta_f * bs_f).mean().item()),
                "guidance_max": float((theta_f * bs_f).max().item()),
                "mean_gl2": float(gl2.mean().item()),
                "mean_gl2_per_dim": float(gl2.mean().item() / max(dim, 1)),
                "std_gl2": float(gl2.std(unbiased=False).item()),
                "mean_kernel_lr": float(kernel_lr.mean().item()),
                "std_kernel_lr": float(kernel_lr.std(unbiased=False).item()),
                "offset_term": float(offset_term.item() if torch.is_tensor(offset_term) else offset_term),
                "corr_gaussian_local": finite_corr(logw_gaussian, logw_local),
                "corr_gaussian_quad": finite_corr(logw_gaussian, logw_quad),
            }
            row.update(tensor_stats("theta", theta_f))
            row.update(tensor_stats("lhat", lhat))
            row.update(tensor_stats("lhat_new", lhat_new))
            row.update(tensor_stats("logw_gaussian", logw_gaussian))
            row.update(tensor_stats("logw_local", logw_local))
            row.update(tensor_stats("logw_quad", logw_quad))
            row.update(tensor_stats("exact_minus_local", exact_minus_local))
            row.update(tensor_stats("exact_minus_quad", exact_minus_quad))
            rows.append(row)

            if step % max(args.log_every, 1) == 0 or step == len(timesteps) - 1:
                print(
                    f"[step {step:03d}/{len(timesteps)} t={t_int:4d}] "
                    f"theta={row['mean_theta']:.4f} guidance={row['guidance_mean']:.3f} "
                    f"ESS={cum_ess:.3f} logw_std={row['std_logw_gaussian']:.3g} "
                    f"diff_std={row['std_exact_minus_local']:.3g} resample={int(do_resample)}",
                    flush=True,
                )

            if do_resample:
                idx = systematic_resample(cum_w, resample_gen)
                latents = x_new[idx]
                lhat = lhat_new[idx]
                logw_total = torch.zeros_like(logw_total)
            else:
                latents = x_new
                lhat = lhat_new
                logw_total = logw_total_new

    elapsed = time.time() - start
    rows_path = args.out_dir / "smc_gaussian_monitor.csv"
    write_rows(rows_path, rows)
    plot_monitor(rows, args.out_dir, args.resample_ess)

    summary = {
        "elapsed_sec": elapsed,
        "n_steps": len(timesteps),
        "n_particles": n,
        "resamples": int(sum(r["resampled"] for r in rows)),
        "final_cum_ess_frac": rows[-1]["cum_ess_frac_used"],
        "mean_std_logw_gaussian": float(np.mean([r["std_logw_gaussian"] for r in rows])),
        "mean_std_exact_minus_local": float(np.mean([r["std_exact_minus_local"] for r in rows])),
        "mean_std_exact_minus_quad": float(np.mean([r["std_exact_minus_quad"] for r in rows])),
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if args.save_images:
        with torch.inference_mode():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            images = pipe.image_processor.postprocess(
                decoded.detach(),
                output_type="pil",
                do_denormalize=[True] * n,
            )
        img_dir = args.out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(img_dir / f"particle_{i:03d}.png")
        grid = make_image_grid(images, cols=args.grid_cols)
        if grid is not None:
            grid.save(args.out_dir / "particles_grid.png")

    print(f"[done] {elapsed:.1f}s", flush=True)
    print(f"[saved] {rows_path}", flush=True)
    print(f"[saved] {args.out_dir / 'smc_gaussian_monitor.png'}", flush=True)
    print(f"[saved] {args.out_dir / 'summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion SMC monitor with finite-step Gaussian log weights")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-id", default=SD_MODEL_ID)
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--negative-kind", choices=["related", "unrelated"], default="related")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "smc_gaussian_monitor_prompt1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--base-mode", choices=["cfg", "positive"], default="cfg")
    parser.add_argument("--cfg-scale", type=float, default=7.5)

    parser.add_argument("--prior", type=float, default=None)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--c0", type=float, default=None, help="Low-noise/final c; default prior/(1-prior)")
    parser.add_argument("--c1", type=float, default=None, help="High-noise/initial c; default prior/(1-prior)")
    parser.add_argument("--bs0", type=float, default=None, help="Low-noise/final bs; default 1/temp")
    parser.add_argument("--bs1", type=float, default=None, help="High-noise/initial bs; default 1/temp")
    parser.add_argument("--c-schedule", choices=["constant", "linear", "log", "exp"], default="constant")
    parser.add_argument("--bs-schedule", choices=["constant", "linear", "log", "exp"], default="constant")
    parser.add_argument("--gamma-c", type=float, default=1.0)
    parser.add_argument("--gamma-bs", type=float, default=2.0)
    parser.add_argument("--theta-eps", type=float, default=1e-6)

    parser.add_argument("--lhat-temp", type=float, default=1.0)
    parser.add_argument(
        "--lhat-offset",
        type=float,
        default=None,
        help="Offset delta used as +0.5*delta/variance in lhat update; default 0 for exact log-ratio.",
    )
    parser.add_argument("--lhat-clip", type=float, default=50.0)
    parser.add_argument("--weight-mode", choices=["gaussian", "local", "quad"], default="gaussian")
    parser.add_argument("--resample-ess", type=float, default=0.5, help="Cumulative ESS/N threshold; set >1 to disable")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monitor(args)


if __name__ == "__main__":
    main()
