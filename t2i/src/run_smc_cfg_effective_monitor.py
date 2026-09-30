#!/usr/bin/env python3
"""Monitor SMC negative guidance with CFG-effective A/B targets.

This script tests the text-to-image interpretation where CFG, not the raw
conditional branch, defines the effective prompt target:

    s_A^eff = s_un + lambda_A (s_A - s_un)
    s_B^eff = s_un + lambda_B (s_B - s_un)

The soft negative target is

    q_t(x) ∝ p_A^eff(x) / [1 + c_t (p_B^eff(x) / p_A^eff(x))^eta_t],

so its score is

    s_q = s_A^eff - theta_t eta_t (s_B^eff - s_A^eff),
    theta_t = sigmoid(log c_t + eta_t lhat_t).

In model-output units this becomes

    out_A_eff = out_un + lambda_A (out_A - out_un)
    out_B_eff = out_un + lambda_B (out_B - out_un)
    out_guided = out_A_eff - theta_t eta_t (out_B_eff - out_A_eff).

The incremental SMC weight is the finite-step Gaussian weight:

    log w =
        log(1 - theta_new) - log(1 - theta_old)
        + log N(x_new; mu_A_eff, sigma_t^2 I)
        - log N(x_new; mu_guided, sigma_t^2 I).

Default task:
    prompt 1, "Medieval feast", related negative prompt "Chalices, candles".
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

from run_dng_eval import DNG_HYPERPARAMS_BY_NAME, PROMPTS_FILE, ddpm_reverse_mean_and_variance  # noqa: E402
from run_np_grid_eval import HF_CACHE, SD_MODEL_ID  # noqa: E402
from run_smc_gaussian_monitor import (  # noqa: E402
    encode_text,
    ess_frac_from_logw,
    finite_corr,
    gaussian_log_prob_isotropic,
    log1m_theta_from_lhat,
    make_image_grid,
    model_output_to_epsilon,
    normalized_weights_from_logw,
    schedule_at_state,
    systematic_resample,
    tensor_stats,
    theta_from_lhat,
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
        local_files_only=os.environ.get("PROP2_LOCAL_FILES_ONLY", "0") == "1",
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


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

    if args.c0 is None:
        args.c0 = 3.0
    if args.c1 is None:
        args.c1 = 0.01
    if args.bs0 is None:
        args.bs0 = 3.0
    if args.bs1 is None:
        args.bs1 = 0.01
    if args.lhat_offset is None:
        args.lhat_offset = 0.0

    args.prior = prior
    args.temp = temp


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decode_and_save_images(
    pipe,
    latents: torch.Tensor,
    args: argparse.Namespace,
    start_index: int = 0,
    make_grid_after: bool = True,
) -> None:
    batch_size = max(1, int(args.vae_decode_batch_size))
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if latents.device.type == "cuda":
        torch.cuda.empty_cache()

    images = []
    with torch.inference_mode():
        for start in range(0, latents.shape[0], batch_size):
            batch = latents[start : start + batch_size]
            decoded = pipe.vae.decode(batch / pipe.vae.config.scaling_factor, return_dict=False)[0]
            decoded = decoded.detach()
            pil_batch = pipe.image_processor.postprocess(
                decoded,
                output_type="pil",
                do_denormalize=[True] * decoded.shape[0],
            )
            for offset, img in enumerate(pil_batch):
                img.save(img_dir / f"particle_{start_index + start + offset:03d}.png")
            images.extend(pil_batch)
            del decoded
            if latents.device.type == "cuda":
                torch.cuda.empty_cache()

    if make_grid_after:
        grid = make_image_grid(images, cols=args.grid_cols)
    else:
        grid = None
    if grid is not None:
        grid.save(args.out_dir / "particles_grid.png")


def assemble_grid_from_saved_images(args: argparse.Namespace, n_images: int) -> None:
    from PIL import Image

    img_dir = args.out_dir / "images"
    images = []
    for idx in range(n_images):
        path = img_dir / f"particle_{idx:03d}.png"
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    grid = make_image_grid(images, cols=args.grid_cols)
    if grid is not None:
        grid.save(args.out_dir / "particles_grid.png")


def write_decode_done_marker(out_dir: Path, rank: int) -> None:
    marker_dir = out_dir / "decode_done"
    marker_dir.mkdir(parents=True, exist_ok=True)
    with (marker_dir / f"rank_{rank}.done").open("w") as f:
        f.write("done\n")


def wait_for_decode_markers(out_dir: Path, world_size: int, timeout_sec: float) -> None:
    marker_dir = out_dir / "decode_done"
    deadline = time.time() + float(timeout_sec)
    expected = [marker_dir / f"rank_{rank}.done" for rank in range(world_size)]
    while time.time() < deadline:
        if all(path.exists() for path in expected):
            return
        time.sleep(1.0)
    missing = [str(path) for path in expected if not path.exists()]
    raise TimeoutError(f"Timed out waiting for decode markers: {missing}")


def all_gather_cat(x: torch.Tensor, world_size: int) -> torch.Tensor:
    pieces = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(pieces, x.contiguous())
    return torch.cat(pieces, dim=0)


def distributed_barrier(device: str) -> None:
    if dist.get_backend() == "nccl" and device.startswith("cuda"):
        dist.barrier(device_ids=[torch.device(device).index])
    else:
        dist.barrier()


def should_skip_smc_weight(scheduler, timestep: int, variance: torch.Tensor, min_variance: float) -> bool:
    prev_t = int(scheduler.previous_timestep(int(timestep)))
    variance_value = float(variance.detach().float().item())
    return prev_t < 0 or variance_value <= float(min_variance)


def global_normalized_weights_from_logw(logw: torch.Tensor) -> torch.Tensor:
    return torch.exp(logw - torch.logsumexp(logw, dim=0))


def global_ess_frac_from_logw(logw: torch.Tensor) -> float:
    w = global_normalized_weights_from_logw(logw)
    return float(1.0 / (w.shape[0] * torch.sum(w * w)).item())


def make_monitor_row(
    args: argparse.Namespace,
    step: int,
    t_int: int,
    c: float,
    eta: float,
    c_new: float,
    eta_new: float,
    variance: torch.Tensor,
    theta: torch.Tensor,
    lhat: torch.Tensor,
    lhat_new: torch.Tensor,
    gl2: torch.Tensor,
    kernel_lr: torch.Tensor,
    logw_gaussian: torch.Tensor,
    logw_local: torch.Tensor,
    logw_quad: torch.Tensor,
    logw_used: torch.Tensor,
    logw_total_new: torch.Tensor,
    offset_term: torch.Tensor | float,
    skipped_weight: bool,
    do_resample: bool,
    dim: int,
) -> dict[str, Any]:
    exact_minus_local = logw_gaussian - logw_local
    exact_minus_quad = logw_gaussian - logw_quad
    theta_f = theta.float()
    eta_f = float(eta)
    row: dict[str, Any] = {
        "step": step,
        "timestep": t_int,
        "variance": float(variance.detach().float().item()),
        "c_t": c,
        "bs_t": eta,
        "c_next": c_new,
        "bs_next": eta_new,
        "lambda_a": float(args.lambda_a),
        "lambda_b": float(args.lambda_b),
        "weight_mode": args.weight_mode,
        "inc_ess_frac_gaussian": ess_frac_from_logw(logw_gaussian),
        "inc_ess_frac_local": ess_frac_from_logw(logw_local),
        "inc_ess_frac_quad": ess_frac_from_logw(logw_quad),
        "inc_ess_frac_used": ess_frac_from_logw(logw_used),
        "cum_ess_frac_used": global_ess_frac_from_logw(logw_total_new),
        "cum_ess_frac_after_resample": 1.0 if do_resample else global_ess_frac_from_logw(logw_total_new),
        "resampled": int(do_resample),
        "skipped_weight": int(skipped_weight),
        "guidance_mean": float((theta_f * eta_f).mean().item()),
        "guidance_max": float((theta_f * eta_f).max().item()),
        "mean_gl2": float(gl2.float().mean().item()),
        "mean_gl2_per_dim": float(gl2.float().mean().item() / max(dim, 1)),
        "std_gl2": float(gl2.float().std(unbiased=False).item()),
        "mean_kernel_lr": float(kernel_lr.float().mean().item()),
        "std_kernel_lr": float(kernel_lr.float().std(unbiased=False).item()),
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
    return row


def plot_monitor(rows: list[dict[str, Any]], out_dir: Path, resample_ess: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["step"])
    step = np.array([r["step"] for r in rows])

    fig, axes = plt.subplots(5, 2, figsize=(15, 18), dpi=140)

    ax = axes[0, 0]
    ax.plot(step, [r["guidance_mean"] for r in rows], label="mean theta*eta")
    ax.plot(step, [r["guidance_max"] for r in rows], ls="--", alpha=0.7, label="max theta*eta")
    ax.set_title("Effective Negative-Gate Guidance")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("theta * eta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(step, [r["mean_theta"] for r in rows], label="mean theta")
    ax.plot(step, [r["min_theta"] for r in rows], ls="--", label="min theta")
    ax.plot(step, [r["max_theta"] for r in rows], ls="--", label="max theta")
    ax.set_title("Theta Statistics")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(step, [r["inc_ess_frac_used"] for r in rows], label="incremental")
    ax.plot(step, [r["cum_ess_frac_used"] for r in rows], label="cumulative before resample")
    if "cum_ess_frac_after_resample" in rows[0]:
        ax.plot(
            step,
            [r["cum_ess_frac_after_resample"] for r in rows],
            ls="--",
            alpha=0.8,
            label="cumulative after resample",
        )
    if resample_ess <= 1.0:
        ax.axhline(resample_ess, color="tab:red", ls=":", label="resample threshold")
    resampled = [r for r in rows if int(r["resampled"]) == 1]
    if resampled:
        ax.scatter(
            [r["step"] for r in resampled],
            [r["cum_ess_frac_used"] for r in resampled],
            s=20,
            color="black",
            label="resampled",
        )
    ax.set_title("ESS/N From Gaussian Weights")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("ESS/N")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step, [r["std_logw_gaussian"] for r in rows], label="gaussian")
    ax.plot(step, [r["std_logw_local"] for r in rows], label="local mean")
    ax.plot(step, [r["std_logw_quad"] for r in rows], label="local + quad")
    ax.set_title("Per-Step Log-Weight Spread")
    ax.set_xlabel("reverse step")
    ax.set_ylabel("std(log w)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(step, [r["mean_exact_minus_local"] for r in rows], label="mean")
    ax.fill_between(
        step,
        [r["mean_exact_minus_local"] - r["std_exact_minus_local"] for r in rows],
        [r["mean_exact_minus_local"] + r["std_exact_minus_local"] for r in rows],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Gaussian Logw Minus Local Mean")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(step, [r["mean_exact_minus_quad"] for r in rows], label="mean")
    ax.fill_between(
        step,
        [r["mean_exact_minus_quad"] - r["std_exact_minus_quad"] for r in rows],
        [r["mean_exact_minus_quad"] + r["std_exact_minus_quad"] for r in rows],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Gaussian Logw Minus Local+Quadratic")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 0]
    ax.plot(step, [r["mean_lhat"] for r in rows], label="mean lhat")
    ax.fill_between(
        step,
        [r["mean_lhat"] - r["std_lhat"] for r in rows],
        [r["mean_lhat"] + r["std_lhat"] for r in rows],
        alpha=0.2,
        label="+/- std",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("lhat Mean +/- Std")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 1]
    ax.plot(step, [r["min_lhat"] for r in rows], label="min")
    ax.plot(step, [r["max_lhat"] for r in rows], label="max")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("lhat Range")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4, 0]
    ax.semilogy(step, [max(r["mean_gl2_per_dim"], 1e-30) for r in rows], label="mean ||g_l||^2 / dim")
    ax.set_title("CFG-Effective Score Difference Norm")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4, 1]
    ax.plot(step, [r["lambda_a"] for r in rows], label="lambda_A")
    ax.plot(step, [r["lambda_b"] for r in rows], label="lambda_B")
    ax.plot(step, [r["bs_t"] for r in rows], label="eta")
    ax.plot(step, [r["c_t"] for r in rows], label="c")
    ax.set_title("CFG and Gate Parameters")
    ax.set_xlabel("reverse step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "smc_cfg_effective_monitor.png", bbox_inches="tight")
    plt.close(fig)


def run_monitor(args: argparse.Namespace) -> None:
    prompt, negative, hp = resolve_prompt(args)
    fill_hyperparameter_defaults(args, hp)

    if args.lambda_b is None:
        args.lambda_b = args.lambda_a

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
    print(f"[cfg-effective] lambda_A={args.lambda_a:g} lambda_B={args.lambda_b:g}", flush=True)
    print(
        f"[gate] c0={args.c0:.6g} c1={args.c1:.6g} eta0={args.bs0:.6g} eta1={args.bs1:.6g} "
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
            c, eta = schedule_at_state(step, len(timesteps), args)
            c_new, eta_new = schedule_at_state(step + 1, len(timesteps), args)

            latent_model_input = torch.cat([latents, latents, latents], dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            model_out = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            out_uncond, out_pos, out_neg = model_out.chunk(3)

            out_a_eff = out_uncond + float(args.lambda_a) * (out_pos - out_uncond)
            out_b_eff = out_uncond + float(args.lambda_b) * (out_neg - out_uncond)

            theta = theta_from_lhat(lhat, c, eta, args.theta_eps).to(dtype=latents.dtype)
            gate_strength = theta[:, None, None, None] * float(eta)
            guided_out = out_a_eff - gate_strength * (out_b_eff - out_a_eff)

            x_new = pipe.scheduler.step(
                guided_out,
                t_int,
                latents,
                generator=step_generators,
                return_dict=False,
            )[0]

            mu_a, variance = ddpm_reverse_mean_and_variance(pipe.scheduler, out_a_eff, t_int, latents)
            mu_b, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, out_b_eff, t_int, latents)
            mu_guided, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, guided_out, t_int, latents)

            eps_a_eff = model_output_to_epsilon(pipe.scheduler, out_a_eff, t_int, latents)
            eps_b_eff = model_output_to_epsilon(pipe.scheduler, out_b_eff, t_int, latents)
            alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)
            beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
            g_l = -(eps_b_eff - eps_a_eff) / beta_prod_t.sqrt()
            g_l_flat = g_l.float().flatten(1)
            gl2 = g_l_flat.pow(2).sum(dim=1)

            theta_f = theta.float()
            eta_f = float(eta)
            var_f = variance.to(device=latents.device, dtype=torch.float32)
            skipped_weight = should_skip_smc_weight(
                pipe.scheduler,
                t_int,
                var_f,
                args.min_weight_variance,
            )

            if skipped_weight:
                logw_gaussian = torch.zeros_like(lhat)
                logw_local = torch.zeros_like(lhat)
                logw_quad = torch.zeros_like(lhat)
                kernel_lr = torch.zeros_like(lhat)
                offset_term = torch.tensor(0.0, device=latents.device, dtype=torch.float32)
                lhat_new = lhat
            else:
                logp_a = gaussian_log_prob_isotropic(x_new, mu_a, variance)
                logp_b = gaussian_log_prob_isotropic(x_new, mu_b, variance)
                logp_guided = gaussian_log_prob_isotropic(x_new, mu_guided, variance)

                kernel_lr = logp_b - logp_a
                offset_term = 0.5 * float(args.lhat_offset) / variance
                lhat_new = lhat + float(args.lhat_temp) * kernel_lr + offset_term
                if args.lhat_clip is not None and args.lhat_clip > 0:
                    lhat_new = lhat_new.clamp(-float(args.lhat_clip), float(args.lhat_clip))

                theta_term = log1m_theta_from_lhat(lhat_new, c_new, eta_new) - log1m_theta_from_lhat(lhat, c, eta)
                logw_gaussian = theta_term + logp_a - logp_guided

                schedule_term = log1m_theta_from_lhat(lhat, c_new, eta_new) - log1m_theta_from_lhat(lhat, c, eta)
                logw_local = schedule_term + 0.5 * var_f * theta_f * eta_f * (
                    1.0 - eta_f + 2.0 * theta_f * eta_f
                ) * gl2

                std_noise = (x_new.float() - mu_guided.float()).flatten(1) / var_f.sqrt()
                gl_dot_noise = (g_l_flat * std_noise).sum(dim=1)
                quad_centered = -0.5 * var_f * theta_f * (1.0 - theta_f) * (eta_f**2) * (
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

            dim = int(latents[0].numel())
            row = make_monitor_row(
                args=args,
                step=step,
                t_int=t_int,
                c=c,
                eta=eta,
                c_new=c_new,
                eta_new=eta_new,
                variance=var_f,
                theta=theta_f,
                lhat=lhat,
                lhat_new=lhat_new,
                gl2=gl2,
                kernel_lr=kernel_lr,
                logw_gaussian=logw_gaussian,
                logw_local=logw_local,
                logw_quad=logw_quad,
                logw_used=logw_used,
                logw_total_new=logw_total_new,
                offset_term=offset_term,
                skipped_weight=skipped_weight,
                do_resample=do_resample,
                dim=dim,
            )
            rows.append(row)

            if step % max(args.log_every, 1) == 0 or step == len(timesteps) - 1:
                print(
                    f"[step {step:03d}/{len(timesteps)} t={t_int:4d}] "
                    f"theta={row['mean_theta']:.4f} gate={row['guidance_mean']:.3f} "
                    f"incESS={inc_ess:.3f} cumESS={cum_ess:.3f} "
                    f"logw_std={row['std_logw_gaussian']:.3g} "
                    f"diff_std={row['std_exact_minus_local']:.3g} "
                    f"skip={int(skipped_weight)} resample={int(do_resample)}",
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
    rows_path = args.out_dir / "smc_cfg_effective_monitor.csv"
    write_rows(rows_path, rows)
    plot_monitor(rows, args.out_dir, args.resample_ess)

    summary = {
        "elapsed_sec": elapsed,
        "n_steps": len(timesteps),
        "n_particles": n,
        "resamples": int(sum(r["resampled"] for r in rows)),
        "final_cum_ess_frac": rows[-1]["cum_ess_frac_after_resample"],
        "final_cum_ess_frac_before_resample": rows[-1]["cum_ess_frac_used"],
        "mean_std_logw_gaussian": float(np.mean([r["std_logw_gaussian"] for r in rows])),
        "mean_std_exact_minus_local": float(np.mean([r["std_exact_minus_local"] for r in rows])),
        "mean_std_exact_minus_quad": float(np.mean([r["std_exact_minus_quad"] for r in rows])),
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if args.save_images:
        decode_and_save_images(pipe, latents, args)

    print(f"[done] {elapsed:.1f}s", flush=True)
    print(f"[saved] {rows_path}", flush=True)
    print(f"[saved] {args.out_dir / 'smc_cfg_effective_monitor.png'}", flush=True)
    print(f"[saved] {args.out_dir / 'summary.json'}", flush=True)


def parse_devices(devices_text: str | None, fallback: str) -> list[str]:
    if not devices_text:
        return [fallback]
    devices = [item.strip() for item in devices_text.split(",") if item.strip()]
    out = []
    for device in devices:
        out.append(f"cuda:{device}" if device.isdigit() else device)
    return out or [fallback]


def run_distributed_monitor(args: argparse.Namespace) -> None:
    devices = parse_devices(args.devices, args.device)
    if len(devices) <= 1:
        args.device = devices[0]
        run_monitor(args)
        return

    if args.n_particles % len(devices) != 0:
        raise ValueError(
            f"--n-particles ({args.n_particles}) must be divisible by the number "
            f"of devices ({len(devices)}) for cross-GPU SMC."
        )

    prompt, negative, hp = resolve_prompt(args)
    fill_hyperparameter_defaults(args, hp)
    if args.lambda_b is None:
        args.lambda_b = args.lambda_a

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "prompts_file": str(args.prompts_file),
        "devices": devices,
        "prompt": prompt,
        "negative_prompt_used": negative,
        "dng_table_hyperparams": hp,
        "distributed": True,
    }
    with (args.out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    print("[task]", prompt["name"], flush=True)
    print("[positive]", prompt["positive"], flush=True)
    print(f"[negative:{args.negative_kind}]", negative, flush=True)
    print(
        f"[distributed] devices={','.join(devices)} total_particles={args.n_particles} "
        f"local_particles={args.n_particles // len(devices)}",
        flush=True,
    )
    print(f"[cfg-effective] lambda_A={args.lambda_a:g} lambda_B={args.lambda_b:g}", flush=True)
    print(
        f"[gate] c0={args.c0:.6g} c1={args.c1:.6g} eta0={args.bs0:.6g} eta1={args.bs1:.6g} "
        f"theta_eps={args.theta_eps:g}",
        flush=True,
    )
    print(
        f"[weights] mode={args.weight_mode} lhat_temp={args.lhat_temp:g} "
        f"lhat_offset={args.lhat_offset:g} lhat_clip={args.lhat_clip:g}",
        flush=True,
    )

    if args.dry_run:
        print("[dry-run] distributed configuration written; models not loaded.", flush=True)
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.dist_port)
    mp.spawn(
        distributed_worker,
        args=(args, devices),
        nprocs=len(devices),
        join=True,
    )


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
        height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        pos_embeds = encode_text(pipe, [prompt["positive"]] * local_n, device)
        uncond_embeds = encode_text(pipe, [""] * local_n, device)
        neg_embeds = encode_text(pipe, [negative] * local_n, device)
        prompt_embeds = torch.cat([uncond_embeds, pos_embeds, neg_embeds], dim=0)

        pipe.scheduler.set_timesteps(args.steps, device=device)
        timesteps = list(pipe.scheduler.timesteps)

        init_generators = [
            torch.Generator(device=device).manual_seed(args.seed + global_start + i)
            for i in range(local_n)
        ]
        step_generators = [
            torch.Generator(device=device).manual_seed(args.seed + 100000 + global_start + i)
            for i in range(local_n)
        ]
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

        rows: list[dict[str, Any]] = []
        start = time.time()
        if rank == 0:
            print(f"[run] {len(timesteps)} reverse steps, {args.n_particles} particles across {world_size} GPUs", flush=True)

        with torch.inference_mode():
            for step, t in enumerate(timesteps):
                t_int = int(t)
                c, eta = schedule_at_state(step, len(timesteps), args)
                c_new, eta_new = schedule_at_state(step + 1, len(timesteps), args)

                latent_model_input = torch.cat([latents, latents, latents], dim=0)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                model_out = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                out_uncond, out_pos, out_neg = model_out.chunk(3)

                out_a_eff = out_uncond + float(args.lambda_a) * (out_pos - out_uncond)
                out_b_eff = out_uncond + float(args.lambda_b) * (out_neg - out_uncond)

                theta = theta_from_lhat(lhat, c, eta, args.theta_eps).to(dtype=latents.dtype)
                gate_strength = theta[:, None, None, None] * float(eta)
                guided_out = out_a_eff - gate_strength * (out_b_eff - out_a_eff)

                x_new = pipe.scheduler.step(
                    guided_out,
                    t_int,
                    latents,
                    generator=step_generators,
                    return_dict=False,
                )[0]

                mu_a, variance = ddpm_reverse_mean_and_variance(pipe.scheduler, out_a_eff, t_int, latents)
                mu_b, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, out_b_eff, t_int, latents)
                mu_guided, _ = ddpm_reverse_mean_and_variance(pipe.scheduler, guided_out, t_int, latents)

                eps_a_eff = model_output_to_epsilon(pipe.scheduler, out_a_eff, t_int, latents)
                eps_b_eff = model_output_to_epsilon(pipe.scheduler, out_b_eff, t_int, latents)
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
                g_l = -(eps_b_eff - eps_a_eff) / beta_prod_t.sqrt()
                g_l_flat = g_l.float().flatten(1)
                gl2 = g_l_flat.pow(2).sum(dim=1)

                theta_f = theta.float()
                eta_f = float(eta)
                var_f = variance.to(device=latents.device, dtype=torch.float32)
                skipped_weight = should_skip_smc_weight(
                    pipe.scheduler,
                    t_int,
                    var_f,
                    args.min_weight_variance,
                )

                if skipped_weight:
                    logw_gaussian = torch.zeros_like(lhat)
                    logw_local = torch.zeros_like(lhat)
                    logw_quad = torch.zeros_like(lhat)
                    kernel_lr = torch.zeros_like(lhat)
                    offset_term = torch.tensor(0.0, device=latents.device, dtype=torch.float32)
                    lhat_new = lhat
                else:
                    logp_a = gaussian_log_prob_isotropic(x_new, mu_a, variance)
                    logp_b = gaussian_log_prob_isotropic(x_new, mu_b, variance)
                    logp_guided = gaussian_log_prob_isotropic(x_new, mu_guided, variance)

                    kernel_lr = logp_b - logp_a
                    offset_term = 0.5 * float(args.lhat_offset) / variance
                    lhat_new = lhat + float(args.lhat_temp) * kernel_lr + offset_term
                    if args.lhat_clip is not None and args.lhat_clip > 0:
                        lhat_new = lhat_new.clamp(-float(args.lhat_clip), float(args.lhat_clip))

                    theta_term = log1m_theta_from_lhat(lhat_new, c_new, eta_new) - log1m_theta_from_lhat(lhat, c, eta)
                    logw_gaussian = theta_term + logp_a - logp_guided

                    schedule_term = log1m_theta_from_lhat(lhat, c_new, eta_new) - log1m_theta_from_lhat(lhat, c, eta)
                    logw_local = schedule_term + 0.5 * var_f * theta_f * eta_f * (
                        1.0 - eta_f + 2.0 * theta_f * eta_f
                    ) * gl2

                    std_noise = (x_new.float() - mu_guided.float()).flatten(1) / var_f.sqrt()
                    gl_dot_noise = (g_l_flat * std_noise).sum(dim=1)
                    quad_centered = -0.5 * var_f * theta_f * (1.0 - theta_f) * (eta_f**2) * (
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

                logw_total_new = logw_total + logw_used.float()

                global_logw_gaussian = all_gather_cat(logw_gaussian.float(), world_size)
                global_logw_local = all_gather_cat(logw_local.float(), world_size)
                global_logw_quad = all_gather_cat(logw_quad.float(), world_size)
                global_logw_used = all_gather_cat(logw_used.float(), world_size)
                global_logw_total_new = all_gather_cat(logw_total_new.float(), world_size)
                global_theta = all_gather_cat(theta_f, world_size)
                global_lhat = all_gather_cat(lhat.float(), world_size)
                global_lhat_new = all_gather_cat(lhat_new.float(), world_size)
                global_gl2 = all_gather_cat(gl2.float(), world_size)
                global_kernel_lr = all_gather_cat(kernel_lr.float(), world_size)

                global_w = global_normalized_weights_from_logw(global_logw_total_new)
                cum_ess = float(1.0 / (args.n_particles * torch.sum(global_w * global_w)).item())
                inc_ess = global_ess_frac_from_logw(global_logw_used)
                do_resample = bool(args.resample_ess <= 1.0 and cum_ess < args.resample_ess)

                if rank == 0:
                    dim = int(latents[0].numel())
                    row = make_monitor_row(
                        args=args,
                        step=step,
                        t_int=t_int,
                        c=c,
                        eta=eta,
                        c_new=c_new,
                        eta_new=eta_new,
                        variance=var_f,
                        theta=global_theta,
                        lhat=global_lhat,
                        lhat_new=global_lhat_new,
                        gl2=global_gl2,
                        kernel_lr=global_kernel_lr,
                        logw_gaussian=global_logw_gaussian,
                        logw_local=global_logw_local,
                        logw_quad=global_logw_quad,
                        logw_used=global_logw_used,
                        logw_total_new=global_logw_total_new,
                        offset_term=offset_term,
                        skipped_weight=skipped_weight,
                        do_resample=do_resample,
                        dim=dim,
                    )
                    rows.append(row)
                    if step % max(args.log_every, 1) == 0 or step == len(timesteps) - 1:
                        print(
                            f"[step {step:03d}/{len(timesteps)} t={t_int:4d}] "
                            f"theta={row['mean_theta']:.4f} gate={row['guidance_mean']:.3f} "
                            f"incESS={inc_ess:.3f} cumESS={cum_ess:.3f} "
                            f"logw_std={row['std_logw_gaussian']:.3g} "
                            f"diff_std={row['std_exact_minus_local']:.3g} "
                            f"skip={int(skipped_weight)} resample={int(do_resample)}",
                            flush=True,
                        )

                if do_resample:
                    if rank == 0:
                        ancestor_idx = systematic_resample(global_w, resample_gen).long()
                    else:
                        ancestor_idx = torch.empty(args.n_particles, device=device, dtype=torch.long)
                    dist.broadcast(ancestor_idx, src=0)

                    all_x_new = all_gather_cat(x_new.contiguous(), world_size)
                    all_lhat_new = all_gather_cat(lhat_new.float(), world_size)
                    local_idx = ancestor_idx[global_start : global_start + local_n]
                    latents = all_x_new[local_idx].to(dtype=latents.dtype)
                    lhat = all_lhat_new[local_idx]
                    logw_total = torch.zeros_like(logw_total)
                else:
                    latents = x_new
                    lhat = lhat_new
                    logw_total = logw_total_new

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
            elapsed = time.time() - start
            rows_path = args.out_dir / "smc_cfg_effective_monitor.csv"
            write_rows(rows_path, rows)
            plot_monitor(rows, args.out_dir, args.resample_ess)
            summary = {
                "elapsed_sec": elapsed,
                "n_steps": len(timesteps),
                "n_particles": args.n_particles,
                "world_size": world_size,
                "local_particles": local_n,
                "resamples": int(sum(r["resampled"] for r in rows)),
                "final_cum_ess_frac": rows[-1]["cum_ess_frac_after_resample"],
                "final_cum_ess_frac_before_resample": rows[-1]["cum_ess_frac_used"],
                "mean_std_logw_gaussian": float(np.mean([r["std_logw_gaussian"] for r in rows])),
                "mean_std_exact_minus_local": float(np.mean([r["std_exact_minus_local"] for r in rows])),
                "mean_std_exact_minus_quad": float(np.mean([r["std_exact_minus_quad"] for r in rows])),
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
            print(f"[saved] {args.out_dir / 'smc_cfg_effective_monitor.png'}", flush=True)
            print(f"[saved] {args.out_dir / 'summary.json'}", flush=True)
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion SMC monitor with CFG-effective A/B targets")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for cross-GPU SMC, e.g. cuda:0,cuda:1,...,cuda:7. "
             "--n-particles is the total particle count.",
    )
    parser.add_argument("--dist-port", type=int, default=29673)
    parser.add_argument("--model-id", default=SD_MODEL_ID)
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--negative-kind", choices=["related", "unrelated"], default="related")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "smc_cfg_effective_monitor_prompt1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--lambda-a", type=float, default=7.5, help="CFG scale for the positive effective target")
    parser.add_argument("--lambda-b", type=float, default=None, help="CFG scale for the negative effective target; default lambda-a")

    parser.add_argument("--prior", type=float, default=None)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--c0", type=float, default=3.0, help="Low-noise/data-side c")
    parser.add_argument("--c1", type=float, default=0.01, help="High-noise/noise-side c")
    parser.add_argument("--bs0", type=float, default=3.0, help="Low-noise/data-side eta")
    parser.add_argument("--bs1", type=float, default=0.01, help="High-noise/noise-side eta")
    parser.add_argument("--c-schedule", choices=["constant", "linear", "log", "exp"], default="exp")
    parser.add_argument("--bs-schedule", choices=["constant", "linear", "log", "exp"], default="exp")
    parser.add_argument("--gamma-c", type=float, default=2.0, help="Matches Linrui's SMC exp c schedule default")
    parser.add_argument("--gamma-bs", type=float, default=3.0, help="Matches Linrui's SMC exp bs schedule default")
    parser.add_argument("--theta-eps", type=float, default=1e-6)

    parser.add_argument("--lhat-temp", type=float, default=1.0)
    parser.add_argument("--lhat-offset", type=float, default=None, help="Default 0 for exact Gaussian log-ratio")
    parser.add_argument("--lhat-clip", type=float, default=50.0)
    parser.add_argument("--weight-mode", choices=["gaussian", "local", "quad"], default="gaussian")
    parser.add_argument(
        "--min-weight-variance",
        type=float,
        default=1e-12,
        help="Skip SMC log-weight/resampling on degenerate final DDPM transitions with variance below this value.",
    )
    parser.add_argument("--resample-ess", type=float, default=0.5, help="Cumulative ESS/N threshold; set >1 to disable")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--vae-decode-batch-size", type=int, default=1)
    parser.add_argument("--image-marker-timeout", type=float, default=1800.0)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_distributed_monitor(args)


if __name__ == "__main__":
    main()
