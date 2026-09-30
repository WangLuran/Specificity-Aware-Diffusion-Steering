#!/usr/bin/env python3
"""Run Dynamic Negative Guidance on the five Stable Diffusion DNG prompts.

This script is the DNG counterpart of run_np_grid_eval.py. It uses the
project's DDPM posterior update and the Dynamic Negative Guidance paper.

For a positive prompt A and a negative prompt B, this practical Stable
Diffusion variant uses a standard positive-CFG image as the baseline, then
applies DNG along the negative-vs-baseline direction:

    baseline = out_uncond + cfg * (out_A - out_uncond)
    model_out = baseline - lambda0 * p_t / (1 - p_t) * (out_B - baseline)

This legacy default mixes a CFG A branch with a raw B branch.  Use
``--base-mode cfg --negative-model-mode cfg`` for the self-consistent
CFG-conditional substitution:

    out_A_cfg = out_uncond + cfg * (out_A - out_uncond)
    out_B_cfg = out_uncond + cfg * (out_B - out_uncond)
    model_out = out_A_cfg
                - lambda0 * p_t / (1 - p_t) * (out_B_cfg - out_A_cfg)

The posterior kernel comparison then also uses CFG-B versus CFG-A.

The posterior p_t is updated after each reverse step by comparing the sampled
new latent to the Gaussian reverse means predicted by the baseline and B:

    p_{t-1} = p_t * exp(0.5 / sigma_t^2 * (-tau * (d_B - d_A) + delta))

where d_B = ||x_{t-1} - mu_B||^2 and
d_A = ||x_{t-1} - mu_baseline||^2.

Use --base-mode positive to instead use the raw positive-prompt prediction as
the base branch:

    baseline = out_A
    model_out = baseline - lambda0 * p_t / (1 - p_t) * (out_B - baseline)

The prompt-specific prior, temperature, and offset are fixed to Table 3 of the
DNG paper.  By default, lambda0 is swept over the four values used in the paper
captions for Stable Diffusion examples: 12, 18, 21, 27.  Use
--paper-caption-guidance to instead run one paper-caption lambda0 per prompt.

Example on 8 GPUs:
  python run_dng_eval.py --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "HF_HOME",
    str(Path.home() / ".cache/huggingface"),
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_np_grid_eval import (  # noqa: E402
    HF_CACHE,
    SD_MODEL_ID,
    detect_devices,
    fmt_float,
    mean,
    parse_float_list,
    parse_negative_kinds,
    score_with_clip,
    sem,
    slug,
    write_csv,
)

TEXT2IMAGE_DIR = SCRIPT_DIR.parent
PROMPTS_FILE = TEXT2IMAGE_DIR / "dng_prompts.json"

DNG_HYPERPARAMS_BY_NAME = {
    "Medieval feast": {"prior": 0.01, "temp": 0.2, "offset": 0.004},
    "A dinner table": {"prior": 0.01, "temp": 0.3, "offset": 0.003},
    "An art workshop": {"prior": 0.01, "temp": 0.2, "offset": 0.002},
    "An antique store": {"prior": 0.01, "temp": 0.2, "offset": 0.004},
    "An English breakfast": {"prior": 0.01, "temp": 0.3, "offset": 0.003},
}

PAPER_CAPTION_GUIDANCE_BY_NAME = {
    "Medieval feast": 27.0,
    "A dinner table": 12.0,
    "An art workshop": 27.0,
    "An antique store": 21.0,
    "An English breakfast": 18.0,
}

DEFAULT_GUIDANCE_SCALES = "12,18,21,27"
DEFAULT_CFG_SCALE = 7.5


@dataclass(frozen=True)
class GenJob:
    prompt_id: int
    prompt_name: str
    positive: str
    negative: str
    neg_kind: str
    guidance_scale: float
    prior: float
    temp: float
    offset: float
    seeds: tuple[int, ...]
    out_paths: tuple[str, ...]


def load_prompts(path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(path.read_text())
    return payload["prompts"], int(payload.get("n_images_per_prompt", 32))


def image_path(
    out_dir: Path,
    prompt_id: int,
    neg_kind: str,
    guidance_scale: float,
    cfg_scale: float,
    seed: int,
) -> Path:
    if neg_kind == "baseline":
        subdir = (
            out_dir
            / "images"
            / f"prompt_{prompt_id:02d}"
            / "baseline"
            / f"cfg_{fmt_float(cfg_scale)}"
        )
    else:
        subdir = (
            out_dir
            / "images"
            / f"prompt_{prompt_id:02d}"
            / neg_kind
            / f"cfg_{fmt_float(cfg_scale)}_lambda0_{fmt_float(guidance_scale)}"
        )
    return subdir / f"seed_{seed:06d}.png"


def monitor_path(
    out_dir: Path,
    prompt_id: int,
    neg_kind: str,
    guidance_scale: float,
    cfg_scale: float,
    base_mode: str,
    scheduler_type: str,
    posterior_reference: str,
    seeds: tuple[int, ...],
) -> Path:
    seed_min = min(seeds)
    seed_max = max(seeds)
    return (
        out_dir
        / "monitors"
        / f"scheduler_{scheduler_type}"
        / f"base_{base_mode}_posterior_{posterior_reference}"
        / f"prompt_{prompt_id:02d}_{neg_kind}_cfg_{fmt_float(cfg_scale)}"
        f"_lambda0_{fmt_float(guidance_scale)}_seeds_{seed_min:06d}_{seed_max:06d}.csv"
    )


def chunks(values: list[int], size: int) -> list[tuple[int, ...]]:
    return [tuple(values[i : i + size]) for i in range(0, len(values), size)]


def guidance_scales_for_prompt(args: argparse.Namespace, name: str) -> list[float]:
    if args.paper_caption_guidance:
        return [PAPER_CAPTION_GUIDANCE_BY_NAME[name]]
    return args.guidance_scales


def hyperparams_for_prompt(name: str) -> dict[str, float]:
    try:
        return DNG_HYPERPARAMS_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(DNG_HYPERPARAMS_BY_NAME))
        raise KeyError(f"No fixed DNG hyperparameters for prompt name {name!r}. Known: {known}") from exc


def build_generation_jobs(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[GenJob]:
    seeds = [args.seed + i for i in range(args.n_images)]
    seed_batches = chunks(seeds, args.batch_size)
    jobs: list[GenJob] = []

    for entry in prompts:
        prompt_id = int(entry["id"])
        name = entry["name"]
        positive = entry["positive"]
        hp = hyperparams_for_prompt(name)

        for seed_batch in seed_batches:
            paths = tuple(
                str(image_path(args.out_dir, prompt_id, "baseline", 0.0, args.cfg_scale, seed))
                for seed in seed_batch
            )
            jobs.append(
                GenJob(
                    prompt_id=prompt_id,
                    prompt_name=name,
                    positive=positive,
                    negative="",
                    neg_kind="baseline",
                    guidance_scale=0.0,
                    prior=hp["prior"],
                    temp=hp["temp"],
                    offset=hp["offset"],
                    seeds=seed_batch,
                    out_paths=paths,
                )
            )

        for neg_kind in args.negative_kinds:
            neg_key = f"{neg_kind}_negative"
            for guidance_scale in guidance_scales_for_prompt(args, name):
                for seed_batch in seed_batches:
                    paths = tuple(
                        str(image_path(args.out_dir, prompt_id, neg_kind, guidance_scale, args.cfg_scale, seed))
                        for seed in seed_batch
                    )
                    jobs.append(
                        GenJob(
                            prompt_id=prompt_id,
                            prompt_name=name,
                            positive=positive,
                            negative=entry[neg_key],
                            neg_kind=neg_kind,
                            guidance_scale=guidance_scale,
                            prior=hp["prior"],
                            temp=hp["temp"],
                            offset=hp["offset"],
                            seeds=seed_batch,
                            out_paths=paths,
                        )
                    )
    return jobs


def load_sd_pipeline(device: str, args: argparse.Namespace):
    import torch
    from diffusers import DDPMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
    )
    # DNG's original posterior update is a Gaussian reverse-kernel likelihood
    # ratio.  DDPM is the closest exact match.  The Karras/Euler option is a
    # Stable-Diffusion-style sigma scheduler diagnostic; its posterior variance
    # below is an explicit surrogate because deterministic Euler steps do not
    # expose a true Gaussian reverse kernel unless churn/noise is modeled.
    if args.scheduler_type == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler_type == "euler-karras":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            timestep_spacing=args.timestep_spacing,
        )
    else:
        raise ValueError(f"Unknown scheduler_type={args.scheduler_type!r}")
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


def ddpm_reverse_mean_and_variance(scheduler, model_output, timestep: int, sample):
    """Return the DDPMScheduler reverse mean and scalar variance for a branch."""
    import torch

    t = int(timestep)
    prev_t = scheduler.previous_timestep(t)

    alpha_prod_t = scheduler.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t].to(device=sample.device, dtype=sample.dtype)
        if int(prev_t) >= 0
        else scheduler.one.to(device=sample.device, dtype=sample.dtype)
    )
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_x0 = model_output
    elif prediction_type == "v_prediction":
        pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")

    if scheduler.config.thresholding:
        pred_x0 = scheduler._threshold_sample(pred_x0)
    elif scheduler.config.clip_sample:
        pred_x0 = pred_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    pred_x0_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
    current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
    mean_prev = pred_x0_coeff * pred_x0 + current_sample_coeff * sample

    variance = scheduler._get_variance(t).to(device=sample.device, dtype=torch.float32).clamp_min(1e-20)
    return mean_prev, variance


def euler_karras_reverse_mean_and_variance(
    scheduler,
    model_output,
    step_index: int,
    sample,
    variance_floor: float,
    variance_scale: float,
):
    """Return an Euler/Karras deterministic next state and a surrogate variance.

    EulerDiscreteScheduler with Karras sigmas is usually an ODE-style sampler.
    With s_churn=0 the step is deterministic, so there is no literal Gaussian
    transition density for DNG's posterior update.  For diagnostics, we use the
    branch-specific Euler update as the mean and a local sigma-step variance as
    an explicit surrogate.  This keeps the sign of log q_B - log q_ref meaningful
    while avoiding the false precision of pretending this is the original DDPM
    posterior.
    """
    import torch

    sigma = scheduler.sigmas[step_index].to(device=sample.device, dtype=torch.float32)
    sigma_next = scheduler.sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)
    sample_f = sample.float()
    model_output_f = model_output.float()

    prediction_type = scheduler.config.prediction_type
    if prediction_type in ("sample", "original_sample"):
        pred_x0 = model_output_f
    elif prediction_type == "epsilon":
        pred_x0 = sample_f - sigma * model_output_f
    elif prediction_type == "v_prediction":
        pred_x0 = model_output_f * (-sigma / (sigma**2 + 1).sqrt()) + sample_f / (sigma**2 + 1)
    else:
        raise ValueError(f"Unsupported Euler/Karras prediction_type={prediction_type!r}")

    derivative = (sample_f - pred_x0) / sigma.clamp_min(1e-20)
    mean_prev = sample_f + derivative * (sigma_next - sigma)

    variance = ((sigma - sigma_next).abs() ** 2) * float(variance_scale)
    variance = variance.clamp_min(float(variance_floor)).to(device=sample.device, dtype=torch.float32)
    return mean_prev.to(dtype=model_output.dtype), variance


def reverse_mean_and_variance(
    scheduler,
    scheduler_type: str,
    model_output,
    timestep,
    sample,
    step_index: int,
    euler_variance_floor: float,
    euler_variance_scale: float,
):
    if scheduler_type == "ddpm":
        return ddpm_reverse_mean_and_variance(scheduler, model_output, int(timestep), sample)
    if scheduler_type == "euler-karras":
        return euler_karras_reverse_mean_and_variance(
            scheduler,
            model_output,
            step_index,
            sample,
            variance_floor=euler_variance_floor,
            variance_scale=euler_variance_scale,
        )
    raise ValueError(f"Unknown scheduler_type={scheduler_type!r}")


def compute_posterior_terms(
    scheduler,
    scheduler_type: str,
    timestep: int,
    step_index: int,
    logp,
    x_new,
    x_old,
    out_ref,
    out_neg,
    temp: float,
    offset: float,
    p_min: float,
    p_max: float,
    euler_variance_floor: float,
    euler_variance_scale: float,
):
    """Update log posterior and return per-sample diagnostic terms.

    kernel_log_ratio is log q_B(x_{t-1}|x_t) - log q_ref(x_{t-1}|x_t).
    Positive values mean the sampled transition looks more like the negative
    branch B than the chosen posterior reference branch.
    """
    import torch

    mu_ref, variance = reverse_mean_and_variance(
        scheduler,
        scheduler_type,
        out_ref,
        timestep,
        x_old,
        step_index,
        euler_variance_floor=euler_variance_floor,
        euler_variance_scale=euler_variance_scale,
    )
    mu_neg, _ = reverse_mean_and_variance(
        scheduler,
        scheduler_type,
        out_neg,
        timestep,
        x_old,
        step_index,
        euler_variance_floor=euler_variance_floor,
        euler_variance_scale=euler_variance_scale,
    )

    x_new_flat = x_new.float().flatten(1)
    mu_ref_flat = mu_ref.float().flatten(1)
    mu_neg_flat = mu_neg.float().flatten(1)

    dist_neg = ((x_new_flat - mu_neg_flat) ** 2).sum(dim=1)
    dist_ref = ((x_new_flat - mu_ref_flat) ** 2).sum(dim=1)
    dist_diff = dist_neg - dist_ref

    kernel_log_ratio = -0.5 / variance * dist_diff
    offset_term = 0.5 / variance * float(offset)
    increment = float(temp) * kernel_log_ratio + offset_term
    logp = logp + increment
    terms = {
        "variance": variance,
        "dist_diff": dist_diff,
        "kernel_log_ratio": kernel_log_ratio,
        "posterior_increment": increment,
    }
    return logp.clamp(min=math.log(p_min), max=math.log(p_max)), terms


def generate_batch(pipe, job: GenJob, args: argparse.Namespace, device: str):
    import torch

    batch_size = len(job.seeds)
    height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    pos_embeds = encode_text(pipe, [job.positive] * batch_size, device)
    uncond_embeds = encode_text(pipe, [""] * batch_size, device)
    neg_text = job.negative if job.neg_kind != "baseline" else ""
    neg_embeds = encode_text(pipe, [neg_text] * batch_size, device)
    prompt_embeds = torch.cat([uncond_embeds, pos_embeds, neg_embeds], dim=0)

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in job.seeds]
    latents = pipe.prepare_latents(
        batch_size,
        pipe.unet.config.in_channels,
        height,
        width,
        pos_embeds.dtype,
        device,
        generators,
        latents=None,
    )

    logp = torch.full(
        (batch_size,),
        math.log(job.prior),
        device=device,
        dtype=torch.float32,
    )
    monitor_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for step_index, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents, latents, latents], dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            model_out = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            out_uncond, out_pos, out_neg = model_out.chunk(3)
            cfg_baseline = out_uncond + args.cfg_scale * (out_pos - out_uncond)
            cfg_negative = out_uncond + args.cfg_scale * (out_neg - out_uncond)
            if args.base_mode == "cfg":
                out_base = cfg_baseline
            elif args.base_mode == "positive":
                out_base = out_pos
            else:
                raise ValueError(f"Unknown base mode {args.base_mode!r}")

            if args.negative_model_mode == "cfg":
                out_negative_model = cfg_negative
            elif args.negative_model_mode == "raw":
                out_negative_model = out_neg
            else:
                raise ValueError(
                    f"Unknown negative_model_mode={args.negative_model_mode!r}"
                )

            if args.posterior_reference == "generation":
                out_posterior_ref = out_base
                posterior_ref_label = args.base_mode
            elif args.posterior_reference == "uncond":
                out_posterior_ref = out_uncond
                posterior_ref_label = "uncond"
            else:
                raise ValueError(f"Unknown posterior_reference={args.posterior_reference!r}")

            if job.neg_kind == "baseline":
                guided_out = out_base
            else:
                p = logp.exp().to(dtype=latents.dtype)
                dynamic_scale = job.guidance_scale * p / (1.0 - p).clamp_min(args.posterior_eps)
                guided_out = out_base - dynamic_scale[:, None, None, None] * (
                    out_negative_model - out_base
                )

            step_timestep = int(t) if args.scheduler_type == "ddpm" else t
            scheduler_step_index = getattr(pipe.scheduler, "step_index", None)
            if scheduler_step_index is None:
                scheduler_step_index = step_index
            step_kwargs: dict[str, Any] = {
                "generator": generators,
                "return_dict": False,
            }
            if args.scheduler_type == "euler-karras":
                step_kwargs.update(
                    {
                        "s_churn": args.euler_s_churn,
                        "s_tmin": args.euler_s_tmin,
                        "s_tmax": args.euler_s_tmax,
                        "s_noise": args.euler_s_noise,
                    }
                )
            x_new = pipe.scheduler.step(guided_out, step_timestep, latents, **step_kwargs)[0]

            if job.neg_kind != "baseline":
                logp_before = logp
                logp, terms = compute_posterior_terms(
                    pipe.scheduler,
                    args.scheduler_type,
                    step_timestep,
                    int(scheduler_step_index),
                    logp,
                    x_new,
                    latents,
                    out_posterior_ref,
                    out_negative_model,
                    temp=job.temp,
                    offset=job.offset,
                    p_min=args.p_min,
                    p_max=args.p_max,
                    euler_variance_floor=args.euler_variance_floor,
                    euler_variance_scale=args.euler_variance_scale,
                )
                p_before = logp_before.exp().float()
                p_after = logp.exp().float()
                dynamic_scale_f = dynamic_scale.float()
                kernel_lr = terms["kernel_log_ratio"].float()
                posterior_increment = terms["posterior_increment"].float()
                dist_diff = terms["dist_diff"].float()
                monitor_rows.append(
                    {
                        "base_mode": args.base_mode,
                        "negative_model_mode": args.negative_model_mode,
                        "scheduler_type": args.scheduler_type,
                        "posterior_reference": args.posterior_reference,
                        "posterior_ref_label": posterior_ref_label,
                        "prompt_id": job.prompt_id,
                        "prompt_name": job.prompt_name,
                        "neg_kind": job.neg_kind,
                        "positive": job.positive,
                        "negative": job.negative,
                        "guidance_scale": job.guidance_scale,
                        "cfg_scale": args.cfg_scale,
                        "prior": job.prior,
                        "temp": job.temp,
                        "offset": job.offset,
                        "batch_seed_min": min(job.seeds),
                        "batch_seed_max": max(job.seeds),
                        "n": batch_size,
                        "step_index": step_index,
                        "timestep": int(t),
                        "variance": float(terms["variance"].detach().float().cpu()),
                        "p_before_mean": float(p_before.mean().cpu()),
                        "p_before_std": float(p_before.std(unbiased=False).cpu()),
                        "p_after_mean": float(p_after.mean().cpu()),
                        "p_after_std": float(p_after.std(unbiased=False).cpu()),
                        "logp_before_mean": float(logp_before.float().mean().cpu()),
                        "logp_after_mean": float(logp.float().mean().cpu()),
                        "guidance_strength_mean": float(dynamic_scale_f.mean().cpu()),
                        "guidance_strength_std": float(dynamic_scale_f.std(unbiased=False).cpu()),
                        "guidance_strength_max": float(dynamic_scale_f.max().cpu()),
                        "kernel_log_ratio_mean": float(kernel_lr.mean().cpu()),
                        "kernel_log_ratio_std": float(kernel_lr.std(unbiased=False).cpu()),
                        "posterior_increment_mean": float(posterior_increment.mean().cpu()),
                        "posterior_increment_std": float(posterior_increment.std(unbiased=False).cpu()),
                        "dist_neg_minus_ref_mean": float(dist_diff.mean().cpu()),
                        "dist_neg_minus_base_mean": float(dist_diff.mean().cpu()),
                    }
                )
            latents = x_new

        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type="pil", do_denormalize=[True] * batch_size)

    for img, out_path in zip(images, job.out_paths):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)

    if monitor_rows:
        write_csv(
            monitor_path(
                args.out_dir,
                job.prompt_id,
                job.neg_kind,
                job.guidance_scale,
                args.cfg_scale,
                args.base_mode,
                args.scheduler_type,
                args.posterior_reference,
                job.seeds,
            ),
            monitor_rows,
        )


def worker_main(worker_idx: int, device: str, jobs: list[dict[str, Any]], args_dict: dict[str, Any]) -> None:
    args = argparse.Namespace(**args_dict)
    args.out_dir = Path(args.out_dir)
    if args.stagger_load_seconds > 0:
        time.sleep(worker_idx * args.stagger_load_seconds)

    print(f"[worker {worker_idx}] loading SD+DNG on {device}; jobs={len(jobs)}", flush=True)
    pipe = load_sd_pipeline(device, args)
    print(f"[worker {worker_idx}] ready on {device}", flush=True)

    for j, job_dict in enumerate(jobs, 1):
        job = GenJob(**job_dict)
        if args.skip_existing and all(Path(p).exists() for p in job.out_paths):
            continue
        generate_batch(pipe, job, args, device)
        if j % args.log_every == 0 or j == len(jobs):
            print(f"[worker {worker_idx}] {j}/{len(jobs)} batches done", flush=True)


def split_jobs(jobs: list[GenJob], n_workers: int) -> list[list[GenJob]]:
    buckets: list[list[GenJob]] = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        buckets[i % n_workers].append(job)
    return buckets


def run_generation(args: argparse.Namespace, jobs: list[GenJob]) -> None:
    import torch.multiprocessing as mp

    buckets = split_jobs(jobs, len(args.devices))
    args_dict = vars(args).copy()
    args_dict["out_dir"] = str(args.out_dir)

    ctx = mp.get_context("spawn")
    procs = []
    for worker_idx, (device, bucket) in enumerate(zip(args.devices, buckets)):
        proc = ctx.Process(target=worker_main, args=(worker_idx, device, [asdict(j) for j in bucket], args_dict))
        proc.start()
        procs.append(proc)

    failures = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures.append(proc.exitcode)
    if failures:
        raise RuntimeError(f"{len(failures)} generation worker(s) failed: {failures}")


def expected_eval_rows(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = [args.seed + i for i in range(args.n_images)]
    for entry in prompts:
        prompt_id = int(entry["id"])
        name = entry["name"]
        hp = hyperparams_for_prompt(name)
        for neg_kind in args.negative_kinds:
            neg_key = f"{neg_kind}_negative"
            for guidance_scale in guidance_scales_for_prompt(args, name):
                for seed in seeds:
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_name": name,
                            "positive": entry["positive"],
                            "negative": entry[neg_key],
                            "neg_kind": neg_kind,
                            "guidance_scale": guidance_scale,
                            "cfg_scale": args.cfg_scale,
                            "base_mode": args.base_mode,
                            "negative_model_mode": args.negative_model_mode,
                            "scheduler_type": args.scheduler_type,
                            "posterior_reference": args.posterior_reference,
                            "dng_variant": (
                                f"{args.scheduler_type}_{args.base_mode}_base_"
                                f"negative_{args.negative_model_mode}_"
                                f"posterior_{args.posterior_reference}"
                            ),
                            "prior": hp["prior"],
                            "temp": hp["temp"],
                            "offset": hp["offset"],
                            "seed": seed,
                            "image_path": str(
                                image_path(args.out_dir, prompt_id, neg_kind, guidance_scale, args.cfg_scale, seed)
                            ),
                            "baseline_path": str(
                                image_path(args.out_dir, prompt_id, "baseline", 0.0, args.cfg_scale, seed)
                            ),
                        }
                    )
    return rows


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("prompt_id", "seed"):
            row[key] = int(row[key])
        for key in (
            "guidance_scale",
            "cfg_scale",
            "prior",
            "temp",
            "offset",
            "clip_pos",
            "clip_neg",
            "clip_img_to_baseline",
            "baseline_clip_pos",
            "baseline_clip_neg",
        ):
            row[key] = float(row[key])
    return rows


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out: list[dict[str, Any]] = []
    for key_vals, group in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key_vals)}
        for metric in ("clip_pos", "clip_neg", "clip_img_to_baseline"):
            vals = [float(g[metric]) for g in group]
            item[f"{metric}_mean"] = mean(vals)
            item[f"{metric}_sem"] = sem(vals)
        item["n"] = len(group)
        out.append(item)
    return out


def write_margin_ranking(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    keys = (
        "prompt_id",
        "prompt_name",
        "neg_kind",
        "guidance_scale",
        "cfg_scale",
        "base_mode",
        "negative_model_mode",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key_vals = []
        for key in keys:
            if key in row:
                key_vals.append(row[key])
            elif key == "base_mode":
                key_vals.append(args.base_mode)
            elif key == "negative_model_mode":
                key_vals.append(args.negative_model_mode)
            else:
                raise KeyError(key)
        groups.setdefault(tuple(key_vals), []).append(row)

    ranking: list[dict[str, Any]] = []
    for key_vals, group in groups.items():
        margins = [float(row["clip_pos"]) - float(row["clip_neg"]) for row in group]
        baseline_margins = [
            float(row["baseline_clip_pos"]) - float(row["baseline_clip_neg"])
            for row in group
        ]
        item = {key: value for key, value in zip(keys, key_vals)}
        item.update(
            {
                "clip_margin_mean": mean(margins),
                "clip_margin_sem": sem(margins),
                "clip_pos_mean": mean([float(row["clip_pos"]) for row in group]),
                "clip_neg_mean": mean([float(row["clip_neg"]) for row in group]),
                "baseline_clip_margin_mean": mean(baseline_margins),
                "margin_gain_over_baseline": mean(margins) - mean(baseline_margins),
                "clip_img_to_baseline_mean": mean(
                    [float(row["clip_img_to_baseline"]) for row in group]
                ),
                "n": len(group),
            }
        )
        ranking.append(item)

    ranking.sort(key=lambda row: float(row["clip_margin_mean"]), reverse=True)
    metrics_dir = args.out_dir / "metrics"
    ranking_path = metrics_dir / "clip_margin_ranking.csv"
    best_path = metrics_dir / "best_clip_margin.json"
    write_csv(ranking_path, ranking)
    best_path.write_text(json.dumps(ranking[0], indent=2) + "\n")
    print(f"[margin] wrote {ranking_path}", flush=True)
    print(f"[margin] wrote {best_path}", flush=True)
    print(
        "[margin] best: "
        f"prompt={ranking[0]['prompt_id']} "
        f"neg_kind={ranking[0]['neg_kind']} "
        f"lambda0={float(ranking[0]['guidance_scale']):g} "
        f"mean={float(ranking[0]['clip_margin_mean']):.6f} "
        f"+/- {float(ranking[0]['clip_margin_sem']):.6f} SEM",
        flush=True,
    )


def baseline_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    dedup: list[dict[str, Any]] = []
    for row in rows:
        key = (row["prompt_id"], row["neg_kind"], row["seed"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in dedup:
        groups.setdefault(row["neg_kind"], []).append(row)

    out = []
    for neg_kind, group in sorted(groups.items()):
        out.append(
            {
                "neg_kind": neg_kind,
                "clip_pos_mean": mean([float(g["baseline_clip_pos"]) for g in group]),
                "clip_neg_mean": mean([float(g["baseline_clip_neg"]) for g in group]),
            }
        )
    return out


def parse_prompt_ids(text: str) -> set[int] | None:
    text = text.strip()
    if not text:
        return None
    return {int(part) for part in text.split(",") if part.strip()}


def select_prompts(prompts: list[dict[str, Any]], prompt_ids: set[int] | None) -> list[dict[str, Any]]:
    if prompt_ids is None:
        return prompts
    selected = [p for p in prompts if int(p["id"]) in prompt_ids]
    missing = sorted(prompt_ids - {int(p["id"]) for p in selected})
    if missing:
        raise ValueError(f"Requested prompt ids not found: {missing}")
    return selected


def plot_one_summary(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    combo_rows = aggregate(rows, ("neg_kind", "guidance_scale"))
    base_rows = baseline_points(rows)
    scales = sorted({float(r["guidance_scale"]) for r in rows})
    colors = plt.cm.plasma([i / max(1, len(scales) - 1) for i in range(len(scales))])
    color_by_scale = {scale: colors[i] for i, scale in enumerate(scales)}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, neg_kind, panel_title in (
        (axes[0], "related", "Related negative prompt"),
        (axes[1], "unrelated", "Unrelated negative prompt"),
    ):
        for item in [r for r in combo_rows if r["neg_kind"] == neg_kind]:
            scale = float(item["guidance_scale"])
            ax.errorbar(
                item["clip_pos_mean"],
                item["clip_neg_mean"],
                xerr=item["clip_pos_sem"],
                yerr=item["clip_neg_sem"],
                marker="o",
                color=color_by_scale[scale],
                linestyle="None",
                markersize=7,
                alpha=0.9,
                label=f"lambda0={scale:g}",
            )

        for item in [b for b in base_rows if b["neg_kind"] == neg_kind]:
            ax.scatter(
                item["clip_pos_mean"],
                item["clip_neg_mean"],
                marker="*",
                s=130,
                color="black",
                alpha=0.75,
                label="baseline",
            )

        ax.set_title(panel_title)
        ax.set_xlabel("CLIP(image, positive prompt)")
        ax.set_ylabel(f"CLIP(image, {neg_kind} negative prompt)")
        ax.grid(True, alpha=0.25)

    preserve_rows = aggregate(rows, ("neg_kind", "guidance_scale"))
    for neg_kind, linestyle in (("related", "-"), ("unrelated", "--")):
        items = [r for r in preserve_rows if r["neg_kind"] == neg_kind]
        items = sorted(items, key=lambda r: float(r["guidance_scale"]))
        xs = [float(r["guidance_scale"]) for r in items]
        ys = [float(r["clip_img_to_baseline_mean"]) for r in items]
        yerr = [float(r["clip_img_to_baseline_sem"]) for r in items]
        axes[2].errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linestyle=linestyle,
            linewidth=1.5,
            markersize=6,
            label=neg_kind,
        )

    axes[2].axhline(1.0, color="black", linewidth=1, alpha=0.4)
    axes[2].set_title("Guidance strength vs preservation")
    axes[2].set_xlabel("DNG initial guidance scale lambda0")
    axes[2].set_ylabel("CLIP(DNG image, baseline image)")
    axes[2].set_ylim(bottom=min(0.0, axes[2].get_ylim()[0]), top=1.02)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    handles = [plt.Line2D([0], [0], color=color_by_scale[s], marker="o", linestyle="None") for s in scales]
    labels = [f"lambda0={s:g}" for s in scales]
    handles.append(plt.Line2D([0], [0], color="black", marker="*", linestyle="None"))
    labels.append("baseline")
    axes[1].legend(handles, labels, fontsize=7, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def read_monitor_rows(args: argparse.Namespace, prompt_id: int) -> list[dict[str, Any]]:
    monitor_dir = (
        args.out_dir
        / "monitors"
        / f"scheduler_{args.scheduler_type}"
        / f"base_{args.base_mode}_posterior_{args.posterior_reference}"
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(monitor_dir.glob(f"prompt_{prompt_id:02d}_*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["source_csv"] = str(path)
                rows.append(row)
    numeric = {
        "prompt_id",
        "guidance_scale",
        "cfg_scale",
        "prior",
        "temp",
        "offset",
        "batch_seed_min",
        "batch_seed_max",
        "n",
        "step_index",
        "timestep",
        "variance",
        "p_before_mean",
        "p_before_std",
        "p_after_mean",
        "p_after_std",
        "logp_before_mean",
        "logp_after_mean",
        "guidance_strength_mean",
        "guidance_strength_std",
        "guidance_strength_max",
        "kernel_log_ratio_mean",
        "kernel_log_ratio_std",
        "posterior_increment_mean",
        "posterior_increment_std",
        "dist_neg_minus_ref_mean",
        "dist_neg_minus_base_mean",
    }
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = float(row[key])
        row["prompt_id"] = int(row["prompt_id"])
        row["step_index"] = int(row["step_index"])
        row["timestep"] = int(row["timestep"])
    return rows


def aggregate_monitor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("neg_kind", "guidance_scale", "step_index", "timestep")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    metrics = (
        "guidance_strength_mean",
        "guidance_strength_max",
        "kernel_log_ratio_mean",
        "posterior_increment_mean",
        "p_before_mean",
        "p_after_mean",
        "logp_before_mean",
        "logp_after_mean",
    )
    out: list[dict[str, Any]] = []
    for key_vals, group in sorted(groups.items(), key=lambda kv: (kv[0][0], float(kv[0][1]), int(kv[0][2]))):
        item = {k: v for k, v in zip(keys, key_vals)}
        for metric in metrics:
            vals = [float(g[metric]) for g in group]
            item[f"{metric}_avg"] = mean(vals)
            item[f"{metric}_sem"] = sem(vals)
        item["n_batches"] = len(group)
        out.append(item)
    return out


def plot_monitor_summary(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    plot_rows = [r for r in rows if float(r.get("variance", 0.0)) > 1e-12]
    if not plot_rows:
        return
    agg = aggregate_monitor_rows(plot_rows)
    scales = sorted({float(r["guidance_scale"]) for r in agg})
    colors = plt.cm.viridis([i / max(1, len(scales) - 1) for i in range(len(scales))])
    color_by_scale = {scale: colors[i] for i, scale in enumerate(scales)}
    line_by_kind = {"related": "-", "unrelated": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(title + " (variance-degenerate final rows omitted)", fontsize=13)

    panels = [
        ("guidance_strength_mean_avg", "Dynamic guidance strength", r"$\lambda_0 p_t/(1-p_t)$"),
        ("kernel_log_ratio_mean_avg", "Reverse-kernel log ratio", r"$\log q_B-\log q_{base}$"),
        ("p_before_mean_avg", "Posterior before step", r"$p_t(B)$"),
    ]
    for ax, (metric, panel_title, ylabel) in zip(axes, panels):
        for neg_kind in sorted({str(r["neg_kind"]) for r in agg}):
            for scale in scales:
                items = [
                    r
                    for r in agg
                    if str(r["neg_kind"]) == neg_kind and float(r["guidance_scale"]) == scale
                ]
                if not items:
                    continue
                items = sorted(items, key=lambda r: int(r["step_index"]))
                xs = [int(r["step_index"]) for r in items]
                ys = [float(r[metric]) for r in items]
                ax.plot(
                    xs,
                    ys,
                    linestyle=line_by_kind.get(neg_kind, "-"),
                    color=color_by_scale[scale],
                    linewidth=1.8,
                    label=f"{neg_kind}, lambda0={scale:g}",
                )
        ax.set_title(panel_title)
        ax.set_xlabel("reverse step index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.35)
    axes[0].legend(fontsize=7, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plots(args: argparse.Namespace, prompts: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    fig_dir = args.out_dir / "figures"
    for entry in prompts:
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == int(entry["id"])]
        hp = hyperparams_for_prompt(entry["name"])
        out_path = fig_dir / f"prompt_{int(entry['id']):02d}_{slug(entry['name'])}.png"
        title = (
            f"Prompt {int(entry['id'])}: {entry['name']} "
            f"(base={args.base_mode}, cfg={args.cfg_scale:g}, "
            f"p={hp['prior']}, tau={hp['temp']}, delta={hp['offset']})"
        )
        plot_one_summary(prompt_rows, out_path, title)
        print(f"[plot] {out_path}", flush=True)
        monitor_rows = read_monitor_rows(args, int(entry["id"]))
        monitor_out = fig_dir / f"prompt_{int(entry['id']):02d}_{slug(entry['name'])}_monitor.png"
        plot_monitor_summary(monitor_rows, monitor_out, title + " monitor")
        if monitor_rows:
            print(f"[plot] {monitor_out}", flush=True)

    if len(prompts) > 1:
        plot_one_summary(
            rows,
            fig_dir / "average.png",
            f"Average across {len(prompts)} prompts (base={args.base_mode}, cfg={args.cfg_scale:g})",
        )
        print(f"[plot] {fig_dir / 'average.png'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE)
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default="",
        help="Comma-separated prompt ids to run. Empty means all prompts.",
    )
    parser.add_argument("--out-dir", type=Path, default=TEXT2IMAGE_DIR / "dng_eval/dng_cfg_self_consistent_eval")
    parser.add_argument("--model-id", type=str, default=SD_MODEL_ID)
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated generation devices.")
    parser.add_argument("--score-device", type=str, default=None)
    parser.add_argument(
        "--base-mode",
        choices=("cfg", "positive"),
        default="cfg",
        help="Base branch for DNG: positive CFG baseline or raw positive-prompt prediction.",
    )
    parser.add_argument(
        "--negative-model-mode",
        choices=("raw", "cfg"),
        default="raw",
        help=(
            "Model used for the negative branch B. raw preserves the legacy "
            "implementation. cfg uses eps_u + cfg_scale*(eps_B-eps_u), so with "
            "--base-mode cfg both A and B are CFG-induced conditional models."
        ),
    )
    parser.add_argument(
        "--negative-kinds",
        type=parse_negative_kinds,
        default=parse_negative_kinds("related,unrelated"),
        help="Comma-separated negative prompt kinds to generate: related,unrelated.",
    )
    parser.add_argument(
        "--scheduler-type",
        choices=("ddpm", "euler-karras"),
        default="ddpm",
        help=(
            "Reverse sampler. ddpm uses a true DDPM Gaussian kernel; "
            "euler-karras uses EulerDiscreteScheduler with Karras sigmas and "
            "a surrogate sigma-step posterior variance."
        ),
    )
    parser.add_argument(
        "--posterior-reference",
        choices=("generation", "uncond"),
        default="generation",
        help=(
            "Reference branch for the DNG posterior likelihood. generation compares "
            "p^B to the branch used for generation (p^A if --base-mode positive); "
            "uncond compares p^B to p_uncond while keeping generation guidance unchanged."
        ),
    )
    parser.add_argument(
        "--timestep-spacing",
        choices=("linspace", "leading", "trailing"),
        default="linspace",
        help="Timestep spacing used by the Euler/Karras scheduler.",
    )
    parser.add_argument(
        "--euler-variance-floor",
        type=float,
        default=1e-4,
        help="Minimum surrogate variance for the Euler/Karras posterior monitor.",
    )
    parser.add_argument(
        "--euler-variance-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for the Euler/Karras sigma-step surrogate variance.",
    )
    parser.add_argument("--euler-s-churn", type=float, default=0.0)
    parser.add_argument("--euler-s-tmin", type=float, default=0.0)
    parser.add_argument("--euler-s-tmax", type=float, default=float("inf"))
    parser.add_argument("--euler-s-noise", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=None, help="Default uses the model native size.")
    parser.add_argument("--width", type=int, default=None, help="Default uses the model native size.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--guidance-scales", type=parse_float_list, default=parse_float_list(DEFAULT_GUIDANCE_SCALES))
    parser.add_argument(
        "--paper-caption-guidance",
        action="store_true",
        help="Use one prompt-specific lambda0 from the paper captions instead of sweeping --guidance-scales.",
    )
    parser.add_argument("--p-min", type=float, default=1e-6)
    parser.add_argument("--p-max", type=float, default=0.99)
    parser.add_argument("--posterior-eps", type=float, default=1e-6)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stagger-load-seconds", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.devices is None:
        args.devices = detect_devices()
    else:
        args.devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not args.devices:
        raise SystemExit("No generation devices were specified.")
    if args.score_device is None:
        args.score_device = args.devices[0]
    if args.force_generation:
        args.skip_existing = False
    if args.negative_model_mode == "cfg" and args.base_mode != "cfg":
        raise SystemExit("--negative-model-mode cfg requires --base-mode cfg")
    return args


def main() -> None:
    args = parse_args()
    prompts, default_n_images = load_prompts(args.prompts_file)
    prompts = select_prompts(prompts, parse_prompt_ids(args.prompt_ids))
    if args.n_images is None:
        args.n_images = default_n_images

    jobs = build_generation_jobs(args, prompts)
    scales_desc = "paper-caption-per-prompt" if args.paper_caption_guidance else str(args.guidance_scales)
    n_scales_total = sum(len(guidance_scales_for_prompt(args, entry["name"])) for entry in prompts)
    n_guided = len(args.negative_kinds) * args.n_images * n_scales_total
    n_baseline = len(prompts) * args.n_images
    print(
        f"[config] prompts={len(prompts)} images/condition={args.n_images} "
        f"negative_kinds={args.negative_kinds} guidance_scales={scales_desc}",
        flush=True,
    )
    print(f"[config] model_id={args.model_id}", flush=True)
    print("[config] fixed prompt hyperparameters:", flush=True)
    for entry in prompts:
        hp = hyperparams_for_prompt(entry["name"])
        print(
            f"  - {entry['name']}: p={hp['prior']}, tau={hp['temp']}, delta={hp['offset']}",
            flush=True,
        )
    print(
        f"[config] generation images: guided={n_guided}, baseline={n_baseline}, total={n_guided + n_baseline}",
        flush=True,
    )
    print(
        f"[config] variant={args.scheduler_type}_{args.base_mode}_base_"
        f"negative_{args.negative_model_mode}_posterior_{args.posterior_reference}; "
        f"cfg_scale={args.cfg_scale}; "
        f"devices={args.devices}; score_device={args.score_device}",
        flush=True,
    )
    if args.scheduler_type == "euler-karras":
        print(
            f"[config] euler_karras: timestep_spacing={args.timestep_spacing}; "
            f"variance_floor={args.euler_variance_floor}; "
            f"variance_scale={args.euler_variance_scale}; "
            f"s_churn={args.euler_s_churn}",
            flush=True,
        )
    print(f"[config] out_dir={args.out_dir}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "prompts_file": str(args.prompts_file),
        "prompts": prompts,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    if args.dry_run:
        return

    if not args.skip_generation:
        run_generation(args, jobs)

    metrics_path = args.out_dir / "metrics" / "clip_scores.csv"
    if args.skip_scoring:
        if not metrics_path.exists():
            raise FileNotFoundError(f"--skip-scoring was set but {metrics_path} does not exist")
        rows = read_metrics_csv(metrics_path)
    elif metrics_path.exists() and not args.force_score:
        print(f"[clip] using existing metrics {metrics_path}", flush=True)
        rows = read_metrics_csv(metrics_path)
    else:
        eval_rows = expected_eval_rows(args, prompts)
        rows = score_with_clip(args, eval_rows)
        write_csv(metrics_path, rows)
        print(f"[clip] wrote {metrics_path}", flush=True)

    write_margin_ranking(args, rows)

    if not args.skip_plots:
        make_plots(args, prompts, rows)


if __name__ == "__main__":
    main()
