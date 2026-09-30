#!/usr/bin/env python3
"""Run a two-term negative-prompt guidance grid and plot CLIP diagnostics.

This script evaluates the five DNG Stable Diffusion prompts in dng_prompts.json.
For each positive prompt A and negative prompt B, it samples with

    eps = eps_uncond
          + lambda1 * (eps_A - eps_uncond)
          + lambda2 * (eps_B - eps_uncond)

where lambda2 should be negative to repel the B concept.

Default grid:
  lambda1 = 5.0, 7.5, 10.0
      Around the Diffusers CFG default of 7.5.
  lambda2 = -1.7, -2.55, -3.4
      Sign-flipped from the DNG paper Stable Diffusion NP scales.

Example on 8 GPUs:
  python run_np_grid_eval.py --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

Outputs:
  dng_eval/np_grid_eval/images/
  dng_eval/np_grid_eval/metrics/clip_scores.csv
  dng_eval/np_grid_eval/figures/prompt_*.png
  dng_eval/np_grid_eval/figures/average.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
PROMPTS_FILE = SCRIPT_DIR.parent / "dng_prompts.json"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

SD_MODEL_ID = "sd2-community/stable-diffusion-2-1"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

DEFAULT_LAMBDA1 = "5.0,7.5,10.0"
DEFAULT_LAMBDA2 = "-1.7,-2.55,-3.4"


@dataclass(frozen=True)
class GenJob:
    prompt_id: int
    prompt_name: str
    positive: str
    negative: str
    neg_kind: str
    lambda1: float
    lambda2: float
    seeds: tuple[int, ...]
    out_paths: tuple[str, ...]


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    return vals


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return vals


def parse_negative_kinds(text: str) -> list[str]:
    vals = [x.strip().lower() for x in text.split(",") if x.strip()]
    allowed = {"related", "unrelated"}
    invalid = sorted(set(vals) - allowed)
    if not vals or invalid:
        expected = ",".join(sorted(allowed))
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {expected}; invalid={invalid}"
        )
    return list(dict.fromkeys(vals))


def fmt_float(value: float) -> str:
    sign = "m" if value < 0 else ""
    value = abs(value)
    text = f"{value:g}".replace(".", "p")
    return f"{sign}{text}"


def slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "prompt"


def chunks(values: list[int], size: int) -> list[tuple[int, ...]]:
    return [tuple(values[i : i + size]) for i in range(0, len(values), size)]


def image_path(
    out_dir: Path,
    prompt_id: int,
    neg_kind: str,
    lambda1: float,
    lambda2: float,
    seed: int,
) -> Path:
    if neg_kind == "baseline":
        subdir = out_dir / "images" / f"prompt_{prompt_id:02d}" / "baseline" / f"l1_{fmt_float(lambda1)}"
    else:
        combo = f"l1_{fmt_float(lambda1)}_l2_{fmt_float(lambda2)}"
        subdir = out_dir / "images" / f"prompt_{prompt_id:02d}" / neg_kind / combo
    return subdir / f"seed_{seed:06d}.png"


def load_prompts(path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(path.read_text())
    return payload["prompts"], int(payload.get("n_images_per_prompt", 32))


def build_generation_jobs(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[GenJob]:
    seeds = [args.seed + i for i in range(args.n_images)]
    seed_batches = chunks(seeds, args.batch_size)
    jobs: list[GenJob] = []

    for entry in prompts:
        prompt_id = int(entry["id"])
        positive = entry["positive"]
        name = entry["name"]

        for lambda1 in args.lambda1:
            for seed_batch in seed_batches:
                paths = tuple(
                    str(image_path(args.out_dir, prompt_id, "baseline", lambda1, 0.0, seed))
                    for seed in seed_batch
                )
                jobs.append(
                    GenJob(
                        prompt_id=prompt_id,
                        prompt_name=name,
                        positive=positive,
                        negative="",
                        neg_kind="baseline",
                        lambda1=lambda1,
                        lambda2=0.0,
                        seeds=seed_batch,
                        out_paths=paths,
                    )
                )

        for neg_kind in args.negative_kinds:
            neg_key = f"{neg_kind}_negative"
            negative = entry[neg_key]
            for lambda1 in args.lambda1:
                for lambda2 in args.lambda2:
                    for seed_batch in seed_batches:
                        paths = tuple(
                            str(image_path(args.out_dir, prompt_id, neg_kind, lambda1, lambda2, seed))
                            for seed in seed_batch
                        )
                        jobs.append(
                            GenJob(
                                prompt_id=prompt_id,
                                prompt_name=name,
                                positive=positive,
                                negative=negative,
                                neg_kind=neg_kind,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                seeds=seed_batch,
                                out_paths=paths,
                            )
                        )

    return jobs


def load_sd_pipeline(device: str, args: argparse.Namespace):
    import torch
    from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
    )
    if args.scheduler == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "dpm_solver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
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
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generators, eta=0.0)

    with torch.inference_mode():
        for t in timesteps:
            latent_model_input = torch.cat([latents, latents, latents], dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            eps_uncond, eps_pos, eps_neg = noise_pred.chunk(3)
            noise_pred = (
                eps_uncond
                + job.lambda1 * (eps_pos - eps_uncond)
                + job.lambda2 * (eps_neg - eps_uncond)
            )
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type="pil", do_denormalize=[True] * batch_size)

    for img, out_path in zip(images, job.out_paths):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)


def worker_main(worker_idx: int, device: str, jobs: list[dict[str, Any]], args_dict: dict[str, Any]) -> None:
    args = argparse.Namespace(**args_dict)
    if args.stagger_load_seconds > 0:
        time.sleep(worker_idx * args.stagger_load_seconds)

    print(f"[worker {worker_idx}] loading SD on {device}; jobs={len(jobs)}", flush=True)
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

    devices = args.devices
    buckets = split_jobs(jobs, len(devices))
    args_dict = vars(args).copy()
    args_dict["out_dir"] = str(args.out_dir)

    ctx = mp.get_context("spawn")
    procs = []
    for worker_idx, (device, bucket) in enumerate(zip(devices, buckets)):
        job_dicts = [asdict(job) for job in bucket]
        proc = ctx.Process(target=worker_main, args=(worker_idx, device, job_dicts, args_dict))
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
        for neg_kind in args.negative_kinds:
            neg_key = f"{neg_kind}_negative"
            for lambda1 in args.lambda1:
                for lambda2 in args.lambda2:
                    for seed in seeds:
                        rows.append(
                            {
                                "prompt_id": prompt_id,
                                "prompt_name": entry["name"],
                                "positive": entry["positive"],
                                "negative": entry[neg_key],
                                "neg_kind": neg_kind,
                                "lambda1": lambda1,
                                "lambda2": lambda2,
                                "seed": seed,
                                "image_path": str(image_path(args.out_dir, prompt_id, neg_kind, lambda1, lambda2, seed)),
                                "baseline_path": str(image_path(args.out_dir, prompt_id, "baseline", lambda1, 0.0, seed)),
                            }
                        )
    return rows


def normalize_features(x):
    import torch

    return x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)


def score_with_clip(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    missing = sorted(
        {
            p
            for row in rows
            for p in (row["image_path"], row["baseline_path"])
            if not Path(p).exists()
        }
    )
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} expected images are missing. First paths:\n{preview}")

    device = args.score_device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[clip] loading {CLIP_MODEL_ID} on {device}", flush=True)
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_ID,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
    )
    model.eval()

    image_paths = sorted({p for row in rows for p in (row["image_path"], row["baseline_path"])})
    texts = sorted({row["positive"] for row in rows} | {row["negative"] for row in rows})

    image_features: dict[str, Any] = {}
    for start in range(0, len(image_paths), args.clip_batch_size):
        batch_paths = image_paths[start : start + args.clip_batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            feats = normalize_features(model.get_image_features(**inputs))
        for path, feat in zip(batch_paths, feats.detach().cpu()):
            image_features[path] = feat.float()
        if (start // args.clip_batch_size + 1) % 20 == 0:
            print(f"[clip] image features {min(start + args.clip_batch_size, len(image_paths))}/{len(image_paths)}", flush=True)

    text_features: dict[str, Any] = {}
    for start in range(0, len(texts), args.text_batch_size):
        batch_texts = texts[start : start + args.text_batch_size]
        inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = normalize_features(model.get_text_features(**inputs))
        for text, feat in zip(batch_texts, feats.detach().cpu()):
            text_features[text] = feat.float()

    scored: list[dict[str, Any]] = []
    for row in rows:
        image_feat = image_features[row["image_path"]]
        base_feat = image_features[row["baseline_path"]]
        pos_feat = text_features[row["positive"]]
        neg_feat = text_features[row["negative"]]
        out = dict(row)
        out["clip_pos"] = float(image_feat @ pos_feat)
        out["clip_neg"] = float(image_feat @ neg_feat)
        out["clip_img_to_baseline"] = float(image_feat @ base_feat)
        out["baseline_clip_pos"] = float(base_feat @ pos_feat)
        out["baseline_clip_neg"] = float(base_feat @ neg_feat)
        scored.append(out)

    return scored


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("prompt_id", "seed"):
            row[key] = int(row[key])
        for key in (
            "lambda1",
            "lambda2",
            "clip_pos",
            "clip_neg",
            "clip_img_to_baseline",
            "baseline_clip_pos",
            "baseline_clip_neg",
        ):
            row[key] = float(row[key])
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


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
    keys = ("prompt_id", "prompt_name", "neg_kind", "lambda1", "lambda2")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

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
    print(
        "[margin] best: "
        f"lambda1={float(ranking[0]['lambda1']):g}, "
        f"lambda2={float(ranking[0]['lambda2']):g}, "
        f"mean={float(ranking[0]['clip_margin_mean']):.6f} "
        f"+/- {float(ranking[0]['clip_margin_sem']):.6f} SEM",
        flush=True,
    )


def baseline_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    dedup: list[dict[str, Any]] = []
    for row in rows:
        key = (row["prompt_id"], row["neg_kind"], row["lambda1"], row["seed"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in dedup:
        groups.setdefault((row["neg_kind"], row["lambda1"]), []).append(row)

    out = []
    for (neg_kind, lambda1), group in sorted(groups.items()):
        out.append(
            {
                "neg_kind": neg_kind,
                "lambda1": lambda1,
                "clip_pos_mean": mean([float(g["baseline_clip_pos"]) for g in group]),
                "clip_neg_mean": mean([float(g["baseline_clip_neg"]) for g in group]),
            }
        )
    return out


def plot_one_summary(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    lambda1_values: list[float],
    lambda2_values: list[float],
) -> None:
    import matplotlib.pyplot as plt

    related = [r for r in rows if r["neg_kind"] == "related"]
    unrelated = [r for r in rows if r["neg_kind"] == "unrelated"]
    combo_rows = aggregate(rows, ("neg_kind", "lambda1", "lambda2"))
    base_rows = baseline_points(rows)

    colors = plt.cm.viridis([i / max(1, len(lambda2_values) - 1) for i in range(len(lambda2_values))])
    color_by_l2 = {l2: colors[i] for i, l2 in enumerate(sorted(lambda2_values))}
    markers = ["o", "s", "^", "D", "P", "X"]
    marker_by_l1 = {l1: markers[i % len(markers)] for i, l1 in enumerate(sorted(lambda1_values))}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, neg_kind, panel_title in (
        (axes[0], "related", "Related negative prompt"),
        (axes[1], "unrelated", "Unrelated negative prompt"),
    ):
        for item in [r for r in combo_rows if r["neg_kind"] == neg_kind]:
            l1 = float(item["lambda1"])
            l2 = float(item["lambda2"])
            ax.errorbar(
                item["clip_pos_mean"],
                item["clip_neg_mean"],
                xerr=item["clip_pos_sem"],
                yerr=item["clip_neg_sem"],
                marker=marker_by_l1[l1],
                color=color_by_l2[l2],
                linestyle="None",
                markersize=7,
                alpha=0.9,
            )

        for item in [b for b in base_rows if b["neg_kind"] == neg_kind]:
            ax.scatter(
                item["clip_pos_mean"],
                item["clip_neg_mean"],
                marker="*",
                s=120,
                color="black",
                alpha=0.7,
            )

        ax.set_title(panel_title)
        ax.set_xlabel("CLIP(image, positive prompt)")
        ax.set_ylabel(f"CLIP(image, {neg_kind} negative prompt)")
        ax.grid(True, alpha=0.25)

    preserve_rows = aggregate(rows, ("neg_kind", "lambda1", "lambda2"))
    for l1 in sorted(lambda1_values):
        for neg_kind, linestyle in (("related", "-"), ("unrelated", "--")):
            items = [r for r in preserve_rows if r["lambda1"] == l1 and r["neg_kind"] == neg_kind]
            items = sorted(items, key=lambda r: abs(float(r["lambda2"])))
            xs = [abs(float(r["lambda2"])) for r in items]
            ys = [float(r["clip_img_to_baseline_mean"]) for r in items]
            yerr = [float(r["clip_img_to_baseline_sem"]) for r in items]
            axes[2].errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=marker_by_l1[l1],
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6,
                label=f"{neg_kind}, lambda1={l1:g}",
            )

    axes[2].axhline(1.0, color="black", linewidth=1, alpha=0.4)
    axes[2].set_title("Guidance strength vs preservation")
    axes[2].set_xlabel("|lambda2| negative guidance strength")
    axes[2].set_ylabel("CLIP(guided image, baseline image)")
    axes[2].set_ylim(bottom=min(0.0, axes[2].get_ylim()[0]), top=1.02)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=7)

    handles = []
    labels = []
    for l2 in sorted(lambda2_values):
        handles.append(plt.Line2D([0], [0], color=color_by_l2[l2], marker="o", linestyle="None"))
        labels.append(f"lambda2={l2:g}")
    for l1 in sorted(lambda1_values):
        handles.append(plt.Line2D([0], [0], color="gray", marker=marker_by_l1[l1], linestyle="None"))
        labels.append(f"lambda1={l1:g}")
    handles.append(plt.Line2D([0], [0], color="black", marker="*", linestyle="None"))
    labels.append("baseline")
    axes[1].legend(handles, labels, fontsize=7, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plots(args: argparse.Namespace, prompts: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    fig_dir = args.out_dir / "figures"
    for entry in prompts:
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == int(entry["id"])]
        out_path = fig_dir / f"prompt_{int(entry['id']):02d}_{slug(entry['name'])}.png"
        plot_one_summary(
            prompt_rows,
            out_path,
            f"Prompt {int(entry['id'])}: {entry['name']}",
            args.lambda1,
            args.lambda2,
        )
        print(f"[plot] {out_path}", flush=True)

    prompt_count = len({int(row["prompt_id"]) for row in rows})
    avg_title = "Average across five prompts" if prompt_count == 5 else f"Average across {prompt_count} selected prompt(s)"
    plot_one_summary(rows, fig_dir / "average.png", avg_title, args.lambda1, args.lambda2)
    print(f"[plot] {fig_dir / 'average.png'}", flush=True)


def detect_devices() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ["cpu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--out-dir", type=Path, default=TEXT2IMAGE_DIR / "dng_eval/np_grid_eval")
    parser.add_argument("--model-id", type=str, default=SD_MODEL_ID)
    parser.add_argument("--scheduler", choices=("ddpm", "dpm_solver"), default="dpm_solver")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated generation devices.")
    parser.add_argument("--score-device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=None, help="Default uses the model native size.")
    parser.add_argument("--width", type=int, default=None, help="Default uses the model native size.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--prompt-id", "--prompt-index", dest="prompt_ids", type=parse_int_list, default=None,
                        help="Comma-separated DNG prompt ids to run. Default: all prompts.")
    parser.add_argument(
        "--negative-kinds",
        type=parse_negative_kinds,
        default=parse_negative_kinds("related,unrelated"),
        help="Comma-separated negative prompt kinds to generate: related,unrelated.",
    )
    parser.add_argument("--lambda1", type=parse_float_list, default=parse_float_list(DEFAULT_LAMBDA1))
    parser.add_argument("--lambda2", type=parse_float_list, default=parse_float_list(DEFAULT_LAMBDA2))
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
    return args


def main() -> None:
    args = parse_args()
    prompts, default_n_images = load_prompts(args.prompts_file)
    if args.prompt_ids is not None:
        keep = set(args.prompt_ids)
        prompts = [prompt for prompt in prompts if int(prompt["id"]) in keep]
        if not prompts:
            raise SystemExit(f"No prompts matched --prompt-id={sorted(keep)}")
    if args.n_images is None:
        args.n_images = default_n_images

    if any(l2 > 0 for l2 in args.lambda2):
        print("[warn] positive lambda2 attracts the negative prompt; use negative values to repel B.", file=sys.stderr)

    jobs = build_generation_jobs(args, prompts)
    n_guided = (
        len(prompts)
        * len(args.negative_kinds)
        * len(args.lambda1)
        * len(args.lambda2)
        * args.n_images
    )
    n_baseline = len(prompts) * len(args.lambda1) * args.n_images
    print(
        f"[config] prompts={len(prompts)} images/combo={args.n_images} "
        f"negative_kinds={args.negative_kinds} "
        f"lambda1={args.lambda1} lambda2={args.lambda2}",
        flush=True,
    )
    print(f"[config] model_id={args.model_id}; scheduler={args.scheduler}; steps={args.steps}", flush=True)
    print(
        f"[config] generation images: guided={n_guided}, baseline={n_baseline}, total={n_guided + n_baseline}",
        flush=True,
    )
    print(f"[config] devices={args.devices}; score_device={args.score_device}", flush=True)
    print(f"[config] out_dir={args.out_dir}", flush=True)

    if args.dry_run:
        return

    if not args.skip_generation:
        run_generation(args, jobs)

    metrics_path = args.out_dir / "metrics" / "clip_scores.csv"
    if args.skip_scoring:
        if not metrics_path.exists():
            raise FileNotFoundError(f"--skip-scoring was set but {metrics_path} does not exist")
        rows = read_csv(metrics_path)
    elif metrics_path.exists() and not args.force_score:
        print(f"[clip] using existing metrics {metrics_path}", flush=True)
        rows = read_csv(metrics_path)
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
