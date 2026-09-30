#!/usr/bin/env python
"""Run Boltz predictions with one process per GPU slot."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils_boltz import discover_yaml_inputs, format_command, logs_dir, output_complete, quote_command
from utils_sequences import load_config, resolve_project_path, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_jobs_per_gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--boltz_executable",
        type=Path,
        help="Override boltz.command.executable from the scheduler config.",
    )
    parser.add_argument(
        "--boltz_cache",
        type=Path,
        help="Override the argument immediately following --cache.",
    )
    parser.add_argument("--status_interval", type=float, default=30.0, help="Seconds between live progress updates. Use 0 to disable.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_log(path: Path, n_lines: int = 8) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " | ".join(lines[-n_lines:])


def run_one(yaml_path: Path, gpu_id: str, config: dict, args: argparse.Namespace, lock: threading.Lock, active: dict) -> dict:
    complex_id = yaml_path.stem
    command = format_command(config, yaml_path)
    command_text = quote_command(command)
    log_path = logs_dir(config) / f"{complex_id}.log"

    if args.resume and output_complete(config, complex_id):
        return {
            "complex_id": complex_id,
            "gpu_id": gpu_id,
            "start_time": "",
            "end_time": "",
            "elapsed_seconds": 0.0,
            "command": command_text,
            "return_code": 0,
            "log_path": str(log_path),
            "success": True,
            "notes": "skipped_existing_output",
        }

    if args.dry_run:
        print(f"[dry-run gpu {gpu_id}] CUDA_VISIBLE_DEVICES={gpu_id} {command_text}")
        return {
            "complex_id": complex_id,
            "gpu_id": gpu_id,
            "start_time": "",
            "end_time": "",
            "elapsed_seconds": 0.0,
            "command": command_text,
            "return_code": None,
            "log_path": str(log_path),
            "success": None,
            "notes": "dry_run",
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    project_root = Path(__file__).resolve().parents[1]
    runtime_cache_base = Path(env.get("BOLTZ_RUNTIME_CACHE_BASE", project_root))
    env.setdefault("XDG_CACHE_HOME", str(runtime_cache_base / "cache_xdg"))
    env["TRITON_CACHE_DIR"] = str(
        runtime_cache_base / "cache_triton" / f"gpu_{gpu_id}"
    )
    env["TORCHINDUCTOR_CACHE_DIR"] = str(
        runtime_cache_base / "cache_torchinductor" / f"gpu_{gpu_id}"
    )
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    start_wall = time.time()
    start_time = now_iso()
    with lock:
        active[complex_id] = {"gpu_id": gpu_id, "start_wall": start_wall, "log_path": str(log_path)}
    print(f"[start gpu {gpu_id}] {complex_id}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"start_time={start_time}\n")
        log_handle.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
        log_handle.write(f"command={command_text}\n\n")
        log_handle.flush()
        proc = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
            check=False,
        )
        end_time = now_iso()
        elapsed = time.time() - start_wall
        log_handle.write(f"\nend_time={end_time}\n")
        log_handle.write(f"elapsed_seconds={elapsed:.3f}\n")
        log_handle.write(f"return_code={proc.returncode}\n")

    success = proc.returncode == 0 and output_complete(config, complex_id)
    print(
        f"[done  gpu {gpu_id}] {complex_id} rc={proc.returncode} success={success} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "complex_id": complex_id,
        "gpu_id": gpu_id,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed,
        "command": command_text,
        "return_code": proc.returncode,
        "log_path": str(log_path),
        "success": success,
        "notes": "" if success else tail_log(log_path, n_lines=6),
    }


def worker(job_queue: queue.Queue, gpu_id: str, config: dict, args: argparse.Namespace, rows: list, lock: threading.Lock, active: dict) -> None:
    while True:
        try:
            yaml_path = job_queue.get_nowait()
        except queue.Empty:
            return
        try:
            row = run_one(yaml_path, gpu_id, config, args, lock, active)
        except Exception as exc:
            row = {
                "complex_id": yaml_path.stem,
                "gpu_id": gpu_id,
                "start_time": "",
                "end_time": now_iso(),
                "elapsed_seconds": 0.0,
                "command": "",
                "return_code": -1,
                "log_path": str(logs_dir(config) / f"{yaml_path.stem}.log"),
                "success": False,
                "notes": f"runner_exception: {exc}",
            }
        with lock:
            active.pop(yaml_path.stem, None)
            rows.append(row)
        job_queue.task_done()


def monitor_progress(job_queue: queue.Queue, rows: list, active: dict, lock: threading.Lock, total: int, stop_event: threading.Event, interval: float) -> None:
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        with lock:
            done = len(rows)
            running = len(active)
            queued = job_queue.qsize()
            failed = sum(1 for row in rows if row.get("success") is False)
            active_preview = ", ".join(
                f"{cid}@gpu{info['gpu_id']}:{time.time() - info['start_wall']:.0f}s"
                for cid, info in list(active.items())[:4]
            )
        print(
            f"[status] done={done}/{total} running={running} queued={queued} failed={failed}"
            + (f" active={active_preview}" if active_preview else ""),
            flush=True,
        )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    command_config = config["boltz"]["command"]
    if args.boltz_executable is not None:
        command_config["executable"] = str(args.boltz_executable.resolve())
    if args.boltz_cache is not None:
        command_args = command_config.get("args", [])
        try:
            cache_index = command_args.index("--cache") + 1
        except ValueError as exc:
            raise ValueError("Scheduler command has no --cache argument to override") from exc
        if cache_index >= len(command_args):
            raise ValueError("Scheduler command ends immediately after --cache")
        command_args[cache_index] = str(args.boltz_cache.resolve())
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if args.max_jobs_per_gpu < 1:
        raise ValueError("--max_jobs_per_gpu must be >= 1")

    yaml_inputs = discover_yaml_inputs(config)
    if args.limit is not None:
        yaml_inputs = yaml_inputs[: args.limit]
    if not yaml_inputs:
        print("No Boltz YAML inputs found. Run scripts/03_make_boltz_inputs.py first.")
        return 0

    jobs: queue.Queue = queue.Queue()
    for yaml_path in yaml_inputs:
        jobs.put(yaml_path)

    rows: list[dict] = []
    lock = threading.Lock()
    active: dict = {}
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_progress,
        args=(jobs, rows, active, lock, len(yaml_inputs), stop_event, args.status_interval),
        daemon=True,
    )
    monitor.start()
    threads = []
    for gpu_id in gpus:
        for _ in range(args.max_jobs_per_gpu):
            thread = threading.Thread(target=worker, args=(jobs, gpu_id, config, args, rows, lock, active), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    stop_event.set()
    monitor.join(timeout=1)

    runtime_path = resolve_project_path(config, "runtime_log")
    if runtime_path.exists():
        previous = pd.read_csv(runtime_path)
        previous_by_id = {
            str(row["complex_id"]): row.to_dict()
            for _, row in previous.iterrows()
        }
        for index, row in enumerate(rows):
            if row.get("notes") != "skipped_existing_output":
                continue
            prior = previous_by_id.get(str(row["complex_id"]))
            if prior and bool(prior.get("success")):
                prior["notes"] = "skipped_existing_output; prior_runtime_preserved"
                rows[index] = prior
    runtime_df = pd.DataFrame(rows).sort_values(["complex_id", "gpu_id"])
    write_csv(runtime_df, runtime_path)
    print(f"wrote runtime log for {len(runtime_df)} jobs to {runtime_path}")

    failed = runtime_df[runtime_df["success"] == False]  # noqa: E712
    if not failed.empty:
        print(f"WARNING: {len(failed)} jobs failed or produced incomplete outputs")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
