"""Boltz command and output-path helpers."""

from __future__ import annotations

import shlex
from pathlib import Path

from utils_sequences import ensure_dir, project_root_from_script, resolve_project_path


def inputs_dir(config: dict) -> Path:
    return ensure_dir(resolve_project_path(config, "boltz_inputs_dir"))


def outputs_dir(config: dict) -> Path:
    return ensure_dir(resolve_project_path(config, "boltz_outputs_dir"))


def logs_dir(config: dict) -> Path:
    return ensure_dir(resolve_project_path(config, "logs_dir"))


def discover_yaml_inputs(config: dict) -> list[Path]:
    root = inputs_dir(config)
    return sorted(root.glob("*.yaml"))


def prediction_dir(config: dict, complex_id: str) -> Path:
    return outputs_dir(config) / f"boltz_results_{complex_id}" / "predictions" / complex_id


def model_cif_path(config: dict, complex_id: str, model_index: int) -> Path:
    return prediction_dir(config, complex_id) / f"{complex_id}_model_{model_index}.cif"


def confidence_json_path(config: dict, complex_id: str, model_index: int) -> Path:
    return prediction_dir(config, complex_id) / f"confidence_{complex_id}_model_{model_index}.json"


def expected_model_paths(config: dict, complex_id: str) -> list[tuple[Path, Path]]:
    n = int(config["boltz"].get("diffusion_samples", 5))
    return [(model_cif_path(config, complex_id, i), confidence_json_path(config, complex_id, i)) for i in range(n)]


def output_complete(config: dict, complex_id: str) -> bool:
    expected = expected_model_paths(config, complex_id)
    return bool(expected) and all(cif.exists() and conf.exists() for cif, conf in expected)


def format_command(config: dict, yaml_path: Path) -> list[str]:
    cmd_cfg = config["boltz"]["command"]
    outputs = outputs_dir(config)
    mapping = {
        "yaml_path": str(yaml_path),
        "outputs_dir": str(outputs),
        "project_root": str(project_root_from_script()),
    }
    command = [cmd_cfg.get("executable", "boltz")]
    for arg in cmd_cfg.get("args", []):
        command.append(str(arg).format(**mapping))
    return command


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
