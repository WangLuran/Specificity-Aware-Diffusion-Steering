"""Sequence and tabular helpers for the Boltz TCR-pMHC pilot."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
PLACEHOLDER_VALUES = {
    "",
    "NA",
    "N/A",
    "NONE",
    "NULL",
    "PLACEHOLDER",
    "PLACEHOLDER_REQUIRED",
    "PLACEHOLDERREQUIRED",
    "TODO",
    "TBD",
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root_from_script() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(config: dict, key: str) -> Path:
    root = project_root_from_script()
    path = Path(config["paths"][key])
    return path if path.is_absolute() else root / path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_sequence(value: object) -> str:
    text = clean_str(value).upper()
    text = re.sub(r"[^A-Z]", "", text)
    return text


def is_placeholder(value: object) -> bool:
    text = clean_str(value).upper()
    return text in PLACEHOLDER_VALUES


def has_real_sequence(value: object) -> bool:
    return bool(clean_sequence(value)) and not is_placeholder(value)


def validate_protein_sequence(seq: object, allow_x: bool = False) -> tuple[bool, str]:
    cleaned = clean_sequence(seq)
    if not cleaned:
        return False, "empty sequence"
    allowed = set(AMINO_ACIDS)
    if allow_x:
        allowed.add("X")
    bad = sorted(set(cleaned) - allowed)
    if bad:
        return False, f"invalid residue letters: {''.join(bad)}"
    return True, ""


def hamming_distance(a: object, b: object) -> int:
    left = clean_sequence(a)
    right = clean_sequence(b)
    shared = sum(aa != bb for aa, bb in zip(left, right))
    return shared + abs(len(left) - len(right))


def safe_id(*parts: object, max_len: int = 120) -> str:
    raw = "_".join(clean_str(part) for part in parts if clean_str(part))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        safe = "complex"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    base_len = max(1, max_len - len(digest) - 1)
    return f"{safe[:base_len]}_{digest}"


def join_unique(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        text = clean_str(value)
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False)


def read_csv_if_exists(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        columns = list(required_columns or [])
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if required_columns is not None:
        require_columns(normalize_columns(df), [c.lower() for c in required_columns], str(path))
    return df
