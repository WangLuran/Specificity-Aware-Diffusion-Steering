from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_t2i_reference_row() -> None:
    module = load_module("t2i_summary", ROOT / "t2i/summarize_results.py")
    rows = module.read_rows(ROOT / "t2i/reference/condition_results.csv")
    table = module.summarize(rows)
    assert len(rows) == 10
    assert round(table["related"]["clip_positive"], 4) == 0.2722
    assert round(table["related"]["clip_negative"], 4) == 0.1413
    assert round(table["related"]["margin"], 4) == 0.1309
    assert round(table["unrelated"]["clip_positive"], 4) == 0.2679
    assert round(table["unrelated"]["clip_negative"], 4) == 0.0655
    assert round(table["unrelated"]["margin"], 4) == 0.2024
    assert round(table["overall_margin"], 4) == 0.1666


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("boltzgen_target_a_only.csv", (0.1785687724, 0.1587113043, 0.2861846387, 0.875)),
        ("fixed_cfg.csv", (0.1654209006, 0.1453312139, 0.2832839750, 0.9375)),
        ("dng.csv", (0.1482974475, 0.1580531647, 0.2622768842, 0.890625)),
        ("ours_prop2.csv", (0.1856199875, 0.1821851209, 0.3336834523, 0.84375)),
    ],
)
def test_binder_reference_rows(filename: str, expected: tuple[float, ...]) -> None:
    module = load_module("binder_summary", ROOT / "binder/summarize_table.py")
    path = ROOT / "binder/reference" / filename
    rows = module.collect(path)
    result = module.metrics(rows)
    assert result["n"] == 64
    for key, value in zip(("mean", "median", "q75", "positive_fraction"), expected):
        assert result[key] == pytest.approx(value, abs=1e-10)


def test_binder_references_are_per_binder_measurements() -> None:
    for path in (ROOT / "binder/reference").glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 64
        assert len({row["binder_id"] for row in rows}) == 64


def test_boltzgen_overlay_is_complete() -> None:
    overlay = ROOT / "binder/boltzgen_overlay"
    expected = [
        "src/boltzgen/cli/boltzgen.py",
        "src/boltzgen/model/modules/diffusion.py",
        "src/boltzgen/task/predict/data_from_yaml.py",
        "tests/test_binder_negative_guidance.py",
        "LICENSE.boltzgen",
    ]
    assert all((overlay / relative).is_file() for relative in expected)

