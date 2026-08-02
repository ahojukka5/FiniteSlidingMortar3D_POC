from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_load_module("rotating_blocks_model", "rotating_blocks_model.py")
REPETITION = _load_module(
    "rotating_blocks_repetition",
    "rotating_blocks_repetition.py",
)


def test_quick_repetition_check_is_independent_and_machine_readable(tmp_path) -> None:
    summary = REPETITION.run(tmp_path, profile="quick", sample_count=17)

    assert summary["comparison"]["passed"]
    assert summary["comparison"]["first_divergence"] is None
    assert summary["comparison"]["frame_count"] == 17
    assert summary["absolute_tolerance"] == 1.0e-12
    assert summary["relative_tolerance"] == 1.0e-10

    stored = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert stored == summary
    assert manifest["provenance"]["solver_settings"]["nonlinear_solver"] is None
    assert {item["path"] for item in manifest["artifacts"]} >= {"summary.json"}
