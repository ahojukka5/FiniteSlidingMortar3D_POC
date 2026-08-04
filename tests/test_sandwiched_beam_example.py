from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


def test_sandwiched_beam_example(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.sandwiched_beam",
            "--output",
            str(tmp_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=900.0,
    )
    summary_path = tmp_path / "summary.json"
    diagnostic = completed.stderr
    if summary_path.is_file():
        diagnostic += "\nsummary.json:\n" + summary_path.read_text(encoding="utf-8")
    assert completed.returncode == 0, diagnostic

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    assert json.loads(completed.stdout) == metrics
    assert summary["passed"]
    assert metrics["converged"]
    assert metrics["final_parameter"] == pytest.approx(1.0)
    assert metrics["final_active_rows"] > 0
    assert metrics["final_supported_rows"] >= metrics["final_active_rows"]
    assert metrics["final_maximum_pressure"] > 0.0
    assert abs(metrics["final_end_rotation"]) > 1.0e-6
    assert metrics["final_end_rotation"] * metrics["reference_end_rotation"] > 0.0
    assert metrics["final_normalized_equilibrium_residual"] <= 1.0e-8
    assert metrics["final_normalized_penetration"] <= 2.0e-7
    assert metrics["final_force_balance_relative"] <= 1.0e-8
    assert math.isfinite(metrics["final_moment_balance_relative"])
    assert "moment_balance" not in summary["checks"]
    assert "relative_moment_balance" not in summary["tolerances"]
    assert (
        summary["diagnostics"]["angular_momentum_balance"]["role"]
        == "reported_only"
    )
    assert metrics["minimum_element_jacobian"] > 0.0
    assert summary["history"][-1]["phase"] == "bending"
    assert summary["reference_history"][-1]["parameter"] == pytest.approx(1.0)
    assert {path.name for path in tmp_path.iterdir()} == {
        "deformed.svg",
        "final.vtu",
        "moment-rotation.svg",
        "summary.json",
    }