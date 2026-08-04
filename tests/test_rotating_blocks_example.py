from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_rotating_blocks_example(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.rotating_blocks",
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
    assert metrics["final_rotation_angle"] == pytest.approx(1.5707963267948966)
    assert metrics["final_active_rows"] > 0
    assert metrics["final_supported_rows"] >= metrics["final_active_rows"]
    assert metrics["event_count"] >= 2
    assert metrics["facet_pair_count_range"] > 0
    assert metrics["unique_overlap_topologies"] > 1
    assert metrics["maximum_normalized_equilibrium_residual"] <= 1.0e-8
    assert metrics["maximum_normalized_penetration"] <= 1.0e-7
    assert metrics["maximum_force_balance_relative"] <= 1.0e-7
    assert metrics["minimum_element_jacobian"] > 0.0
    assert summary["history"][-1]["parameter"] == pytest.approx(1.0)

    vtk_states = summary["artifacts"]["vtk_states"]
    assert set(vtk_states) == {
        "compression.vtu",
        "mid-rotation.vtu",
        "final.vtu",
    }
    assert 0.0 < vtk_states["compression.vtu"] < vtk_states["mid-rotation.vtu"]
    assert vtk_states["mid-rotation.vtu"] < vtk_states["final.vtu"]
    assert vtk_states["final.vtu"] == pytest.approx(1.0)
    assert {path.name for path in tmp_path.iterdir()} == {
        "compression.vtu",
        "mid-rotation.vtu",
        "final.vtu",
        "deformed.svg",
        "reaction-path.svg",
        "summary.json",
    }
