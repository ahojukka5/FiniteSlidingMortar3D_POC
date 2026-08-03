from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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
SPECIAL = _load_module(
    "rotating_blocks_special_states",
    "rotating_blocks_special_states.py",
)


def test_special_states_use_rotating_block_facet_dimensions() -> None:
    cases = SPECIAL.rotating_blocks_special_states()
    geometry = SPECIAL.RotatingBlocksGeometry()
    profile = SPECIAL.QUICK_PROFILE
    expected_master_width = (
        geometry.lower_maximum[0] - geometry.lower_minimum[0]
    ) / profile.lower_cells[0]
    expected_slave_width = (
        geometry.upper_maximum[0] - geometry.upper_minimum[0]
    ) / profile.upper_cells[0]
    expected_slave_height = (
        geometry.upper_maximum[1] - geometry.upper_minimum[1]
    ) / profile.upper_cells[1]

    assert [case.name for case in cases] == ["edge-on-edge", "on-vertex"]
    for case in cases:
        assert np.isclose(np.ptp(case.master[:, 0]), expected_master_width)
        assert np.isclose(np.ptp(case.slave[:, 0]), expected_slave_width)
        assert np.isclose(np.ptp(case.slave[:, 1]), expected_slave_height)
        assert np.all(case.slave[:, 2] == geometry.lower_maximum[2])
        assert np.all(case.master[:, 2] == geometry.lower_maximum[2])


def test_exact_states_have_typed_diagnostics_and_verified_branches() -> None:
    summary = SPECIAL.run()

    assert summary["passed"]
    assert {case["name"] for case in summary["cases"]} == {
        "edge-on-edge",
        "on-vertex",
    }
    for case in summary["cases"]:
        assert case["passed"]
        assert case["exact_diagnostic"]["type"] in {
            "ClippingTopologyError",
            "PalletTopologyError",
            "InverseMapTopologyError",
        }
        assert "clipping_vertex_edge" in case["event_kinds"]
        assert case["left_selected_branch"] == "left"
        assert case["right_selected_branch"] == "right"
        assert case["criteria"]["left_branch_selected"]
        assert case["criteria"]["right_branch_selected"]
        assert case["criteria"]["selected_branches_distinct"]
        assert np.isfinite(case["left_selected_fraction"])
        assert np.isfinite(case["right_selected_fraction"])
        assert case["maximum_tangent_error"] <= 5.0e-5
    json.dumps(summary, allow_nan=False)


def test_special_state_artifacts_are_manifested(tmp_path: Path) -> None:
    output = tmp_path / "special-states"
    summary = SPECIAL.run(output)

    assert summary["passed"]
    assert (output / "summary.json").is_file()
    assert (output / "branches.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert {"summary.json", "branches.csv"}.issubset(paths)
