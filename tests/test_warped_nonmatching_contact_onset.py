from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from contact3d import MortarContactInterface, evaluate_coupled_equilibrium

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "warped_nonmatching_contact_onset.py"
)
sys.path.insert(0, str(BENCHMARK_PATH.parent))
SPEC = importlib.util.spec_from_file_location("warped_nonmatching_contact_onset", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_model_uses_warped_nonmatching_production_interface() -> None:
    benchmark = MODULE.model()
    interface = benchmark.problem.interfaces[0]
    assert isinstance(interface, MortarContactInterface)
    assert tuple(len(facet) for facet in interface.pair.slave.facets) == (4,)
    assert tuple(len(facet) for facet in interface.pair.master.facets) == (3, 3)
    assert interface.pair.slave.normal_sign == -1.0
    assert benchmark.initial_separation > 0.0

    slave = interface.pair.slave.reference_nodes
    master = interface.pair.master.reference_nodes
    assert np.ptp(slave[:, 2]) > 0.0
    assert np.ptp(master[:, 2]) > 0.0
    assert not np.allclose(slave[:, :2], master[:, :2])


def test_reference_mesh_is_positive_and_contact_path_is_mixed() -> None:
    benchmark = MODULE.model()
    determinants = []
    nodes = benchmark.problem.mesh.reference_nodes
    for element in benchmark.problem.mesh.elements:
        origin = nodes[element[0]]
        matrix = np.column_stack(
            [nodes[element[index]] - origin for index in (1, 2, 3)]
        )
        determinants.append(np.linalg.det(matrix))
    assert min(determinants) > 0.49

    start = benchmark.path.evaluate(benchmark.problem, 0.0)
    end = benchmark.path.evaluate(benchmark.problem, 1.0)
    assert end.value("tool_x") == pytest.approx(0.04)
    assert end.value("tool_z") == pytest.approx(-0.09)
    assert end.value("dead_load_x") == pytest.approx(0.50)
    assert np.linalg.norm(end.prescribed_values - start.prescribed_values) > 0.0
    assert np.linalg.norm(end.effective_force - start.effective_force) > 0.0


def _translated_state(parameter: float):
    benchmark = MODULE.model()
    path_state = benchmark.path.evaluate(benchmark.problem, parameter)
    displacement = np.zeros(3 * benchmark.problem.mesh.node_count)
    values = displacement.reshape((-1, 3))
    values[9:18, 0] = 0.04 * parameter
    values[9:18, 2] = -0.09 * parameter
    displacement[path_state.prescribed_dofs] = path_state.prescribed_values
    states = path_state.problem.initial_states()
    return path_state, displacement, states


@pytest.mark.parametrize("parameter", [0.20, 0.55, 0.90])
def test_production_coupled_tangent_matches_centered_difference(parameter: float) -> None:
    path_state, displacement, states = _translated_state(parameter)
    base = evaluate_coupled_equilibrium(
        path_state.problem,
        displacement,
        states,
        load_factor=path_state.solver_load_factor,
    )
    assert base.tangent is not None

    rng = np.random.default_rng(17000 + int(100 * parameter))
    direction = np.zeros_like(displacement)
    direction[base.free_dofs] = rng.normal(size=len(base.free_dofs))
    direction /= np.linalg.norm(direction)
    increment = 2.0e-7
    plus = evaluate_coupled_equilibrium(
        path_state.problem,
        displacement + increment * direction,
        states,
        load_factor=path_state.solver_load_factor,
        assemble_tangent=False,
    )
    minus = evaluate_coupled_equilibrium(
        path_state.problem,
        displacement - increment * direction,
        states,
        load_factor=path_state.solver_load_factor,
        assemble_tangent=False,
    )
    assert plus.signatures == base.signatures == minus.signatures
    numerical = (plus.residual - minus.residual) / (2.0 * increment)
    analytical = base.tangent.matvec(direction)
    free = base.free_dofs
    relative = np.linalg.norm(analytical[free] - numerical[free]) / np.linalg.norm(
        numerical[free]
    )
    assert relative < 2.0e-6


def test_contact_patch_example(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.contact_patch",
            "--output",
            str(tmp_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=600.0,
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    printed_metrics = json.loads(completed.stdout)
    metrics = summary["metrics"]
    assert printed_metrics == metrics
    assert summary["passed"]
    assert metrics["converged"]
    assert metrics["final_parameter"] == pytest.approx(1.0)
    assert 0.0 < metrics["contact_onset_parameter"] < 1.0
    assert metrics["final_active_rows"] > 0
    assert metrics["final_facet_pairs"] == 2
    assert metrics["final_supported_rows"] == 4
    assert metrics["maximum_partition_error"] < 1.0e-10
    assert metrics["minimum_element_jacobian"] > 0.0
    assert metrics["final_normalized_residual"] < 1.0e-8
    assert metrics["final_normalized_penetration"] < 2.0e-7
    assert {path.name for path in tmp_path.iterdir()} == {
        "final.vtu",
        "pressure.svg",
        "summary.json",
    }
