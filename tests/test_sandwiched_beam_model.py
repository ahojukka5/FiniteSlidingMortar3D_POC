from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "sandwiched_beam_model.py"
)
sys.path.insert(0, str(BENCHMARK_PATH.parent))
SPEC = importlib.util.spec_from_file_location("sandwiched_beam_model", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

COARSE_LEVEL = MODULE.COARSE_LEVEL
FINE_LEVEL = MODULE.FINE_LEVEL
LEVELS = MODULE.LEVELS
MEDIUM_LEVEL = MODULE.MEDIUM_LEVEL
SandwichedBeamGeometry = MODULE.SandwichedBeamGeometry
SandwichedBeamLevel = MODULE.SandwichedBeamLevel
build_sandwiched_beam_model = MODULE.build_sandwiched_beam_model
reference_determinants = MODULE.reference_determinants
sandwiched_beam_level = MODULE.sandwiched_beam_level


def _resultant_moment(
    nodes: np.ndarray,
    force: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.cross(nodes - origin, np.asarray(force).reshape((-1, 3))),
        axis=0,
    )


def test_coarse_model_is_deterministic_and_oriented() -> None:
    first = build_sandwiched_beam_model("coarse")
    second = build_sandwiched_beam_model("coarse")

    np.testing.assert_array_equal(
        first.problem.mesh.reference_nodes,
        second.problem.mesh.reference_nodes,
    )
    np.testing.assert_array_equal(
        first.problem.mesh.elements,
        second.problem.mesh.elements,
    )
    np.testing.assert_array_equal(first.slave_nodes, second.slave_nodes)
    np.testing.assert_array_equal(first.master_nodes, second.master_nodes)
    np.testing.assert_array_equal(
        first.reference_problem.mesh.reference_nodes,
        second.reference_problem.mesh.reference_nodes,
    )

    determinants = reference_determinants(
        first.problem.mesh.reference_nodes,
        first.problem.mesh.elements,
    )
    reference_values = reference_determinants(
        first.reference_problem.mesh.reference_nodes,
        first.reference_problem.mesh.elements,
    )
    assert np.all(determinants > 0.0)
    assert np.all(reference_values > 0.0)
    assert first.minimum_reference_determinant == pytest.approx(np.min(determinants))
    assert first.reference_minimum_determinant == pytest.approx(
        np.min(reference_values)
    )


def test_three_levels_refine_both_contact_bodies_and_reference() -> None:
    models = [build_sandwiched_beam_model(name) for name in LEVELS]

    contact_nodes = [model.problem.mesh.node_count for model in models]
    reference_nodes = [model.reference_problem.mesh.node_count for model in models]
    assert contact_nodes == sorted(contact_nodes)
    assert reference_nodes == sorted(reference_nodes)
    assert len(set(contact_nodes)) == 3
    assert len(set(reference_nodes)) == 3

    for model in models:
        interface = model.problem.interfaces[0]
        assert interface.pair.slave.node_count != interface.pair.master.node_count
        assert model.minimum_reference_determinant > 0.0
        assert model.reference_minimum_determinant > 0.0


def test_slave_side_choice_only_swaps_the_interface_mapping() -> None:
    upper = build_sandwiched_beam_model("coarse", slave_side="upper")
    lower = build_sandwiched_beam_model("coarse", slave_side="lower")

    np.testing.assert_array_equal(
        upper.problem.mesh.reference_nodes,
        lower.problem.mesh.reference_nodes,
    )
    np.testing.assert_array_equal(upper.path.pressure_force, lower.path.pressure_force)
    np.testing.assert_array_equal(upper.path.moment_force, lower.path.moment_force)
    np.testing.assert_array_equal(upper.slave_nodes, lower.master_nodes)
    np.testing.assert_array_equal(upper.master_nodes, lower.slave_nodes)

    upper_interface = upper.problem.interfaces[0]
    lower_interface = lower.problem.interfaces[0]
    np.testing.assert_array_equal(
        upper_interface.pair.slave.reference_nodes,
        upper.problem.mesh.reference_nodes[upper.upper_interface_nodes],
    )
    np.testing.assert_array_equal(
        lower_interface.pair.slave.reference_nodes,
        lower.problem.mesh.reference_nodes[lower.lower_interface_nodes],
    )
    assert all(len(facet) == 4 for facet in upper_interface.pair.slave.facets)
    assert all(len(facet) == 4 for facet in lower_interface.pair.slave.facets)


def test_pressure_and_end_couple_have_exact_resultants() -> None:
    model = build_sandwiched_beam_model("medium")
    geometry = model.geometry
    nodes = model.problem.mesh.reference_nodes
    pressure = model.path.pressure_force.reshape((-1, 3))
    moment = model.path.moment_force.reshape((-1, 3))

    np.testing.assert_allclose(np.sum(pressure, axis=0), 0.0, atol=1.0e-12)
    assert np.sum(pressure[: len(model.lower_nodes), 2]) == pytest.approx(
        geometry.ambient_pressure * geometry.loaded_area
    )
    assert np.sum(pressure[len(model.lower_nodes) :, 2]) == pytest.approx(
        -geometry.ambient_pressure * geometry.loaded_area
    )
    np.testing.assert_allclose(np.sum(moment, axis=0), 0.0, atol=1.0e-12)

    origin = np.array([geometry.length, 0.0, geometry.interface_z])
    resultant = _resultant_moment(nodes, moment, origin)
    np.testing.assert_allclose(resultant[[0, 2]], 0.0, atol=1.0e-12)
    assert resultant[1] == pytest.approx(geometry.end_moment)


def test_load_path_compresses_before_bending() -> None:
    model = build_sandwiched_beam_model("coarse")
    split = model.geometry.compression_end

    start = model.path.evaluate(model.problem, 0.0)
    halfway = model.path.evaluate(model.problem, 0.5 * split)
    compression_end = model.path.evaluate(model.problem, split)
    final = model.path.evaluate(model.problem, 1.0)

    assert model.path.phase_name(0.0) == "compression"
    assert model.path.phase_name(split) == "bending"
    np.testing.assert_allclose(start.effective_force, 0.0)
    assert halfway.value("pressure_scale") == pytest.approx(0.5)
    assert halfway.value("moment_scale") == pytest.approx(0.0)
    assert compression_end.value("pressure_scale") == pytest.approx(1.0)
    assert compression_end.value("moment_scale") == pytest.approx(0.0)
    assert compression_end.value("phase_index") == pytest.approx(1.0)
    assert final.value("pressure_scale") == pytest.approx(1.0)
    assert final.value("moment_scale") == pytest.approx(1.0)
    np.testing.assert_allclose(
        final.effective_force,
        model.path.pressure_force + model.path.moment_force,
    )


def test_reference_is_monolithic_with_the_same_physical_loads() -> None:
    model = build_sandwiched_beam_model("medium")
    geometry = model.geometry
    reference_nodes = model.reference_problem.mesh.reference_nodes

    interface_layer = reference_nodes[
        np.isclose(reference_nodes[:, 2], geometry.interface_z)
    ]
    assert len(interface_layer) > 0
    assert model.reference_problem.mesh.node_count != model.problem.mesh.node_count

    contact_pressure = model.path.pressure_force.reshape((-1, 3))
    reference_pressure = model.reference_path.pressure_force.reshape((-1, 3))
    np.testing.assert_allclose(np.sum(contact_pressure, axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(reference_pressure, axis=0),
        0.0,
        atol=1.0e-12,
    )

    origin = np.array([geometry.length, 0.0, geometry.interface_z])
    reference_moment = _resultant_moment(
        reference_nodes,
        model.reference_path.moment_force,
        origin,
    )
    assert reference_moment[1] == pytest.approx(geometry.end_moment)
    np.testing.assert_allclose(reference_moment[[0, 2]], 0.0, atol=1.0e-12)


def test_symmetry_constraints_remove_rigid_modes_without_clamping_interface() -> None:
    model = build_sandwiched_beam_model("coarse")
    nodes = model.problem.mesh.reference_nodes
    constrained = set(int(value) for value in model.problem.constraints.dofs)

    x_nodes = np.flatnonzero(np.isclose(nodes[:, 0], 0.0))
    y_nodes = np.flatnonzero(np.isclose(nodes[:, 1], 0.0))
    assert all(3 * int(node) in constrained for node in x_nodes)
    assert all(3 * int(node) + 1 in constrained for node in y_nodes)

    z_constraints = [dof for dof in constrained if dof % 3 == 2]
    assert len(z_constraints) == 1
    interface_dofs = {
        3 * int(node) + component
        for node in np.concatenate(
            [model.lower_interface_nodes, model.upper_interface_nodes]
        )
        for component in range(3)
    }
    assert interface_dofs - constrained


def test_level_and_input_validation() -> None:
    assert sandwiched_beam_level("coarse") is COARSE_LEVEL
    assert sandwiched_beam_level("medium") is MEDIUM_LEVEL
    assert sandwiched_beam_level("fine") is FINE_LEVEL
    with pytest.raises(ValueError, match="level must be"):
        sandwiched_beam_level("unknown")
    with pytest.raises(ValueError, match="slave side"):
        build_sandwiched_beam_model("coarse", slave_side="invalid")
    with pytest.raises(ValueError, match="nonmatching"):
        SandwichedBeamLevel("coarse", (2, 1, 1), (2, 1, 1), (4, 1, 2))
    with pytest.raises(ValueError, match="interface"):
        SandwichedBeamLevel("coarse", (2, 1, 1), (3, 1, 1), (4, 1, 3))
    with pytest.raises(ValueError, match="positive"):
        SandwichedBeamGeometry(end_moment=0.0)
    with pytest.raises(ValueError, match="inside"):
        model = build_sandwiched_beam_model("coarse")
        model.path.force(1.1)
