from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "rotating_blocks_model.py"
)
sys.path.insert(0, str(BENCHMARK_PATH.parent))
SPEC = importlib.util.spec_from_file_location("rotating_blocks_model", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FULL_PROFILE = MODULE.FULL_PROFILE
QUICK_PROFILE = MODULE.QUICK_PROFILE
RotatingBlocksGeometry = MODULE.RotatingBlocksGeometry
RotatingBlocksProfile = MODULE.RotatingBlocksProfile
build_rotating_blocks_model = MODULE.build_rotating_blocks_model
reference_determinants = MODULE.reference_determinants
rotating_blocks_profile = MODULE.rotating_blocks_profile


def _controlled_dofs(nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [3 * int(node) + component for node in nodes for component in range(3)],
        dtype=np.int64,
    )


def _assert_distinct_grid_lines(slave: np.ndarray, master: np.ndarray) -> None:
    for axis in (0, 1):
        slave_values = np.unique(slave[:, axis])
        master_values = np.unique(master[:, axis])
        distances = np.abs(slave_values[:, None] - master_values[None, :])
        assert float(np.min(distances)) > 1.0e-12


def test_quick_model_is_deterministic_and_oriented() -> None:
    first = build_rotating_blocks_model("quick")
    second = build_rotating_blocks_model("quick")

    np.testing.assert_array_equal(
        first.problem.mesh.reference_nodes,
        second.problem.mesh.reference_nodes,
    )
    np.testing.assert_array_equal(
        first.problem.mesh.elements,
        second.problem.mesh.elements,
    )
    np.testing.assert_array_equal(first.fixed_nodes, second.fixed_nodes)
    np.testing.assert_array_equal(first.controlled_nodes, second.controlled_nodes)
    np.testing.assert_array_equal(first.slave_nodes, second.slave_nodes)
    np.testing.assert_array_equal(first.master_nodes, second.master_nodes)

    determinants = reference_determinants(
        first.problem.mesh.reference_nodes,
        first.problem.mesh.elements,
    )
    assert np.all(determinants > 0.0)
    assert first.minimum_reference_determinant == pytest.approx(np.min(determinants))
    assert len(first.lower_elements) == 24


def test_contact_surfaces_are_exact_nonmatching_quad_mappings() -> None:
    model = build_rotating_blocks_model("quick")
    interface = model.problem.interfaces[0]
    mesh = model.problem.mesh

    np.testing.assert_array_equal(
        interface.pair.slave.reference_nodes,
        mesh.reference_nodes[model.slave_nodes],
    )
    np.testing.assert_array_equal(
        interface.pair.master.reference_nodes,
        mesh.reference_nodes[model.master_nodes],
    )
    assert all(len(facet) == 4 for facet in interface.pair.slave.facets)
    assert all(len(facet) == 4 for facet in interface.pair.master.facets)
    assert interface.pair.slave.node_count == 12
    assert interface.pair.master.node_count == 9
    assert len(interface.pair.slave.facets) == 6
    assert len(interface.pair.master.facets) == 4
    assert interface.pair.search_distance > model.initial_separation > 0.0


def test_contact_grid_lines_are_smooth_at_path_endpoints() -> None:
    model = build_rotating_blocks_model("quick")
    mesh = model.problem.mesh
    master = mesh.reference_nodes[model.master_nodes]
    slave = mesh.reference_nodes[model.slave_nodes]
    _assert_distinct_grid_lines(slave, master)

    final = model.path.evaluate(model.problem, model.path.end_parameter)
    displacement = np.zeros(3 * mesh.node_count)
    displacement[final.prescribed_dofs] = final.prescribed_values
    current = mesh.reference_nodes + displacement.reshape((-1, 3))
    _assert_distinct_grid_lines(current[model.slave_nodes], master)


def test_contact_onset_lies_inside_nominal_continuation_steps() -> None:
    model = build_rotating_blocks_model("quick")
    geometry = model.geometry
    onset = geometry.contact_onset_parameter

    assert onset == pytest.approx(0.13125)
    assert 0.0 < onset < geometry.compression_end
    for requested_steps in (16, 64):
        step_coordinate = onset * requested_steps
        assert not np.isclose(step_coordinate, round(step_coordinate), atol=1.0e-12)

    epsilon = 1.0e-8
    before = model.path.evaluate(model.problem, onset - epsilon)
    after = model.path.evaluate(model.problem, onset + epsilon)
    gap_before = model.initial_separation + before.value("translation_z")
    gap_after = model.initial_separation + after.value("translation_z")
    assert gap_before > 0.0
    assert gap_after < 0.0


def test_upper_block_is_completely_rigidly_controlled() -> None:
    model = build_rotating_blocks_model("quick")
    lower_count = len(model.lower_nodes)
    expected_upper = np.arange(
        lower_count,
        lower_count + len(model.upper_nodes),
        dtype=np.int64,
    )
    np.testing.assert_array_equal(model.controlled_nodes, expected_upper)
    assert np.intersect1d(model.fixed_nodes, model.controlled_nodes).size == 0

    constrained = model.problem.constraints.dofs
    fixed_dofs = _controlled_dofs(model.fixed_nodes)
    controlled_dofs = _controlled_dofs(model.controlled_nodes)
    np.testing.assert_array_equal(constrained, np.concatenate([fixed_dofs, controlled_dofs]))
    np.testing.assert_allclose(model.problem.constraints.values, 0.0)


def test_staged_path_compresses_then_rotates_about_the_contact_center() -> None:
    model = build_rotating_blocks_model("quick")
    geometry = model.geometry
    split = geometry.compression_end

    compression_end = model.path.evaluate(model.problem, split)
    assert model.path.phase_name(split) == "rotation"
    assert compression_end.value("phase_index") == pytest.approx(1.0)
    assert compression_end.value("phase_parameter") == pytest.approx(0.0)
    assert compression_end.value("rotation_angle") == pytest.approx(0.0)
    assert compression_end.value("translation_z") == pytest.approx(
        geometry.compression[2]
    )

    final = model.path.evaluate(model.problem, 1.0)
    assert final.value("rotation_angle") == pytest.approx(geometry.final_angle)
    assert final.value("translation_x") == pytest.approx(
        geometry.compression[0] + geometry.tangential_translation[0]
    )
    assert final.value("translation_z") == pytest.approx(
        geometry.compression[2] + geometry.tangential_translation[2]
    )

    reference = model.problem.mesh.reference_nodes[model.controlled_nodes]
    displacement = final.problem.constraints.values[len(model.fixed_nodes) * 3 :]
    current = reference + displacement.reshape((-1, 3))
    pivot = np.asarray(geometry.pivot)
    translation = np.asarray(geometry.compression) + np.asarray(
        geometry.tangential_translation
    )
    relative = reference[0] - pivot
    expected = pivot + translation + np.array([-relative[1], relative[0], relative[2]])
    np.testing.assert_allclose(current[0], expected, atol=2.0e-15)


def test_quick_and_full_profiles_share_physical_geometry_and_motion() -> None:
    quick = build_rotating_blocks_model(QUICK_PROFILE)
    full = build_rotating_blocks_model(FULL_PROFILE)

    assert quick.geometry == full.geometry
    assert quick.initial_separation == pytest.approx(full.initial_separation)
    assert quick.path.end_parameter == pytest.approx(full.path.end_parameter)
    assert len(quick.lower_nodes) < len(full.lower_nodes)
    assert len(quick.upper_nodes) == len(full.upper_nodes)

    for parameter in (0.0, quick.geometry.compression_end, 0.625, 1.0):
        quick_state = quick.path.evaluate(quick.problem, parameter)
        full_state = full.path.evaluate(full.problem, parameter)
        for name in (
            "rotation_angle",
            "translation_x",
            "translation_y",
            "translation_z",
            "phase_index",
            "phase_parameter",
        ):
            assert quick_state.value(name) == pytest.approx(full_state.value(name))


def test_profile_lookup_and_custom_profile_validation() -> None:
    assert rotating_blocks_profile("quick") is QUICK_PROFILE
    assert rotating_blocks_profile("full") is FULL_PROFILE
    with pytest.raises(ValueError, match="profile must be"):
        rotating_blocks_profile("unknown")
    custom = RotatingBlocksProfile("quick", (2, 2, 1), (1, 1, 1))
    custom_model = build_rotating_blocks_model(custom)
    assert custom_model.profile is custom
    assert custom_model.profile.lower_cells == (2, 2, 1)


def test_search_distance_must_cover_initial_separation() -> None:
    geometry = RotatingBlocksGeometry(search_distance=0.01)
    with pytest.raises(ValueError, match="search distance"):
        build_rotating_blocks_model("quick", geometry=geometry)
