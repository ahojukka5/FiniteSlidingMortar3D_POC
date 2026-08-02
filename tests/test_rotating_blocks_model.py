from __future__ import annotations

import numpy as np
import pytest

from benchmarks.rotating_blocks_model import (
    FULL_PROFILE,
    QUICK_PROFILE,
    RotatingBlocksGeometry,
    RotatingBlocksProfile,
    build_rotating_blocks_model,
    reference_determinants,
    rotating_blocks_profile,
)


def _controlled_dofs(nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [3 * int(node) + component for node in nodes for component in range(3)],
        dtype=np.int64,
    )


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
    assert interface.pair.master.node_count == 25
    assert len(interface.pair.slave.facets) == 6
    assert len(interface.pair.master.facets) == 16
    assert interface.pair.search_distance > model.initial_separation > 0.0


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
    expected_dofs = np.concatenate([fixed_dofs, controlled_dofs])
    np.testing.assert_array_equal(constrained, expected_dofs)
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


def test_profile_lookup_and_intermediate_profile_construction() -> None:
    assert rotating_blocks_profile("quick") is QUICK_PROFILE
    assert rotating_blocks_profile("full") is FULL_PROFILE
    with pytest.raises(ValueError, match="profile must be"):
        rotating_blocks_profile("unknown")

    intermediate_profile = RotatingBlocksProfile(
        "quick",
        (2, 2, 1),
        (1, 1, 1),
    )
    intermediate = build_rotating_blocks_model(intermediate_profile)
    assert intermediate.profile is intermediate_profile
    assert intermediate.problem.mesh.element_count == 30
    assert len(intermediate.problem.interfaces[0].pair.slave.facets) == 1
    assert len(intermediate.problem.interfaces[0].pair.master.facets) == 4


def test_search_distance_must_cover_initial_separation() -> None:
    geometry = RotatingBlocksGeometry(search_distance=0.01)
    with pytest.raises(ValueError, match="search distance"):
        build_rotating_blocks_model("quick", geometry=geometry)
