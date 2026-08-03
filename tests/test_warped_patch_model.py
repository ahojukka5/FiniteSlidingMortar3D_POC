from __future__ import annotations

import importlib.util
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


MODEL = _load_module("warped_patch_model", "warped_patch_model.py")


def test_refinement_profiles_are_nonmatching_and_ordered() -> None:
    profiles = tuple(
        MODEL.PROFILES[name] for name in ("coarse", "medium", "fine")
    )

    assert all(
        profile.lower_cells[:2] != profile.upper_cells[:2]
        for profile in profiles
    )
    assert all(
        left.characteristic_size > right.characteristic_size
        for left, right in zip(profiles[:-1], profiles[1:], strict=True)
    )
    assert all(
        left.lower_cells[0] < right.lower_cells[0]
        and left.upper_cells[0] < right.upper_cells[0]
        for left, right in zip(profiles[:-1], profiles[1:], strict=True)
    )


def test_every_surface_family_and_bias_builds_a_valid_model() -> None:
    for family_name, family in MODEL.SURFACE_FAMILIES.items():
        for bias_side in ("lower", "upper"):
            model = MODEL.build_warped_patch_model(
                "coarse",
                surface_family=family_name,
                bias_side=bias_side,
            )
            interface = model.interface
            expected_slave_kind = (
                family.lower_kind if bias_side == "lower" else family.upper_kind
            )
            expected_master_kind = (
                family.upper_kind if bias_side == "lower" else family.lower_kind
            )

            assert model.minimum_reference_determinant > 0.0
            assert model.minimum_reference_separation > 0.0
            assert len(model.lower_interface_nodes) != len(
                model.upper_interface_nodes
            )
            assert all(
                len(facet) == (3 if expected_slave_kind == "tri3" else 4)
                for facet in interface.pair.slave.facets
            )
            assert all(
                len(facet) == (3 if expected_master_kind == "tri3" else 4)
                for facet in interface.pair.master.facets
            )
            if bias_side == "lower":
                np.testing.assert_array_equal(
                    interface.slave_nodes,
                    model.lower_interface_nodes,
                )
                np.testing.assert_array_equal(
                    interface.master_nodes,
                    model.upper_interface_nodes,
                )
            else:
                np.testing.assert_array_equal(
                    interface.slave_nodes,
                    model.upper_interface_nodes,
                )
                np.testing.assert_array_equal(
                    interface.master_nodes,
                    model.lower_interface_nodes,
                )


def test_warp_decreases_but_remains_nonplanar_at_every_level() -> None:
    models = tuple(
        MODEL.build_warped_patch_model(name)
        for name in ("coarse", "medium", "fine")
    )

    assert all(
        left.warp_amplitude > right.warp_amplitude
        for left, right in zip(models[:-1], models[1:], strict=True)
    )
    for model in models:
        lower_z = model.problem.mesh.reference_nodes[
            model.lower_interface_nodes,
            2,
        ]
        upper_z = model.problem.mesh.reference_nodes[
            model.upper_interface_nodes,
            2,
        ]
        assert np.ptp(lower_z) > 0.0
        assert np.ptp(upper_z) > 0.0
        assert model.warp_amplitude > 0.0


def test_manufactured_field_satisfies_constraints_and_closes_surface() -> None:
    model = MODEL.build_warped_patch_model(
        "medium",
        surface_family="tri-quad",
        bias_side="lower",
    )
    displacement = MODEL.manufactured_displacement(model)
    constraints = model.problem.constraints

    np.testing.assert_allclose(
        displacement[constraints.dofs],
        constraints.values,
        atol=2.0e-15,
    )
    for x, y in ((0.0, 0.0), (0.23, 0.41), (0.5, 0.5), (0.88, 0.17)):
        np.testing.assert_allclose(
            MODEL.analytical_deformed_gap(model, x, y),
            0.0,
            atol=2.0e-14,
        )


def test_flat_limit_pressure_is_positive_and_profile_independent() -> None:
    models = tuple(
        MODEL.build_warped_patch_model(name, bias_side="upper")
        for name in ("coarse", "medium", "fine")
    )
    pressures = np.asarray([MODEL.reference_pressure(model) for model in models])
    reactions = np.asarray(
        [MODEL.reference_vertical_reaction(model) for model in models]
    )

    assert np.all(pressures > 0.0)
    np.testing.assert_allclose(pressures, pressures[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reactions, pressures, rtol=0.0, atol=0.0)
    assert all(model.axial_strain == models[0].axial_strain for model in models)
