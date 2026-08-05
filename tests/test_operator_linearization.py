from __future__ import annotations

import numpy as np

from contact3d.geometry import ContactSurface
from contact3d.mortar import (
    ContactPair,
    analytical_mortar_weight_jacobian,
    integrate_facet_pair,
    integrate_facet_pair_linearized,
    moving_mortar_contact_tangent,
    numerical_contact_tangent,
    numerical_mortar_weight_jacobian,
)


def _warped_facets() -> tuple[np.ndarray, np.ndarray]:
    slave = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.05, 0.03],
            [1.1, 1.05, -0.02],
            [-0.05, 0.95, 0.04],
        ]
    )
    master = np.array(
        [
            [0.18, -0.12, -0.13],
            [1.35, -0.05, -0.08],
            [1.28, 0.86, -0.11],
            [0.12, 0.92, -0.16],
        ]
    )
    return slave, master


def _pair() -> ContactPair:
    slave, master = _warped_facets()
    facet = (np.array([0, 1, 2, 3]),)
    return ContactPair(
        ContactSurface(slave, facet),
        ContactSurface(master, facet),
        normal_penalty=37.0,
        search_distance=0.5,
    )


def test_local_operator_jacobian_matches_centered_difference() -> None:
    slave, master = _warped_facets()
    analytical = integrate_facet_pair_linearized(slave, master)
    base = integrate_facet_pair(slave, master)

    np.testing.assert_allclose(analytical.d, base.d, atol=2.0e-12)
    np.testing.assert_allclose(analytical.m, base.m, atol=2.0e-12)

    total_dofs = 3 * (len(slave) + len(master))
    numerical_d = np.zeros_like(analytical.d_jacobian)
    numerical_m = np.zeros_like(analytical.m_jacobian)
    coordinates = np.vstack([slave, master])
    for column in range(total_dofs):
        node, component = divmod(column, 3)
        step = 4.0e-7 * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave.copy()
        minus_slave = slave.copy()
        plus_master = master.copy()
        minus_master = master.copy()
        if node < len(slave):
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - len(slave)
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step
        plus = integrate_facet_pair(plus_slave, plus_master)
        minus = integrate_facet_pair(minus_slave, minus_master)
        numerical_d[:, :, column] = (plus.d - minus.d) / (2.0 * step)
        numerical_m[:, :, column] = (plus.m - minus.m) / (2.0 * step)

    assert np.max(np.abs(analytical.d_jacobian - numerical_d)) < 2.0e-6
    assert np.max(np.abs(analytical.m_jacobian - numerical_m)) < 2.0e-6
    assert analytical.consistency_error < 2.0e-12
    assert analytical.consistency_jacobian_error < 2.0e-10
    assert analytical.area_consistency_error < 2.0e-12
    assert analytical.area_jacobian_consistency_error < 2.0e-10


def test_global_operator_jacobian_matches_numerical_oracle() -> None:
    pair = _pair()
    analytical = analytical_mortar_weight_jacobian(pair)
    numerical = numerical_mortar_weight_jacobian(pair, relative_step=5.0e-7)

    d_error = np.linalg.norm(analytical.d - numerical.d) / np.linalg.norm(numerical.d)
    m_error = np.linalg.norm(analytical.m - numerical.m) / np.linalg.norm(numerical.m)
    assert d_error < 2.0e-5
    assert m_error < 2.0e-5
    assert analytical.value_consistency_error < 2.0e-11
    assert analytical.consistency_error < 2.0e-9
    assert np.max(
        np.abs(np.sum(analytical.d, axis=(0, 1)) - analytical.total_area_jacobian)
    ) < 2.0e-9
    assert np.max(
        np.abs(np.sum(analytical.m, axis=(0, 1)) - analytical.total_area_jacobian)
    ) < 2.0e-9


def test_analytical_operator_jacobian_has_common_translation_nullspace() -> None:
    pair = _pair()
    jacobian = analytical_mortar_weight_jacobian(pair)
    total_nodes = pair.slave.node_count + pair.master.node_count

    for component in range(3):
        direction = np.zeros(3 * total_nodes)
        direction[component::3] = 1.0
        derivative_d = np.tensordot(jacobian.d, direction, axes=(2, 0))
        derivative_m = np.tensordot(jacobian.m, direction, axes=(2, 0))
        assert np.linalg.norm(derivative_d) < 2.0e-9
        assert np.linalg.norm(derivative_m) < 2.0e-9


def test_analytical_moving_tangent_matches_residual_oracle() -> None:
    pair = _pair()
    analytical = moving_mortar_contact_tangent(pair)
    numerical = numerical_contact_tangent(
        pair,
        relative_step=5.0e-7,
        freeze_facet_pairs=True,
        freeze_active_rows=True,
    )
    relative_error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)

    assert relative_error < 5.0e-5


def test_numerical_geometry_mode_remains_available() -> None:
    pair = _pair()
    analytical = moving_mortar_contact_tangent(pair)
    numerical_geometry = moving_mortar_contact_tangent(
        pair,
        geometry_jacobian="numerical",
        relative_step=5.0e-7,
    )
    relative_error = np.linalg.norm(analytical - numerical_geometry) / np.linalg.norm(
        numerical_geometry
    )

    assert relative_error < 5.0e-5
