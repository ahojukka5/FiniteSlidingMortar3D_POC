from __future__ import annotations

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    evaluate_contact,
    fixed_mortar_contact_tangent,
    moving_mortar_contact_tangent,
    numerical_contact_tangent,
    numerical_mortar_weight_jacobian,
)


def _pair() -> ContactPair:
    slave_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.05, 0.03],
            [1.1, 1.05, -0.02],
            [-0.05, 0.95, 0.04],
        ]
    )
    master_nodes = np.array(
        [
            [0.18, -0.12, -0.13],
            [1.35, -0.05, -0.08],
            [1.28, 0.86, -0.11],
            [0.12, 0.92, -0.16],
        ]
    )
    facet = (np.array([0, 1, 2, 3]),)
    return ContactPair(
        ContactSurface(slave_nodes, facet),
        ContactSurface(master_nodes, facet),
        normal_penalty=37.0,
        search_distance=0.5,
    )


def test_weight_jacobian_preserves_partition_of_unity_derivative() -> None:
    jacobian = numerical_mortar_weight_jacobian(_pair(), relative_step=8.0e-7)

    assert jacobian.consistency_error < 2.0e-9


def test_weight_jacobian_has_common_translation_nullspace() -> None:
    pair = _pair()
    jacobian = numerical_mortar_weight_jacobian(pair, relative_step=8.0e-7)
    ndof = 3 * (pair.slave.node_count + pair.master.node_count)

    for component in range(3):
        direction = np.zeros(ndof)
        direction[component::3] = 1.0
        derivative_d = np.tensordot(jacobian.d, direction, axes=(2, 0))
        derivative_m = np.tensordot(jacobian.m, direction, axes=(2, 0))
        assert np.linalg.norm(derivative_d) < 2.0e-8
        assert np.linalg.norm(derivative_m) < 2.0e-8


def test_moving_tangent_matches_full_numerical_oracle() -> None:
    pair = _pair()
    base = evaluate_contact(pair)
    assert np.all(base.active_rows)

    analytical_law_with_geometry_oracle = moving_mortar_contact_tangent(
        pair,
        relative_step=5.0e-7,
    )
    numerical = numerical_contact_tangent(
        pair,
        relative_step=5.0e-7,
        freeze_facet_pairs=True,
        freeze_active_rows=True,
    )
    relative_error = np.linalg.norm(
        analytical_law_with_geometry_oracle - numerical
    ) / np.linalg.norm(numerical)

    assert relative_error < 3.0e-6


def test_moving_overlap_contribution_is_nonzero_for_partial_overlap() -> None:
    pair = _pair()
    fixed = fixed_mortar_contact_tangent(pair)
    moving = moving_mortar_contact_tangent(pair, relative_step=5.0e-7)

    assert np.linalg.norm(moving - fixed) > 1.0e-3


def test_moving_tangent_retains_common_translation_nullspace() -> None:
    pair = _pair()
    tangent = moving_mortar_contact_tangent(pair, relative_step=5.0e-7)
    direction = np.tile(
        [0.4, -0.7, 0.2],
        pair.slave.node_count + pair.master.node_count,
    )

    assert np.linalg.norm(tangent @ direction) < 2.0e-6
