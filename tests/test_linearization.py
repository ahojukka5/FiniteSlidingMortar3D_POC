from __future__ import annotations

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    averaged_nodal_normal_jacobian,
    averaged_nodal_normals,
    evaluate_contact,
    fixed_mortar_contact_tangent,
    numerical_contact_tangent,
)


def test_appendix_a_normal_jacobian_matches_centered_difference() -> None:
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    surface = ContactSurface(
        reference,
        (
            np.array([0, 1, 4, 3]),
            np.array([1, 2, 5, 4]),
        ),
    )
    current = reference + np.array(
        [
            [0.00, 0.00, 0.01],
            [0.03, -0.01, 0.05],
            [-0.02, 0.02, -0.02],
            [0.01, 0.02, 0.04],
            [-0.01, -0.03, 0.08],
            [0.02, 0.01, 0.03],
        ]
    )

    analytical = averaged_nodal_normal_jacobian(surface, current).reshape((18, 18))
    numerical = np.zeros_like(analytical)
    step = 1.0e-7
    for column in range(18):
        plus = current.copy().reshape(-1)
        minus = current.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (
            averaged_nodal_normals(surface, plus.reshape((-1, 3))).ravel()
            - averaged_nodal_normals(surface, minus.reshape((-1, 3))).ravel()
        ) / (2.0 * step)

    relative_error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)
    assert relative_error <= 2.0e-9


def _warped_contact_pair() -> ContactPair:
    slave_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    master_nodes = np.array(
        [
            [0.0, 0.0, -0.12],
            [1.0, 0.0, -0.10],
            [1.0, 1.0, -0.11],
            [0.0, 1.0, -0.13],
        ]
    )
    slave = ContactSurface(slave_nodes, (np.array([0, 1, 2, 3]),))
    master = ContactSurface(master_nodes, (np.array([0, 1, 2, 3]),))
    return ContactPair(slave, master, 200.0, search_distance=0.5)


def test_fixed_mortar_tangent_matches_frozen_weight_oracle() -> None:
    pair = _warped_contact_pair()
    slave_displacement = np.array(
        [
            [0.00, 0.00, 0.01],
            [0.01, 0.00, 0.04],
            [0.00, -0.01, 0.06],
            [-0.01, 0.01, 0.02],
        ]
    )
    master_displacement = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, -0.01],
            [0.00, 0.01, 0.00],
            [-0.01, 0.00, 0.01],
        ]
    )
    base = evaluate_contact(pair, slave_displacement, master_displacement)
    assert np.all(base.active_rows)

    analytical = fixed_mortar_contact_tangent(
        pair,
        slave_displacement,
        master_displacement,
    )
    numerical = numerical_contact_tangent(
        pair,
        slave_displacement,
        master_displacement,
        relative_step=5.0e-7,
        freeze_weights=True,
    )

    relative_error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)
    assert relative_error <= 2.0e-8


def test_fixed_mortar_tangent_has_common_translation_nullspace() -> None:
    pair = _warped_contact_pair()

    tangent = fixed_mortar_contact_tangent(pair)
    translation = np.tile(
        [0.4, -0.7, 0.2],
        pair.slave.node_count + pair.master.node_count,
    )

    assert np.linalg.norm(tangent @ translation) <= 1.0e-11
