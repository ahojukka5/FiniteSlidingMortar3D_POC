"""Fixed-operator and numerical frictionless contact tangents."""

from __future__ import annotations

import numpy as np

from ..geometry import FacetPair, FloatArray, averaged_nodal_normal_jacobian
from .model import ContactPair
from .residual import evaluate_contact


def fixed_mortar_contact_tangent(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    tolerance: float = 1.0e-12,
) -> FloatArray:
    """Return the analytical penalty tangent with D/M and activity frozen."""

    base = evaluate_contact(
        pair,
        slave_displacement,
        master_displacement,
        facet_pairs=facet_pairs,
        active_rows=active_rows,
        tolerance=tolerance,
    )
    slave_count = pair.slave.node_count
    master_count = pair.master.node_count
    ndof = 3 * (slave_count + master_count)
    dmat = base.weights.d
    mmat = base.weights.m
    row_areas = base.weights.row_areas
    supported = row_areas > tolerance

    derivative_normals = np.zeros((slave_count, 3, ndof), dtype=float)
    derivative_normals[:, :, : 3 * slave_count] = averaged_nodal_normal_jacobian(
        pair.slave,
        base.slave_nodes,
    ).reshape((slave_count, 3, 3 * slave_count))

    derivative_gap_vectors = np.zeros((slave_count, 3, ndof), dtype=float)
    for component in range(3):
        derivative_gap_vectors[:, component, component : 3 * slave_count : 3] = dmat
        derivative_gap_vectors[
            :,
            component,
            3 * slave_count + component : ndof : 3,
        ] = -mmat

    derivative_normal_gaps = np.zeros((slave_count, ndof), dtype=float)
    numerator = np.einsum(
        "icq,ic->iq",
        derivative_normals,
        base.weighted_gap_vectors,
    ) + np.einsum(
        "ic,icq->iq",
        base.nodal_normals,
        derivative_gap_vectors,
    )
    derivative_normal_gaps[supported] = numerator[supported] / row_areas[
        supported,
        None,
    ]
    derivative_pressure = (
        pair.normal_penalty * base.active_rows[:, None] * derivative_normal_gaps
    )
    derivative_traction = (
        derivative_pressure[:, None, :] * base.nodal_normals[:, :, None]
        + base.pressure[:, None, None] * derivative_normals
    )

    derivative_slave_force = np.einsum("ji,jcq->icq", dmat, derivative_traction)
    derivative_master_force = -np.einsum("ji,jcq->icq", mmat, derivative_traction)
    return np.concatenate(
        [
            derivative_slave_force.reshape((3 * slave_count, ndof)),
            derivative_master_force.reshape((3 * master_count, ndof)),
        ],
        axis=0,
    )


def numerical_contact_tangent(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    relative_step: float = 2.0e-7,
    freeze_facet_pairs: bool = True,
    freeze_active_rows: bool = True,
    freeze_weights: bool = False,
) -> FloatArray:
    """Return a dense centered-difference tangent for verification models."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    slave_u = (
        np.zeros_like(pair.slave.reference_nodes)
        if slave_displacement is None
        else np.asarray(slave_displacement, dtype=float)
        .reshape(pair.slave.reference_nodes.shape)
        .copy()
    )
    master_u = (
        np.zeros_like(pair.master.reference_nodes)
        if master_displacement is None
        else np.asarray(master_displacement, dtype=float)
        .reshape(pair.master.reference_nodes.shape)
        .copy()
    )
    base = evaluate_contact(pair, slave_u, master_u)
    frozen_weights = base.weights if freeze_weights else None
    frozen_pairs = (
        base.weights.facet_pairs
        if freeze_facet_pairs and not freeze_weights
        else None
    )
    frozen_active = base.active_rows if freeze_active_rows else None
    coordinates = np.vstack(
        [pair.slave.reference_nodes + slave_u, pair.master.reference_nodes + master_u]
    )
    ndof = 3 * (pair.slave.node_count + pair.master.node_count)
    tangent = np.zeros((ndof, ndof), dtype=float)

    for column in range(ndof):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave_u.copy()
        minus_slave = slave_u.copy()
        plus_master = master_u.copy()
        minus_master = master_u.copy()
        if node < pair.slave.node_count:
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - pair.slave.node_count
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step
        plus = evaluate_contact(
            pair,
            plus_slave,
            plus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
            frozen_weights=frozen_weights,
        ).residual
        minus = evaluate_contact(
            pair,
            minus_slave,
            minus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
            frozen_weights=frozen_weights,
        ).residual
        tangent[:, column] = (plus - minus) / (2.0 * step)
    return tangent
