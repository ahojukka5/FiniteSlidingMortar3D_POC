"""Analytical augmented-Lagrange smooth-branch contact tangent."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...geometry import FacetPair, FloatArray, averaged_nodal_normal_jacobian
from ..model import ContactEvaluation, ContactPair
from ..moving import (
    MortarWeightJacobian,
    analytical_mortar_weight_jacobian,
    numerical_mortar_weight_jacobian,
)
from .evaluation import evaluate_augmented_lagrange
from .state import AugmentedLagrangeState


def _assemble_contact_tangent(
    pair: ContactPair,
    base: ContactEvaluation,
    weight_jacobian: MortarWeightJacobian,
    *,
    tolerance: float,
) -> FloatArray:
    """Assemble a contact tangent from an evaluated pressure state."""

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

    derivative_gap_vectors = np.einsum(
        "ijq,jc->icq", weight_jacobian.d, base.slave_nodes
    ) - np.einsum("ijq,jc->icq", weight_jacobian.m, base.master_nodes)
    for component in range(3):
        derivative_gap_vectors[
            :, component, component : 3 * slave_count : 3
        ] += dmat
        derivative_gap_vectors[
            :, component, 3 * slave_count + component : ndof : 3
        ] -= mmat

    derivative_weighted_gap = np.einsum(
        "icq,ic->iq", derivative_normals, base.weighted_gap_vectors
    ) + np.einsum("ic,icq->iq", base.nodal_normals, derivative_gap_vectors)
    derivative_area = weight_jacobian.row_area
    derivative_gap = np.zeros((slave_count, ndof), dtype=float)
    derivative_gap[supported] = (
        derivative_weighted_gap[supported] * row_areas[supported, None]
        - base.weighted_normal_gaps[supported, None] * derivative_area[supported]
    ) / row_areas[supported, None] ** 2

    derivative_pressure = (
        pair.normal_penalty * base.active_rows[:, None] * derivative_gap
    )
    derivative_traction = (
        derivative_pressure[:, None, :] * base.nodal_normals[:, :, None]
        + base.pressure[:, None, None] * derivative_normals
    )
    traction = base.pressure[:, None] * base.nodal_normals

    derivative_slave_force = np.einsum(
        "ji,jcq->icq", dmat, derivative_traction
    ) + np.einsum("jiq,jc->icq", weight_jacobian.d, traction)
    derivative_master_force = -np.einsum(
        "ji,jcq->icq", mmat, derivative_traction
    ) - np.einsum("jiq,jc->icq", weight_jacobian.m, traction)
    return np.concatenate(
        [
            derivative_slave_force.reshape((3 * slave_count, ndof)),
            derivative_master_force.reshape((3 * master_count, ndof)),
        ],
        axis=0,
    )


def augmented_lagrange_contact_tangent(
    pair: ContactPair,
    state: AugmentedLagrangeState,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    geometry_jacobian: Literal["analytical", "numerical"] = "analytical",
    relative_step: float = 2.0e-7,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
    maximum_inverse_iterations: int = 25,
) -> FloatArray:
    """Return the smooth inner-Newton tangent with multipliers fixed."""

    state.validate_for(pair)
    base = evaluate_augmented_lagrange(
        pair,
        state,
        slave_displacement,
        master_displacement,
        facet_pairs=facet_pairs,
        active_rows=active_rows,
        tolerance=tolerance,
    ).contact
    if geometry_jacobian == "analytical":
        weight_jacobian = analytical_mortar_weight_jacobian(
            pair,
            slave_displacement,
            master_displacement,
            facet_pairs=base.weights.facet_pairs,
            tolerance=tolerance,
            event_tolerance=event_tolerance,
            maximum_inverse_iterations=maximum_inverse_iterations,
        )
    elif geometry_jacobian == "numerical":
        weight_jacobian = numerical_mortar_weight_jacobian(
            pair,
            slave_displacement,
            master_displacement,
            facet_pairs=base.weights.facet_pairs,
            relative_step=relative_step,
            tolerance=tolerance,
        )
    else:
        raise ValueError("geometry_jacobian must be 'analytical' or 'numerical'")
    return _assemble_contact_tangent(
        pair,
        base,
        weight_jacobian,
        tolerance=tolerance,
    )
