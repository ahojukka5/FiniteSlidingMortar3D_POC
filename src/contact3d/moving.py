"""Topology-frozen moving-overlap tangent verification.

This module differentiates the assembled mortar operators numerically while
keeping the current facet-pair set fixed.  It closes the smooth tangent of the
contact residual without pretending that the geometric derivative is already
analytical.  Later Section 4 / Appendix B derivatives can replace the numerical
operator columns one layer at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import ContactPair, assemble_mortar_weights, evaluate_contact
from .model import FloatArray
from .surface import FacetPair, averaged_nodal_normal_jacobian


@dataclass(frozen=True, slots=True)
class MortarWeightJacobian:
    """Derivatives of global standard-mortar operators.

    ``d`` has shape ``(slave_nodes, slave_nodes, total_dofs)`` and ``m`` has
    shape ``(slave_nodes, master_nodes, total_dofs)``.
    """

    d: FloatArray
    m: FloatArray

    @property
    def row_area(self) -> FloatArray:
        """Derivative of each mortar row area with respect to all DOFs."""

        return np.sum(self.d, axis=1)

    @property
    def consistency_error(self) -> float:
        """Derivative-level partition-of-unity violation."""

        difference = np.sum(self.d, axis=1) - np.sum(self.m, axis=1)
        return float(np.max(np.abs(difference), initial=0.0))


def _displacements(
    pair: ContactPair,
    slave_displacement: FloatArray | None,
    master_displacement: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    slave = (
        np.zeros_like(pair.slave.reference_nodes)
        if slave_displacement is None
        else np.asarray(slave_displacement, dtype=float)
        .reshape(pair.slave.reference_nodes.shape)
        .copy()
    )
    master = (
        np.zeros_like(pair.master.reference_nodes)
        if master_displacement is None
        else np.asarray(master_displacement, dtype=float)
        .reshape(pair.master.reference_nodes.shape)
        .copy()
    )
    return slave, master


def numerical_mortar_weight_jacobian(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    relative_step: float = 2.0e-7,
    tolerance: float = 1.0e-12,
) -> MortarWeightJacobian:
    """Differentiate assembled ``D`` and ``M`` with fixed facet candidates.

    The overlap polygon is rebuilt for every perturbation, but the broad-phase
    facet-pair set is retained.  This is the geometric analogue of freezing the
    unilateral active set during one generalized Newton derivative.
    """

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")

    slave_u, master_u = _displacements(pair, slave_displacement, master_displacement)
    slave_nodes = pair.slave.reference_nodes + slave_u
    master_nodes = pair.master.reference_nodes + master_u
    base = assemble_mortar_weights(
        pair,
        slave_nodes,
        master_nodes,
        facet_pairs=facet_pairs,
        tolerance=tolerance,
    )
    frozen_pairs = base.facet_pairs if facet_pairs is None else tuple(facet_pairs)

    slave_count = pair.slave.node_count
    master_count = pair.master.node_count
    ndof = 3 * (slave_count + master_count)
    derivative_d = np.zeros((slave_count, slave_count, ndof), dtype=float)
    derivative_m = np.zeros((slave_count, master_count, ndof), dtype=float)
    coordinates = np.vstack([slave_nodes, master_nodes])

    for column in range(ndof):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave_u.copy()
        minus_slave = slave_u.copy()
        plus_master = master_u.copy()
        minus_master = master_u.copy()

        if node < slave_count:
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - slave_count
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step

        plus = assemble_mortar_weights(
            pair,
            pair.slave.reference_nodes + plus_slave,
            pair.master.reference_nodes + plus_master,
            facet_pairs=frozen_pairs,
            tolerance=tolerance,
        )
        minus = assemble_mortar_weights(
            pair,
            pair.slave.reference_nodes + minus_slave,
            pair.master.reference_nodes + minus_master,
            facet_pairs=frozen_pairs,
            tolerance=tolerance,
        )
        derivative_d[:, :, column] = (plus.d - minus.d) / (2.0 * step)
        derivative_m[:, :, column] = (plus.m - minus.m) / (2.0 * step)

    return MortarWeightJacobian(derivative_d, derivative_m)


def moving_mortar_contact_tangent(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    relative_step: float = 2.0e-7,
    tolerance: float = 1.0e-12,
) -> FloatArray:
    """Return the complete smooth residual tangent for verification models.

    Appendix A normal derivatives and the contact law are analytical.  The
    moving-overlap part enters through a topology-frozen numerical Jacobian of
    ``D`` and ``M``.  Thus this function is a decomposition oracle for the later
    fully analytical geometric tangent, not the final production implementation.
    """

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

    weight_jacobian = numerical_mortar_weight_jacobian(
        pair,
        slave_displacement,
        master_displacement,
        facet_pairs=base.weights.facet_pairs,
        relative_step=relative_step,
        tolerance=tolerance,
    )

    derivative_normals = np.zeros((slave_count, 3, ndof), dtype=float)
    derivative_normals[:, :, : 3 * slave_count] = averaged_nodal_normal_jacobian(
        pair.slave,
        base.slave_nodes,
    ).reshape((slave_count, 3, 3 * slave_count))

    derivative_gap_vectors = np.einsum(
        "ijq,jc->icq",
        weight_jacobian.d,
        base.slave_nodes,
    ) - np.einsum(
        "ijq,jc->icq",
        weight_jacobian.m,
        base.master_nodes,
    )
    for component in range(3):
        derivative_gap_vectors[
            :, component, component : 3 * slave_count : 3
        ] += dmat
        derivative_gap_vectors[
            :, component, 3 * slave_count + component : ndof : 3
        ] -= mmat

    derivative_weighted_normal_gap = np.einsum(
        "icq,ic->iq",
        derivative_normals,
        base.weighted_gap_vectors,
    ) + np.einsum(
        "ic,icq->iq",
        base.nodal_normals,
        derivative_gap_vectors,
    )
    derivative_row_area = weight_jacobian.row_area
    derivative_normal_gap = np.zeros((slave_count, ndof), dtype=float)
    derivative_normal_gap[supported] = (
        derivative_weighted_normal_gap[supported] * row_areas[supported, None]
        - base.weighted_normal_gaps[supported, None]
        * derivative_row_area[supported]
    ) / row_areas[supported, None] ** 2

    derivative_pressure = (
        pair.normal_penalty * base.active_rows[:, None] * derivative_normal_gap
    )
    derivative_traction = (
        derivative_pressure[:, None, :] * base.nodal_normals[:, :, None]
        + base.pressure[:, None, None] * derivative_normals
    )
    traction = base.pressure[:, None] * base.nodal_normals

    derivative_slave_force = np.einsum(
        "ji,jcq->icq",
        dmat,
        derivative_traction,
    ) + np.einsum(
        "jiq,jc->icq",
        weight_jacobian.d,
        traction,
    )
    derivative_master_force = -np.einsum(
        "ji,jcq->icq",
        mmat,
        derivative_traction,
    ) - np.einsum(
        "jiq,jc->icq",
        weight_jacobian.m,
        traction,
    )

    return np.concatenate(
        [
            derivative_slave_force.reshape((3 * slave_count, ndof)),
            derivative_master_force.reshape((3 * master_count, ndof)),
        ],
        axis=0,
    )
