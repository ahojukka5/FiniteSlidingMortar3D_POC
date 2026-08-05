"""Analytical and numerical moving-overlap contact tangent assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..geometry import FacetPair, FloatArray, averaged_nodal_normal_jacobian
from .model import ContactPair
from .operators import assemble_mortar_weights, integrate_facet_pair_linearized
from .residual import evaluate_contact


@dataclass(frozen=True, slots=True)
class MortarWeightJacobian:
    """Derivatives of global standard-mortar operators.

    ``d`` has shape ``(slave_nodes, slave_nodes, total_dofs)`` and ``m`` has
    shape ``(slave_nodes, master_nodes, total_dofs)``. When available,
    ``overlap_areas`` has shape ``(integrated_facet_pairs, total_dofs)``.
    """

    d: FloatArray
    m: FloatArray
    overlap_areas: FloatArray | None = None
    value_consistency_error: float = 0.0

    @property
    def row_area(self) -> FloatArray:
        """Derivative of each mortar row area with respect to all DOFs."""

        return np.sum(self.d, axis=1)

    @property
    def consistency_error(self) -> float:
        """Derivative-level partition-of-unity violation."""

        difference = np.sum(self.d, axis=1) - np.sum(self.m, axis=1)
        return float(np.max(np.abs(difference), initial=0.0))

    @property
    def total_area_jacobian(self) -> FloatArray:
        """Derivative of total integrated overlap area."""

        if self.overlap_areas is None:
            return np.sum(self.d, axis=(0, 1))
        return np.sum(self.overlap_areas, axis=0)


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
    facet-pair set is retained. This remains the independent centered-difference
    oracle for the analytical operator Jacobian.
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


def _global_column(
    local_column: int,
    slave_facet: np.ndarray,
    master_facet: np.ndarray,
    slave_node_count: int,
) -> int:
    slave_dofs = 3 * len(slave_facet)
    if local_column < slave_dofs:
        node, component = divmod(local_column, 3)
        return 3 * int(slave_facet[node]) + component
    node, component = divmod(local_column - slave_dofs, 3)
    return 3 * slave_node_count + 3 * int(master_facet[node]) + component


def analytical_mortar_weight_jacobian(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
    maximum_inverse_iterations: int = 25,
) -> MortarWeightJacobian:
    """Assemble analytical global ``D`` and ``M`` geometry Jacobians.

    Broad-phase candidates are evaluated once. The integrated facet-pair set is
    then frozen, and every pair is differentiated through projection, clipping,
    pallets, inverse maps, quadrature shape values, and integration weights.
    """

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

    slave_count = pair.slave.node_count
    master_count = pair.master.node_count
    ndof = 3 * (slave_count + master_count)
    derivative_d = np.zeros((slave_count, slave_count, ndof), dtype=float)
    derivative_m = np.zeros((slave_count, master_count, ndof), dtype=float)
    assembled_d = np.zeros_like(base.d)
    assembled_m = np.zeros_like(base.m)
    assembled_areas: list[float] = []
    derivative_areas: list[FloatArray] = []

    for slave_index, master_index in base.facet_pairs:
        slave_facet = pair.slave.facets[slave_index]
        master_facet = pair.master.facets[master_index]
        local = integrate_facet_pair_linearized(
            slave_nodes[slave_facet],
            master_nodes[master_facet],
            quadrature_points=pair.quadrature_points,
            tolerance=tolerance,
            event_tolerance=event_tolerance,
            maximum_inverse_iterations=maximum_inverse_iterations,
        )
        assembled_d[np.ix_(slave_facet, slave_facet)] += local.d
        assembled_m[np.ix_(slave_facet, master_facet)] += local.m

        local_area_jacobian = local.quadrature.geometry.fan.total_area_jacobian
        global_area_jacobian = np.zeros(ndof, dtype=float)
        for local_column in range(local.d_jacobian.shape[2]):
            global_column = _global_column(
                local_column,
                slave_facet,
                master_facet,
                slave_count,
            )
            global_area_jacobian[global_column] += local_area_jacobian[local_column]
            for local_row, global_row in enumerate(slave_facet):
                for local_column_node, global_column_node in enumerate(slave_facet):
                    derivative_d[
                        global_row,
                        global_column_node,
                        global_column,
                    ] += local.d_jacobian[
                        local_row,
                        local_column_node,
                        local_column,
                    ]
                for local_master, global_master in enumerate(master_facet):
                    derivative_m[
                        global_row,
                        global_master,
                        global_column,
                    ] += local.m_jacobian[
                        local_row,
                        local_master,
                        local_column,
                    ]
        assembled_areas.append(local.quadrature.geometry.fan.total_area)
        derivative_areas.append(global_area_jacobian)

    area_values = np.asarray(assembled_areas, dtype=float)
    area_jacobian = (
        np.asarray(derivative_areas, dtype=float)
        if derivative_areas
        else np.empty((0, ndof), dtype=float)
    )
    value_error = max(
        float(np.max(np.abs(assembled_d - base.d), initial=0.0)),
        float(np.max(np.abs(assembled_m - base.m), initial=0.0)),
        float(np.max(np.abs(area_values - base.overlap_areas), initial=0.0)),
    )
    return MortarWeightJacobian(
        derivative_d,
        derivative_m,
        overlap_areas=area_jacobian,
        value_consistency_error=value_error,
    )


def moving_mortar_contact_tangent(
    pair: ContactPair,
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
    """Return the complete smooth residual tangent.

    The default path uses the analytical moving-overlap operator Jacobian. Set
    ``geometry_jacobian="numerical"`` to retain the centered-difference geometry
    oracle for verification. Facet pairs and unilateral activity are frozen in
    either mode.
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
