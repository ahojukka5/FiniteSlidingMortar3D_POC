"""Analytical local standard-mortar operators and their geometry Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray
from .parametric import FacetQuadratureLinearization, linearize_facet_quadrature


@dataclass(frozen=True, slots=True)
class LocalMortarWeightLinearization:
    """Local ``D``/``M`` operators and derivatives for one facet pair.

    ``d_jacobian`` has axes ``(slave_node, slave_node, local_dof)`` and
    ``m_jacobian`` has axes ``(slave_node, master_node, local_dof)``. Local
    coordinate columns are ordered as all slave coordinates followed by all
    master coordinates, each in node-major xyz order.
    """

    d: FloatArray
    m: FloatArray
    d_jacobian: FloatArray
    m_jacobian: FloatArray
    quadrature: FacetQuadratureLinearization

    @property
    def consistency_error(self) -> float:
        """Value-level row partition-of-unity violation."""

        difference = np.sum(self.d, axis=1) - np.sum(self.m, axis=1)
        return float(np.max(np.abs(difference), initial=0.0))

    @property
    def consistency_jacobian_error(self) -> float:
        """Derivative-level row partition-of-unity violation."""

        difference = np.sum(self.d_jacobian, axis=1) - np.sum(
            self.m_jacobian,
            axis=1,
        )
        return float(np.max(np.abs(difference), initial=0.0))

    @property
    def area_consistency_error(self) -> float:
        """Difference between integrated operators and geometric overlap area."""

        area = self.quadrature.geometry.fan.total_area
        return max(abs(float(np.sum(self.d)) - area), abs(float(np.sum(self.m)) - area))

    @property
    def area_jacobian_consistency_error(self) -> float:
        """Derivative-level operator/overlap-area consistency error."""

        area_jacobian = self.quadrature.geometry.fan.total_area_jacobian
        return max(
            float(
                np.max(
                    np.abs(np.sum(self.d_jacobian, axis=(0, 1)) - area_jacobian),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(np.sum(self.m_jacobian, axis=(0, 1)) - area_jacobian),
                    initial=0.0,
                )
            ),
        )


def integrate_facet_pair_linearized(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    quadrature_points: int = 7,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
    maximum_inverse_iterations: int = 25,
) -> LocalMortarWeightLinearization:
    """Integrate local standard-mortar operators and their analytical Jacobians.

    At each common physical quadrature point,

    ``D += w * outer(Ns, Ns)`` and ``M += w * outer(Ns, Nm)``.

    The derivative includes the moving integration weight and both shape fields.
    Projection, clipping, pallet, inverse-map, and shape-value derivatives are
    supplied by :func:`linearize_facet_quadrature`.
    """

    slave = np.asarray(slave_points, dtype=float)
    master = np.asarray(master_points, dtype=float)
    if slave.ndim != 2 or slave.shape[1] != 3:
        raise ValueError("slave facet coordinates must have shape (node_count, 3)")
    if master.ndim != 2 or master.shape[1] != 3:
        raise ValueError("master facet coordinates must have shape (node_count, 3)")

    quadrature = linearize_facet_quadrature(
        slave,
        master,
        quadrature_points=quadrature_points,
        tolerance=tolerance,
        event_tolerance=event_tolerance,
        maximum_inverse_iterations=maximum_inverse_iterations,
    )
    slave_count = len(slave)
    master_count = len(master)
    local_dofs = 3 * (slave_count + master_count)
    dmat = np.zeros((slave_count, slave_count), dtype=float)
    mmat = np.zeros((slave_count, master_count), dtype=float)
    derivative_d = np.zeros((slave_count, slave_count, local_dofs), dtype=float)
    derivative_m = np.zeros((slave_count, master_count, local_dofs), dtype=float)

    for point in quadrature.points:
        slave_shape = point.slave.shape
        master_shape = point.master.shape
        derivative_slave_shape = point.slave.shape_jacobian
        derivative_master_shape = point.master.shape_jacobian
        weight = point.integration_weight
        derivative_weight = point.integration_weight_jacobian

        slave_product = np.outer(slave_shape, slave_shape)
        mixed_product = np.outer(slave_shape, master_shape)
        dmat += weight * slave_product
        mmat += weight * mixed_product

        derivative_d += slave_product[:, :, None] * derivative_weight[None, None, :]
        derivative_d += weight * (
            np.einsum("iq,j->ijq", derivative_slave_shape, slave_shape)
            + np.einsum("i,jq->ijq", slave_shape, derivative_slave_shape)
        )
        derivative_m += mixed_product[:, :, None] * derivative_weight[None, None, :]
        derivative_m += weight * (
            np.einsum("iq,j->ijq", derivative_slave_shape, master_shape)
            + np.einsum("i,jq->ijq", slave_shape, derivative_master_shape)
        )

    return LocalMortarWeightLinearization(
        d=dmat,
        m=mmat,
        d_jacobian=derivative_d,
        m_jacobian=derivative_m,
        quadrature=quadrature,
    )
