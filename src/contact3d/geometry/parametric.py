"""Inverse parent maps and mortar quadrature analytical derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import facet_projection_plane_jacobian, project_to_plane_jacobian
from .model import FacetKind, FloatArray
from .pallets import FacetPalletLinearization, linearize_facet_pallets
from .quadrature import triangle_rule
from .shapes import (
    infer_facet_kind,
    inverse_map_2d,
    shape_gradients,
    shape_values,
)


class InverseMapTopologyError(ValueError):
    """Raised when a projected facet map is singular on the current smooth branch."""


@dataclass(frozen=True, slots=True)
class InverseMapLinearization:
    """Parent coordinates and shape values differentiated with respect to all DOFs.

    ``parent_jacobian`` has axes ``(parent_component, input_dof)`` and
    ``shape_jacobian`` has axes ``(facet_node, input_dof)``.
    """

    parent: FloatArray
    parent_jacobian: FloatArray
    shape: FloatArray
    shape_jacobian: FloatArray
    mapping_residual: float
    mapping_jacobian_residual: float

    @property
    def partition_error(self) -> float:
        """Violation of the shape-function partition of unity."""

        return abs(float(np.sum(self.shape)) - 1.0)

    @property
    def partition_jacobian_error(self) -> float:
        """Derivative-level partition-of-unity violation."""

        return float(
            np.max(np.abs(np.sum(self.shape_jacobian, axis=0)), initial=0.0)
        )


@dataclass(frozen=True, slots=True)
class MortarQuadraturePointLinearization:
    """One pallet quadrature point and both facet interpolation derivatives."""

    pallet_index: int
    quadrature_index: int
    barycentric: FloatArray
    rule_weight: float
    point: FloatArray
    point_jacobian: FloatArray
    integration_weight: float
    integration_weight_jacobian: FloatArray
    slave: InverseMapLinearization
    master: InverseMapLinearization


@dataclass(frozen=True, slots=True)
class FacetQuadratureLinearization:
    """Full projection, clipping, pallet, and quadrature derivative chain."""

    geometry: FacetPalletLinearization
    quadrature_point_count: int
    points: tuple[MortarQuadraturePointLinearization, ...]

    @property
    def integration_weight_sum(self) -> float:
        """Sum of all physical pallet quadrature weights."""

        return float(sum(point.integration_weight for point in self.points))

    @property
    def integration_weight_jacobian_sum(self) -> FloatArray:
        """Derivative of the summed physical quadrature weight."""

        answer = np.zeros_like(self.geometry.fan.total_area_jacobian)
        for point in self.points:
            answer += point.integration_weight_jacobian
        return answer

    @property
    def weight_consistency_error(self) -> float:
        """Difference between quadrature-weight sum and overlap area."""

        return abs(self.integration_weight_sum - self.geometry.fan.total_area)

    @property
    def weight_jacobian_consistency_error(self) -> float:
        """Derivative-level quadrature-area consistency error."""

        difference = (
            self.integration_weight_jacobian_sum
            - self.geometry.fan.total_area_jacobian
        )
        return float(np.max(np.abs(difference), initial=0.0))


def _validate_inverse_inputs(
    polygon: FloatArray,
    kind: FacetKind,
    point: FloatArray,
    polygon_jacobian: FloatArray,
    point_jacobian: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    coordinates = np.asarray(polygon, dtype=float)
    target = np.asarray(point, dtype=float)
    derivative_coordinates = np.asarray(polygon_jacobian, dtype=float)
    derivative_target = np.asarray(point_jacobian, dtype=float)

    expected_nodes = 3 if kind == "tri3" else 4 if kind == "quad4" else 0
    if expected_nodes == 0:
        raise ValueError("facet kind must be 'tri3' or 'quad4'")
    if coordinates.shape != (expected_nodes, 2):
        raise ValueError("projected polygon shape does not match the facet kind")
    if target.shape != (2,):
        raise ValueError("projected target point must have shape (2,)")
    if derivative_coordinates.ndim != 3 or derivative_coordinates.shape[:2] != (
        expected_nodes,
        2,
    ):
        raise ValueError(
            "polygon Jacobian must have shape (facet_node, 2, input_dof)"
        )
    if derivative_target.ndim != 2 or derivative_target.shape[0] != 2:
        raise ValueError("point Jacobian must have shape (2, input_dof)")
    if derivative_coordinates.shape[2] != derivative_target.shape[1]:
        raise ValueError("polygon and point Jacobians must share the same DOF count")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            coordinates,
            target,
            derivative_coordinates,
            derivative_target,
        )
    ):
        raise ValueError("inverse-map inputs must be finite")
    return coordinates, target, derivative_coordinates, derivative_target


def inverse_map_2d_linearized(
    polygon: FloatArray,
    kind: FacetKind,
    point: FloatArray,
    polygon_jacobian: FloatArray,
    point_jacobian: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 25,
) -> InverseMapLinearization:
    """Differentiate a TRI3 or QUAD4 inverse map by the implicit-function theorem.

    The parent point ``xi`` satisfies ``N(xi) @ polygon - point = 0``. On one
    smooth branch,

    ``d xi = J^-1 (d point - sum_i N_i d polygon_i)``,

    where ``J = polygon.T @ dN/dxi``.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    (
        coordinates,
        target,
        derivative_coordinates,
        derivative_target,
    ) = _validate_inverse_inputs(
        polygon,
        kind,
        point,
        polygon_jacobian,
        point_jacobian,
    )

    try:
        parent = inverse_map_2d(
            coordinates,
            kind,
            target,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
    except ValueError as error:
        raise InverseMapTopologyError(
            "projected facet inverse map is singular or did not converge"
        ) from error

    values = shape_values(kind, parent)
    gradients = shape_gradients(kind, parent)
    mapping_jacobian = coordinates.T @ gradients
    determinant = float(np.linalg.det(mapping_jacobian))
    if abs(determinant) <= tolerance:
        raise InverseMapTopologyError(
            "projected facet map has a singular parent Jacobian"
        )

    direct_mapping_derivative = np.einsum(
        "i,icq->cq",
        values,
        derivative_coordinates,
    )
    parent_jacobian = np.linalg.solve(
        mapping_jacobian,
        derivative_target - direct_mapping_derivative,
    )
    shape_jacobian = gradients @ parent_jacobian

    mapped_point = values @ coordinates
    mapped_point_jacobian = (
        coordinates.T @ shape_jacobian + direct_mapping_derivative
    )
    return InverseMapLinearization(
        parent=parent,
        parent_jacobian=parent_jacobian,
        shape=values,
        shape_jacobian=shape_jacobian,
        mapping_residual=float(np.linalg.norm(mapped_point - target)),
        mapping_jacobian_residual=float(
            np.max(
                np.abs(mapped_point_jacobian - derivative_target),
                initial=0.0,
            )
        ),
    )


def _projected_facet_jacobians(
    slave_points: FloatArray,
    master_points: FloatArray,
    geometry: FacetPalletLinearization,
) -> tuple[FacetKind, FacetKind, FloatArray, FloatArray]:
    slave = np.asarray(slave_points, dtype=float)
    master = np.asarray(master_points, dtype=float)
    slave_kind = infer_facet_kind(slave)
    master_kind = infer_facet_kind(master)
    plane = geometry.intersection.plane
    plane_jacobian = facet_projection_plane_jacobian(slave, slave_kind)

    slave_projected = project_to_plane_jacobian(
        slave,
        plane,
        plane_jacobian,
    ).combined_shared_coordinates()
    master_projected = project_to_plane_jacobian(
        master,
        plane,
        plane_jacobian,
    )

    slave_count = len(slave)
    master_count = len(master)
    total_dofs = 3 * (slave_count + master_count)
    slave_jacobian = np.zeros((slave_count, 2, total_dofs), dtype=float)
    master_jacobian = np.zeros((master_count, 2, total_dofs), dtype=float)
    slave_jacobian[:, :, : 3 * slave_count] = slave_projected.reshape(
        (slave_count, 2, 3 * slave_count)
    )
    master_jacobian[:, :, : 3 * slave_count] = master_projected.plane.reshape(
        (master_count, 2, 3 * slave_count)
    )
    master_jacobian[:, :, 3 * slave_count :] = master_projected.points.reshape(
        (master_count, 2, 3 * master_count)
    )
    return slave_kind, master_kind, slave_jacobian, master_jacobian


def linearize_facet_quadrature(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    quadrature_points: int = 7,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
    maximum_inverse_iterations: int = 25,
) -> FacetQuadratureLinearization:
    """Differentiate physical pallet points, inverse maps, and shape values."""

    geometry = linearize_facet_pallets(
        slave_points,
        master_points,
        tolerance=tolerance,
        event_tolerance=event_tolerance,
    )
    barycentric_points, rule_weights = triangle_rule(quadrature_points)
    if not geometry.fan.pallets:
        return FacetQuadratureLinearization(geometry, quadrature_points, ())

    (
        slave_kind,
        master_kind,
        slave_polygon_jacobian,
        master_polygon_jacobian,
    ) = _projected_facet_jacobians(
        slave_points,
        master_points,
        geometry,
    )
    slave_polygon = geometry.intersection.slave_polygon
    master_polygon = geometry.intersection.master_polygon

    points: list[MortarQuadraturePointLinearization] = []
    for pallet_index, pallet in enumerate(geometry.fan.pallets):
        for quadrature_index, (barycentric, rule_weight) in enumerate(
            zip(barycentric_points, rule_weights, strict=True)
        ):
            point = barycentric @ pallet.pallet.vertices
            point_jacobian = np.einsum(
                "i,icq->cq",
                barycentric,
                pallet.vertex_jacobian,
            )
            slave_map = inverse_map_2d_linearized(
                slave_polygon,
                slave_kind,
                point,
                slave_polygon_jacobian,
                point_jacobian,
                tolerance=tolerance,
                maximum_iterations=maximum_inverse_iterations,
            )
            master_map = inverse_map_2d_linearized(
                master_polygon,
                master_kind,
                point,
                master_polygon_jacobian,
                point_jacobian,
                tolerance=tolerance,
                maximum_iterations=maximum_inverse_iterations,
            )
            weight = float(rule_weight)
            points.append(
                MortarQuadraturePointLinearization(
                    pallet_index=pallet_index,
                    quadrature_index=quadrature_index,
                    barycentric=np.asarray(barycentric, dtype=float).copy(),
                    rule_weight=weight,
                    point=point,
                    point_jacobian=point_jacobian,
                    integration_weight=pallet.signed_area * weight,
                    integration_weight_jacobian=pallet.area_jacobian * weight,
                    slave=slave_map,
                    master=master_map,
                )
            )

    return FacetQuadratureLinearization(
        geometry=geometry,
        quadrature_point_count=quadrature_points,
        points=tuple(points),
    )
