"""Projection-plane geometry and convex polygon operations."""

from __future__ import annotations

import numpy as np

from .model import (
    FacetKind,
    FloatArray,
    MortarPallet,
    ProjectedPointsJacobian,
    ProjectionPlane,
    ProjectionPlaneJacobian,
)
from .shapes import center_parent, map_to_physical, shape_gradients, shape_values


def _normalize(vector: FloatArray, *, tolerance: float = 1.0e-14) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def _normalization_jacobian(
    vector: FloatArray,
    *,
    tolerance: float = 1.0e-14,
) -> FloatArray:
    """Return the derivative of ``vector / ||vector||`` with respect to ``vector``."""

    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("cannot differentiate a zero-length vector normalization")
    unit = vector / norm
    return (np.eye(3) - np.outer(unit, unit)) / norm


def _cross_matrix(vector: FloatArray) -> FloatArray:
    """Return ``W`` such that ``W @ value == vector x value``."""

    x, y, z = map(float, vector)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def facet_projection_plane(points: FloatArray, kind: FacetKind) -> ProjectionPlane:
    """Form the center tangent plane used by Puso and Laursen's integration scheme."""

    parent = center_parent(kind)
    gradients = shape_gradients(kind, parent)
    tangents = points.T @ gradients
    tangent_u = _normalize(tangents[:, 0])
    normal = _normalize(np.cross(tangents[:, 0], tangents[:, 1]))
    tangent_v = _normalize(np.cross(normal, tangent_u))
    origin = map_to_physical(points, kind, parent)
    return ProjectionPlane(origin, tangent_u, tangent_v, normal)


def facet_projection_plane_jacobian(
    points: FloatArray,
    kind: FacetKind,
    *,
    tolerance: float = 1.0e-14,
) -> ProjectionPlaneJacobian:
    """Differentiate the center projection-plane frame analytically.

    The output tensors have axes ``(frame_component, facet_node, node_component)``.
    The derivative includes the center origin, normalized first covariant tangent,
    normalized surface normal, and the normalized in-plane second tangent.
    """

    coordinates = np.asarray(points, dtype=float)
    parent = center_parent(kind)
    values = shape_values(kind, parent)
    gradients = shape_gradients(kind, parent)
    if coordinates.shape != (len(values), 3):
        raise ValueError("facet coordinates do not match the requested facet kind")

    covariant = coordinates.T @ gradients
    first = covariant[:, 0]
    second = covariant[:, 1]
    tangent_u = _normalize(first, tolerance=tolerance)
    area_vector = np.cross(first, second)
    normal = _normalize(area_vector, tolerance=tolerance)
    transverse = np.cross(normal, tangent_u)

    derivative_u = _normalization_jacobian(first, tolerance=tolerance)
    derivative_n = _normalization_jacobian(area_vector, tolerance=tolerance)
    derivative_v = _normalization_jacobian(transverse, tolerance=tolerance)

    node_count = len(coordinates)
    origin_jacobian = np.zeros((3, node_count, 3), dtype=float)
    tangent_u_jacobian = np.zeros_like(origin_jacobian)
    tangent_v_jacobian = np.zeros_like(origin_jacobian)
    normal_jacobian = np.zeros_like(origin_jacobian)

    identity = np.eye(3)
    for node in range(node_count):
        origin_jacobian[:, node, :] = values[node] * identity
        derivative_first = gradients[node, 0] * identity
        derivative_second = gradients[node, 1] * identity
        du = derivative_u @ derivative_first
        darea = (
            -_cross_matrix(second) @ derivative_first
            + _cross_matrix(first) @ derivative_second
        )
        dn = derivative_n @ darea
        dtransverse = -_cross_matrix(tangent_u) @ dn + _cross_matrix(normal) @ du
        dv = derivative_v @ dtransverse

        tangent_u_jacobian[:, node, :] = du
        tangent_v_jacobian[:, node, :] = dv
        normal_jacobian[:, node, :] = dn

    return ProjectionPlaneJacobian(
        origin=origin_jacobian,
        tangent_u=tangent_u_jacobian,
        tangent_v=tangent_v_jacobian,
        normal=normal_jacobian,
    )


def project_to_plane(points: FloatArray, plane: ProjectionPlane) -> FloatArray:
    """Orthogonally project 3D points and return coordinates in the plane frame."""

    relative = points - plane.origin
    return np.column_stack([relative @ plane.tangent_u, relative @ plane.tangent_v])


def project_to_plane_jacobian(
    points: FloatArray,
    plane: ProjectionPlane,
    plane_jacobian: ProjectionPlaneJacobian,
) -> ProjectedPointsJacobian:
    """Differentiate projected coordinates with separated coordinate groups.

    ``plane`` contains derivatives with respect to the nodes that define the
    projection frame. ``points`` contains the direct derivatives with respect to
    the projected points. For a slave facet projected onto its own plane, combine
    both tensors with :meth:`ProjectedPointsJacobian.combined_shared_coordinates`.
    """

    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("projected points must have shape (point_count, 3)")

    node_count = plane_jacobian.node_count
    expected = (3, node_count, 3)
    for value in (
        plane_jacobian.origin,
        plane_jacobian.tangent_u,
        plane_jacobian.tangent_v,
        plane_jacobian.normal,
    ):
        if value.shape != expected:
            raise ValueError("projection-plane Jacobian tensors have incompatible shapes")

    plane_part = np.zeros((len(coordinates), 2, node_count, 3), dtype=float)
    point_part = np.zeros((len(coordinates), 2, len(coordinates), 3), dtype=float)
    for point_index, point in enumerate(coordinates):
        relative = point - plane.origin
        for node in range(node_count):
            plane_part[point_index, 0, node, :] = (
                -plane.tangent_u @ plane_jacobian.origin[:, node, :]
                + relative @ plane_jacobian.tangent_u[:, node, :]
            )
            plane_part[point_index, 1, node, :] = (
                -plane.tangent_v @ plane_jacobian.origin[:, node, :]
                + relative @ plane_jacobian.tangent_v[:, node, :]
            )
        point_part[point_index, 0, point_index, :] = plane.tangent_u
        point_part[point_index, 1, point_index, :] = plane.tangent_v

    return ProjectedPointsJacobian(plane=plane_part, points=point_part)


def polygon_signed_area(polygon: FloatArray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_counterclockwise(polygon: FloatArray) -> FloatArray:
    return polygon.copy() if polygon_signed_area(polygon) >= 0.0 else polygon[::-1].copy()


def _cross2(a: FloatArray, b: FloatArray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _deduplicate_polygon(polygon: FloatArray, tolerance: float) -> FloatArray:
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=float)
    answer = [np.asarray(polygon[0], dtype=float)]
    for point in polygon[1:]:
        if float(np.linalg.norm(point - answer[-1])) > tolerance:
            answer.append(np.asarray(point, dtype=float))
    if len(answer) > 1 and float(np.linalg.norm(answer[0] - answer[-1])) <= tolerance:
        answer.pop()
    return np.asarray(answer, dtype=float)


def clip_convex_polygon(
    subject: FloatArray,
    clipper: FloatArray,
    *,
    tolerance: float = 1.0e-12,
) -> FloatArray:
    """Intersect two convex polygons with Sutherland-Hodgman clipping."""

    output = ensure_counterclockwise(np.asarray(subject, dtype=float))
    clip = ensure_counterclockwise(np.asarray(clipper, dtype=float))
    for clip_start, clip_end in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        if len(output) == 0:
            break
        edge = clip_end - clip_start
        input_polygon = output
        output_points: list[FloatArray] = []
        previous = input_polygon[-1]
        previous_distance = _cross2(edge, previous - clip_start)
        previous_inside = previous_distance >= -tolerance
        for current in input_polygon:
            current_distance = _cross2(edge, current - clip_start)
            current_inside = current_distance >= -tolerance
            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                if abs(denominator) > tolerance:
                    fraction = previous_distance / denominator
                    output_points.append(previous + fraction * (current - previous))
            if current_inside:
                output_points.append(current)
            previous = current
            previous_distance = current_distance
            previous_inside = current_inside
        output = _deduplicate_polygon(np.asarray(output_points, dtype=float), tolerance)
    if len(output) < 3 or abs(polygon_signed_area(output)) <= tolerance:
        return np.empty((0, 2), dtype=float)
    return ensure_counterclockwise(output)


def triangulate_convex_polygon(
    polygon: FloatArray,
    *,
    tolerance: float = 1.0e-14,
) -> tuple[MortarPallet, ...]:
    """Create centroid-fan triangular pallets as in the paper's Figure 3(d)."""

    polygon = ensure_counterclockwise(np.asarray(polygon, dtype=float))
    if len(polygon) < 3:
        return ()
    center = np.mean(polygon, axis=0)
    pallets: list[MortarPallet] = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        vertices = np.vstack([center, first, second])
        area = 0.5 * abs(_cross2(first - center, second - center))
        if area > tolerance:
            pallets.append(MortarPallet(vertices, area))
    return tuple(pallets)
