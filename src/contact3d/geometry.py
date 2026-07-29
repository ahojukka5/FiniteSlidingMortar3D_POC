"""Projection-plane geometry and convex polygon operations."""

from __future__ import annotations

import numpy as np

from .model import FacetKind, FloatArray, MortarPallet, ProjectionPlane
from .shapes import center_parent, map_to_physical, shape_gradients


def _normalize(vector: FloatArray, *, tolerance: float = 1.0e-14) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


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


def project_to_plane(points: FloatArray, plane: ProjectionPlane) -> FloatArray:
    """Orthogonally project 3D points and return coordinates in the plane frame."""

    relative = points - plane.origin
    return np.column_stack([relative @ plane.tangent_u, relative @ plane.tangent_v])


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
