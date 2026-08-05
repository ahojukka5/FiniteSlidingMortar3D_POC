"""Linear triangle and bilinear quadrilateral surface interpolation."""

from __future__ import annotations

import numpy as np

from .model import FacetKind, FloatArray


def infer_facet_kind(points: FloatArray) -> FacetKind:
    if points.shape == (3, 3):
        return "tri3"
    if points.shape == (4, 3):
        return "quad4"
    raise ValueError("facet coordinates must have shape (3, 3) or (4, 3)")


def shape_values(kind: FacetKind, parent: FloatArray) -> FloatArray:
    if kind == "tri3":
        r, s = map(float, parent)
        return np.array([1.0 - r - s, r, s], dtype=float)
    xi, eta = map(float, parent)
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def shape_gradients(kind: FacetKind, parent: FloatArray) -> FloatArray:
    if kind == "tri3":
        return np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    xi, eta = map(float, parent)
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )


def center_parent(kind: FacetKind) -> FloatArray:
    return np.array([1.0 / 3.0, 1.0 / 3.0]) if kind == "tri3" else np.zeros(2)


def map_to_physical(points: FloatArray, kind: FacetKind, parent: FloatArray) -> FloatArray:
    return shape_values(kind, parent) @ points


def inverse_map_2d(
    polygon: FloatArray,
    kind: FacetKind,
    point: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 25,
) -> FloatArray:
    """Invert a projected linear/bilinear facet map in a two-dimensional plane."""

    if kind == "tri3":
        jacobian = np.column_stack([polygon[1] - polygon[0], polygon[2] - polygon[0]])
        determinant = float(np.linalg.det(jacobian))
        if abs(determinant) <= tolerance:
            raise ValueError("degenerate projected triangle")
        return np.linalg.solve(jacobian, point - polygon[0])

    parent = np.zeros(2, dtype=float)
    for _ in range(maximum_iterations):
        values = shape_values(kind, parent)
        gradients = shape_gradients(kind, parent)
        residual = values @ polygon - point
        if float(np.linalg.norm(residual)) <= tolerance:
            return parent
        jacobian = polygon.T @ gradients
        determinant = float(np.linalg.det(jacobian))
        if abs(determinant) <= tolerance:
            raise ValueError("singular projected quadrilateral inverse map")
        increment = np.linalg.solve(jacobian, residual)
        parent -= increment
        if float(np.linalg.norm(increment)) <= tolerance:
            return parent
    raise ValueError("projected quadrilateral inverse map did not converge")
