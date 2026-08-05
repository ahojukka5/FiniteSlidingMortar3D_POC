from __future__ import annotations

import numpy as np

from contact3d.geometry import (
    ProjectionPlane,
    facet_projection_plane,
    shape_values,
    triangle_rule,
)
from contact3d.model import ProjectionPlane as LegacyProjectionPlane
from contact3d.quadrature import triangle_rule as legacy_triangle_rule
from contact3d.shapes import shape_values as legacy_shape_values


def test_flat_geometry_imports_are_compatibility_reexports() -> None:
    assert LegacyProjectionPlane is ProjectionPlane
    assert legacy_shape_values is shape_values
    assert legacy_triangle_rule is triangle_rule


def test_geometry_package_owns_projection_and_quadrature() -> None:
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    plane = facet_projection_plane(triangle, "tri3")
    points, weights = triangle_rule(3)

    assert isinstance(plane, ProjectionPlane)
    assert np.allclose(plane.normal, [0.0, 0.0, 1.0])
    assert np.allclose(np.sum(points, axis=1), 1.0)
    assert np.isclose(np.sum(weights), 1.0)
