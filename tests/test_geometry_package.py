from __future__ import annotations

import numpy as np

from contact3d.broad_phase import FacetAABBTree as FlatFacetAABBTree
from contact3d.clipping import (
    clip_convex_polygon_linearized as flat_clip_convex_polygon_linearized,
)
from contact3d.geometry import (
    ContactSurface,
    FacetAABBTree,
    InverseMapLinearization,
    ProjectionPlane,
    clip_convex_polygon_linearized,
    discover_facet_pairs,
    facet_projection_plane,
    linearize_facet_pallets,
    shape_values,
    triangle_rule,
)
from contact3d.model import ProjectionPlane as FlatProjectionPlane
from contact3d.pallets import linearize_facet_pallets as flat_linearize_facet_pallets
from contact3d.parametric import (
    InverseMapLinearization as FlatInverseMapLinearization,
)
from contact3d.quadrature import triangle_rule as flat_triangle_rule
from contact3d.shapes import shape_values as flat_shape_values
from contact3d.surface import ContactSurface as FlatContactSurface
from contact3d.surface import discover_facet_pairs as flat_discover_facet_pairs


def test_flat_geometry_imports_are_temporary_reexports() -> None:
    assert FlatProjectionPlane is ProjectionPlane
    assert flat_shape_values is shape_values
    assert flat_triangle_rule is triangle_rule
    assert flat_clip_convex_polygon_linearized is clip_convex_polygon_linearized
    assert flat_linearize_facet_pallets is linearize_facet_pallets
    assert FlatInverseMapLinearization is InverseMapLinearization
    assert FlatFacetAABBTree is FacetAABBTree
    assert FlatContactSurface is ContactSurface
    assert flat_discover_facet_pairs is discover_facet_pairs


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
