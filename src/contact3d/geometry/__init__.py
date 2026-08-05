"""Pure surface-geometry primitives for mortar contact."""

from .geometry import (
    clip_convex_polygon,
    ensure_counterclockwise,
    facet_projection_plane,
    facet_projection_plane_jacobian,
    polygon_signed_area,
    project_to_plane,
    project_to_plane_jacobian,
    triangulate_convex_polygon,
)
from .model import (
    FacetKind,
    FacetOverlap,
    FloatArray,
    IntArray,
    MortarPallet,
    ProjectedPointsJacobian,
    ProjectionPlane,
    ProjectionPlaneJacobian,
)
from .quadrature import triangle_rule
from .shapes import (
    center_parent,
    infer_facet_kind,
    inverse_map_2d,
    map_to_physical,
    shape_gradients,
    shape_values,
)

__all__ = [
    "FacetKind",
    "FacetOverlap",
    "FloatArray",
    "IntArray",
    "MortarPallet",
    "ProjectedPointsJacobian",
    "ProjectionPlane",
    "ProjectionPlaneJacobian",
    "center_parent",
    "clip_convex_polygon",
    "ensure_counterclockwise",
    "facet_projection_plane",
    "facet_projection_plane_jacobian",
    "infer_facet_kind",
    "inverse_map_2d",
    "map_to_physical",
    "polygon_signed_area",
    "project_to_plane",
    "project_to_plane_jacobian",
    "shape_gradients",
    "shape_values",
    "triangle_rule",
    "triangulate_convex_polygon",
]
