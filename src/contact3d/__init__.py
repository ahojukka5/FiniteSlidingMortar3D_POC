"""Three-dimensional finite-sliding mortar contact research package."""

from .geometry import (
    clip_convex_polygon,
    facet_projection_plane,
    polygon_signed_area,
    project_to_plane,
    triangulate_convex_polygon,
)
from .model import FacetOverlap, LocalMortarWeights, MortarPallet, ProjectionPlane
from .overlap import build_facet_overlap, integrate_facet_pair
from .shapes import infer_facet_kind, inverse_map_2d, shape_values

__all__ = [
    "FacetOverlap",
    "LocalMortarWeights",
    "MortarPallet",
    "ProjectionPlane",
    "build_facet_overlap",
    "clip_convex_polygon",
    "facet_projection_plane",
    "infer_facet_kind",
    "integrate_facet_pair",
    "inverse_map_2d",
    "polygon_signed_area",
    "project_to_plane",
    "shape_values",
    "triangulate_convex_polygon",
]
