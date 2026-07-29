"""Three-dimensional finite-sliding mortar contact research package."""

from .contact import (
    ContactEvaluation,
    ContactPair,
    GlobalMortarWeights,
    assemble_mortar_weights,
    evaluate_contact,
    numerical_contact_tangent,
)
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
from .surface import ContactSurface, averaged_nodal_normals, discover_facet_pairs

__all__ = [
    "ContactEvaluation",
    "ContactPair",
    "ContactSurface",
    "FacetOverlap",
    "GlobalMortarWeights",
    "LocalMortarWeights",
    "MortarPallet",
    "ProjectionPlane",
    "assemble_mortar_weights",
    "averaged_nodal_normals",
    "build_facet_overlap",
    "clip_convex_polygon",
    "discover_facet_pairs",
    "evaluate_contact",
    "facet_projection_plane",
    "infer_facet_kind",
    "integrate_facet_pair",
    "inverse_map_2d",
    "numerical_contact_tangent",
    "polygon_signed_area",
    "project_to_plane",
    "shape_values",
    "triangulate_convex_polygon",
]
