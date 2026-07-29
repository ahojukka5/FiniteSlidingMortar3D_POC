"""Three-dimensional finite-sliding mortar contact research package."""

from .contact import (
    ContactEvaluation,
    ContactPair,
    GlobalMortarWeights,
    assemble_mortar_weights,
    evaluate_contact,
    fixed_mortar_contact_tangent,
    numerical_contact_tangent,
)
from .geometry import (
    clip_convex_polygon,
    facet_projection_plane,
    facet_projection_plane_jacobian,
    polygon_signed_area,
    project_to_plane,
    project_to_plane_jacobian,
    triangulate_convex_polygon,
)
from .model import (
    FacetOverlap,
    LocalMortarWeights,
    MortarPallet,
    ProjectedPointsJacobian,
    ProjectionPlane,
    ProjectionPlaneJacobian,
)
from .moving import (
    MortarWeightJacobian,
    moving_mortar_contact_tangent,
    numerical_mortar_weight_jacobian,
)
from .overlap import build_facet_overlap, integrate_facet_pair
from .shapes import infer_facet_kind, inverse_map_2d, shape_values
from .surface import (
    ContactSurface,
    averaged_nodal_normal_jacobian,
    averaged_nodal_normals,
    discover_facet_pairs,
)

__all__ = [
    "ContactEvaluation",
    "ContactPair",
    "ContactSurface",
    "FacetOverlap",
    "GlobalMortarWeights",
    "LocalMortarWeights",
    "MortarPallet",
    "MortarWeightJacobian",
    "ProjectedPointsJacobian",
    "ProjectionPlane",
    "ProjectionPlaneJacobian",
    "assemble_mortar_weights",
    "averaged_nodal_normal_jacobian",
    "averaged_nodal_normals",
    "build_facet_overlap",
    "clip_convex_polygon",
    "discover_facet_pairs",
    "evaluate_contact",
    "facet_projection_plane",
    "facet_projection_plane_jacobian",
    "fixed_mortar_contact_tangent",
    "infer_facet_kind",
    "integrate_facet_pair",
    "inverse_map_2d",
    "moving_mortar_contact_tangent",
    "numerical_contact_tangent",
    "numerical_mortar_weight_jacobian",
    "polygon_signed_area",
    "project_to_plane",
    "project_to_plane_jacobian",
    "shape_values",
    "triangulate_convex_polygon",
]
