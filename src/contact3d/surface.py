"""Temporary migration exports for contact-surface geometry.

Remove this module in issue #136 after repository imports use
``contact3d.geometry`` directly.
"""

from .geometry.surface import (
    BroadPhaseDiagnostics,
    ContactSurface,
    FacetAABBTree,
    FacetPair,
    FacetPairSearchResult,
    averaged_nodal_normal_jacobian,
    averaged_nodal_normals,
    discover_facet_pairs,
    discover_facet_pairs_brute_force,
    discover_facet_pairs_with_diagnostics,
)

__all__ = [
    "BroadPhaseDiagnostics",
    "ContactSurface",
    "FacetAABBTree",
    "FacetPair",
    "FacetPairSearchResult",
    "averaged_nodal_normal_jacobian",
    "averaged_nodal_normals",
    "discover_facet_pairs",
    "discover_facet_pairs_brute_force",
    "discover_facet_pairs_with_diagnostics",
]
