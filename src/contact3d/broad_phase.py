"""Temporary migration exports for geometry broad-phase search.

Remove this module in issue #136 after repository imports use
``contact3d.geometry`` directly.
"""

from .geometry.broad_phase import (
    AABBTreeNode,
    BroadPhaseDiagnostics,
    FacetAABBTree,
    FacetPair,
    FacetPairSearchResult,
    RefitFacetAABBTree,
    facet_aabbs,
)

__all__ = [
    "AABBTreeNode",
    "BroadPhaseDiagnostics",
    "FacetAABBTree",
    "FacetPair",
    "FacetPairSearchResult",
    "RefitFacetAABBTree",
    "facet_aabbs",
]
