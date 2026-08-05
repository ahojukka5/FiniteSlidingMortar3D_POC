"""Temporary migration exports for geometry clipping.

Remove this module in issue #136 after repository imports use
``contact3d.geometry`` directly.
"""

from .geometry.clipping import (
    ClippedPolygonLinearization,
    ClippingOperation,
    ClippingStage,
    ClippingTopology,
    ClippingTopologyError,
    FacetIntersectionLinearization,
    OperationKind,
    clip_convex_polygon_linearized,
    linearize_clipping_topology,
    linearize_facet_intersection,
    replay_clipping_topology,
    trace_clipping_topology,
)

__all__ = [
    "ClippedPolygonLinearization",
    "ClippingOperation",
    "ClippingStage",
    "ClippingTopology",
    "ClippingTopologyError",
    "FacetIntersectionLinearization",
    "OperationKind",
    "clip_convex_polygon_linearized",
    "linearize_clipping_topology",
    "linearize_facet_intersection",
    "replay_clipping_topology",
    "trace_clipping_topology",
]
