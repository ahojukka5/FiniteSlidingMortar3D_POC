"""Temporary flat-path exports for geometry and local mortar records.

This module is migration scaffolding only and is removed by issue #136.
"""

from .geometry.model import (
    FacetKind,
    FacetOverlap,
    FloatArray,
    IntArray,
    MortarPallet,
    ProjectedPointsJacobian,
    ProjectionPlane,
    ProjectionPlaneJacobian,
)
from .mortar.model import LocalMortarWeights

__all__ = [
    "FacetKind",
    "FacetOverlap",
    "FloatArray",
    "IntArray",
    "LocalMortarWeights",
    "MortarPallet",
    "ProjectedPointsJacobian",
    "ProjectionPlane",
    "ProjectionPlaneJacobian",
]
