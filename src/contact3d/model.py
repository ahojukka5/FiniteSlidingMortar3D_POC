"""Compatibility exports for geometry and local mortar records.

New code should import geometry records from :mod:`contact3d.geometry`.
"""

from .geometry.model import (
    FacetKind,
    FacetOverlap,
    FloatArray,
    IntArray,
    LocalMortarWeights,
    MortarPallet,
    ProjectedPointsJacobian,
    ProjectionPlane,
    ProjectionPlaneJacobian,
)

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
