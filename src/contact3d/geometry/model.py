"""Core data structures for pure facet geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FacetKind = Literal["tri3", "quad4"]


@dataclass(frozen=True, slots=True)
class ProjectionPlane:
    """Orthonormal frame used to flatten one non-mortar facet."""

    origin: FloatArray
    tangent_u: FloatArray
    tangent_v: FloatArray
    normal: FloatArray


@dataclass(frozen=True, slots=True)
class ProjectionPlaneJacobian:
    """Coordinate derivatives of a projection-plane frame.

    Every tensor has axes ``(output_component, input_node, input_component)``.
    """

    origin: FloatArray
    tangent_u: FloatArray
    tangent_v: FloatArray
    normal: FloatArray

    @property
    def node_count(self) -> int:
        return int(self.origin.shape[1])


@dataclass(frozen=True, slots=True)
class ProjectedPointsJacobian:
    """Derivatives of projected coordinates with separated variable groups.

    ``plane`` has axes ``(point, projected_component, plane_node, component)``.
    ``points`` has axes ``(point, projected_component, point_node, component)``.
    """

    plane: FloatArray
    points: FloatArray

    def combined_shared_coordinates(self) -> FloatArray:
        """Combine both terms when projected and plane-defining nodes coincide."""

        if self.plane.shape != self.points.shape:
            raise ValueError("plane and point Jacobians do not share the same nodes")
        return self.plane + self.points


@dataclass(frozen=True, slots=True)
class MortarPallet:
    """One triangular integration pallet in projection-plane coordinates."""

    vertices: FloatArray
    area: float


@dataclass(frozen=True, slots=True)
class FacetOverlap:
    """Projected intersection of one non-mortar and one mortar facet."""

    plane: ProjectionPlane
    slave_polygon: FloatArray
    master_polygon: FloatArray
    intersection_polygon: FloatArray
    pallets: tuple[MortarPallet, ...]

    @property
    def area(self) -> float:
        return float(sum(pallet.area for pallet in self.pallets))
