"""Core data structures for facet overlap and mortar integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FacetKind = Literal["tri3", "quad4"]


@dataclass(frozen=True, slots=True)
class ProjectionPlane:
    """Orthonormal frame used to flatten one non-mortar facet."""

    origin: FloatArray
    tangent_u: FloatArray
    tangent_v: FloatArray
    normal: FloatArray


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


@dataclass(frozen=True, slots=True)
class LocalMortarWeights:
    """Local standard-mortar matrices for one overlapping facet pair."""

    d: FloatArray
    m: FloatArray
    overlap: FacetOverlap

    @property
    def consistency_error(self) -> float:
        """Maximum violation of the row-wise partition-of-unity identity."""

        return float(np.max(np.abs(np.sum(self.d, axis=1) - np.sum(self.m, axis=1))))
