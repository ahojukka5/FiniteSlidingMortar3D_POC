"""Data contracts for biased frictionless standard mortar contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import ContactSurface, FacetOverlap, FacetPair, FloatArray


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


@dataclass(frozen=True, slots=True)
class ContactPair:
    """Biased frictionless contact pair with a non-mortar slave surface."""

    slave: ContactSurface
    master: ContactSurface
    normal_penalty: float
    search_distance: float
    quadrature_points: int = 7

    def __post_init__(self) -> None:
        if self.normal_penalty <= 0.0:
            raise ValueError("normal_penalty must be positive")
        if self.search_distance <= 0.0:
            raise ValueError("search_distance must be positive")
        if self.quadrature_points not in (1, 3, 7):
            raise ValueError("quadrature_points must be 1, 3, or 7")


@dataclass(frozen=True, slots=True)
class GlobalMortarWeights:
    """Global standard-mortar operators assembled from overlapping facet pairs."""

    d: FloatArray
    m: FloatArray
    facet_pairs: tuple[FacetPair, ...]
    overlap_areas: FloatArray

    @property
    def row_areas(self) -> FloatArray:
        return np.sum(self.d, axis=1)

    @property
    def consistency_error(self) -> float:
        difference = np.sum(self.d, axis=1) - np.sum(self.m, axis=1)
        return float(np.max(np.abs(difference)))

    @property
    def total_area(self) -> float:
        return float(np.sum(self.overlap_areas))


@dataclass(frozen=True, slots=True)
class ContactEvaluation:
    """Current frictionless mortar residual and its principal diagnostics."""

    residual: FloatArray
    slave_nodes: FloatArray
    master_nodes: FloatArray
    slave_force: FloatArray
    master_force: FloatArray
    nodal_normals: FloatArray
    weighted_gap_vectors: FloatArray
    weighted_normal_gaps: FloatArray
    normal_gaps: FloatArray
    pressure: FloatArray
    active_rows: np.ndarray
    weights: GlobalMortarWeights

    @property
    def force_balance(self) -> FloatArray:
        return np.sum(self.slave_force, axis=0) + np.sum(self.master_force, axis=0)

    @property
    def force_balance_norm(self) -> float:
        return float(np.linalg.norm(self.force_balance))

    @property
    def moment_balance(self) -> FloatArray:
        slave_moment = np.sum(np.cross(self.slave_nodes, self.slave_force), axis=0)
        master_moment = np.sum(np.cross(self.master_nodes, self.master_force), axis=0)
        return slave_moment + master_moment

    @property
    def moment_balance_norm(self) -> float:
        return float(np.linalg.norm(self.moment_balance))

    @property
    def maximum_penetration(self) -> float:
        return float(np.max(np.maximum(self.normal_gaps, 0.0), initial=0.0))
