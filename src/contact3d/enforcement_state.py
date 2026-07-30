"""Augmented-Lagrange multiplier state and KKT diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import ContactPair, GlobalMortarWeights
from .model import FloatArray


@dataclass(frozen=True, slots=True)
class AugmentedLagrangeState:
    """Accepted nonnegative nodal normal multipliers for one slave surface."""

    multipliers: FloatArray
    augmentation: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(self.multipliers, dtype=float)
        if values.ndim != 1:
            raise ValueError("augmented-Lagrange multipliers must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError("augmented-Lagrange multipliers must be finite")
        if np.any(values < 0.0):
            raise ValueError("augmented-Lagrange multipliers must be nonnegative")
        if self.augmentation < 0:
            raise ValueError("augmentation index must be nonnegative")
        object.__setattr__(self, "multipliers", values.copy())

    @classmethod
    def zeros(cls, node_count: int) -> AugmentedLagrangeState:
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        return cls(np.zeros(node_count, dtype=float))

    def validate_for(self, pair: ContactPair) -> None:
        if self.multipliers.shape != (pair.slave.node_count,):
            raise ValueError("multiplier state must match the slave-node count")


@dataclass(frozen=True, slots=True)
class KKTDiagnostics:
    """Nodal unilateral-contact residuals in the penetration-positive convention."""

    penetration: FloatArray
    multiplier_violation: FloatArray
    complementarity: FloatArray
    projection_residual: FloatArray
    unsupported_multiplier: FloatArray

    @property
    def maximum_penetration(self) -> float:
        return float(np.max(self.penetration, initial=0.0))

    @property
    def maximum_multiplier_violation(self) -> float:
        return float(np.max(self.multiplier_violation, initial=0.0))

    @property
    def maximum_complementarity(self) -> float:
        return float(np.max(self.complementarity, initial=0.0))

    @property
    def maximum_projection_residual(self) -> float:
        return float(np.max(self.projection_residual, initial=0.0))

    @property
    def maximum_unsupported_multiplier(self) -> float:
        return float(np.max(self.unsupported_multiplier, initial=0.0))

    @property
    def l2_residual(self) -> float:
        blocks = (
            self.penetration,
            self.multiplier_violation,
            self.complementarity,
            self.projection_residual,
            self.unsupported_multiplier,
        )
        return float(np.sqrt(sum(float(np.dot(value, value)) for value in blocks)))

    def converged(
        self,
        *,
        gap_tolerance: float,
        complementarity_tolerance: float,
        projection_tolerance: float,
        multiplier_tolerance: float | None = None,
    ) -> bool:
        limits = (gap_tolerance, complementarity_tolerance, projection_tolerance)
        if any(limit < 0.0 for limit in limits):
            raise ValueError("KKT tolerances must be nonnegative")
        multiplier_limit = (
            projection_tolerance
            if multiplier_tolerance is None
            else float(multiplier_tolerance)
        )
        if multiplier_limit < 0.0:
            raise ValueError("KKT tolerances must be nonnegative")
        return (
            self.maximum_penetration <= gap_tolerance
            and self.maximum_multiplier_violation <= multiplier_limit
            and self.maximum_complementarity <= complementarity_tolerance
            and self.maximum_projection_residual <= projection_tolerance
            and self.maximum_unsupported_multiplier <= multiplier_limit
        )


def supported_rows(weights: GlobalMortarWeights, tolerance: float) -> np.ndarray:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    return weights.row_areas > tolerance


def augmented_pressure_projection(
    multipliers: FloatArray,
    normal_gaps: FloatArray,
    penalty: float,
    supported_rows: np.ndarray,
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Return trial pressure, positive projection, and its active branch."""

    values = np.asarray(multipliers, dtype=float)
    gaps = np.asarray(normal_gaps, dtype=float)
    supported = np.asarray(supported_rows, dtype=bool)
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if values.ndim != 1 or gaps.shape != values.shape or supported.shape != values.shape:
        raise ValueError("projection inputs must share one nodal shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gaps)):
        raise ValueError("projection inputs must be finite")
    if np.any(values < 0.0):
        raise ValueError("multipliers must be nonnegative")
    trial = values + penalty * gaps
    pressure = np.where(supported, np.maximum(trial, 0.0), 0.0)
    return trial, pressure, supported & (trial > 0.0)


def kkt_diagnostics(
    multipliers: FloatArray,
    normal_gaps: FloatArray,
    penalty: float,
    supported_rows: np.ndarray,
) -> KKTDiagnostics:
    """Evaluate primal, dual, complementarity, and projection residuals."""

    values = np.asarray(multipliers, dtype=float)
    gaps = np.asarray(normal_gaps, dtype=float)
    supported = np.asarray(supported_rows, dtype=bool)
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if values.ndim != 1 or gaps.shape != values.shape or supported.shape != values.shape:
        raise ValueError("KKT inputs must share one nodal shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gaps)):
        raise ValueError("KKT inputs must be finite")
    projected = np.where(supported, np.maximum(values + penalty * gaps, 0.0), 0.0)
    return KKTDiagnostics(
        penetration=np.where(supported, np.maximum(gaps, 0.0), 0.0),
        multiplier_violation=np.maximum(-values, 0.0),
        complementarity=np.where(supported, np.abs(values * gaps), 0.0),
        projection_residual=np.where(supported, np.abs(values - projected), 0.0),
        unsupported_multiplier=np.where(~supported, np.abs(values), 0.0),
    )
