"""Immutable options and result records for coupled nonlinear solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..coupling import CoupledEquilibriumEvaluation
from ..equilibrium import NewtonOptions
from ..mechanics import FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from .linear import LinearSolveDiagnostics

ContactEventPolicy = Literal["restart", "reject"]
CoupledTerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "line_search_failed",
    "singular_tangent",
    "linear_solve_failed",
    "contact_linearization_event",
]
AugmentedTerminationReason = Literal[
    "converged",
    "maximum_augmentations",
    "inner_equilibrium_failed",
]


@dataclass(frozen=True, slots=True)
class CoupledNewtonIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    bulk_potential: float
    minimum_jacobian: float
    maximum_penetration: float
    step_norm: float
    accepted_step: float
    line_search_iterations: int
    contact_branch_changed: bool
    linear_solve: LinearSolveDiagnostics


@dataclass(frozen=True, slots=True)
class CoupledNewtonResult:
    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: CoupledTerminationReason
    evaluation: CoupledEquilibriumEvaluation
    history: tuple[CoupledNewtonIteration, ...]
    contact_event_restarts: int
    linear_solve_failure: LinearSolveDiagnostics | None = None

    @property
    def iteration_count(self) -> int:
        return len(self.history)


@dataclass(frozen=True, slots=True)
class AugmentedContactOptions:
    maximum_augmentations: int = 12
    gap_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-8
    projection_tolerance: float = 1.0e-8
    multiplier_tolerance: float = 1.0e-8
    event_policy: ContactEventPolicy = "restart"
    newton: NewtonOptions = field(default_factory=NewtonOptions)

    def __post_init__(self) -> None:
        if self.maximum_augmentations <= 0:
            raise ValueError("maximum_augmentations must be positive")
        for value in (
            self.gap_tolerance,
            self.complementarity_tolerance,
            self.projection_tolerance,
            self.multiplier_tolerance,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "augmented-contact tolerances must be finite and nonnegative"
                )
        if self.event_policy not in ("restart", "reject"):
            raise ValueError("event_policy must be 'restart' or 'reject'")


@dataclass(frozen=True, slots=True)
class AugmentationIteration:
    augmentation: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    maximum_penetration: float
    maximum_complementarity: float
    maximum_projection_residual: float
    maximum_multiplier_increment: float
    active_rows: int
    maximum_pressure: float


@dataclass(frozen=True, slots=True)
class AugmentedContactResult:
    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: AugmentedTerminationReason
    equilibrium: CoupledNewtonResult
    equilibria: tuple[CoupledNewtonResult, ...]
    history: tuple[AugmentationIteration, ...]


__all__ = [
    "AugmentationIteration",
    "AugmentedContactOptions",
    "AugmentedContactResult",
    "AugmentedTerminationReason",
    "ContactEventPolicy",
    "CoupledNewtonIteration",
    "CoupledNewtonResult",
    "CoupledTerminationReason",
    "NewtonOptions",
]
