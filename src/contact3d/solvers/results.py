"""Immutable options and result records for nonlinear solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..coupling import CoupledEquilibriumEvaluation
from ..mechanics import EquilibriumEvaluation, FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from .linear import LinearSolveDiagnostics, LinearSolverOptions

TerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "line_search_failed",
    "singular_tangent",
    "linear_solve_failed",
]
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
class NewtonOptions:
    maximum_iterations: int = 30
    absolute_tolerance: float = 1.0e-10
    relative_tolerance: float = 1.0e-10
    armijo_coefficient: float = 1.0e-4
    line_search_reduction: float = 0.5
    minimum_step: float = 2.0**-20
    maximum_line_search_iterations: int = 24
    linear_solver: LinearSolverOptions = field(default_factory=LinearSolverOptions)

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.maximum_line_search_iterations <= 0:
            raise ValueError("maximum_line_search_iterations must be positive")
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
            ("armijo_coefficient", self.armijo_coefficient),
            ("minimum_step", self.minimum_step),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.armijo_coefficient < 1.0:
            raise ValueError("armijo_coefficient must lie between zero and one")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must lie between zero and one")


@dataclass(frozen=True, slots=True)
class NewtonIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    potential: float
    minimum_jacobian: float
    step_norm: float
    accepted_step: float
    line_search_iterations: int
    linear_solve: LinearSolveDiagnostics


@dataclass(frozen=True, slots=True)
class NewtonResult:
    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: TerminationReason
    evaluation: EquilibriumEvaluation
    history: tuple[NewtonIteration, ...]
    linear_solve_failure: LinearSolveDiagnostics | None = None

    @property
    def iteration_count(self) -> int:
        return len(self.history)


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
    "NewtonIteration",
    "NewtonOptions",
    "NewtonResult",
    "TerminationReason",
]
