"""Immutable adaptive-contact continuation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .coupled import AugmentedContactResult, CoupledEquilibriumProblem
from .enforcement_state import AugmentedLagrangeState
from .load_path import CoupledPathState
from .model import FloatArray

AdaptiveAttemptAction = Literal["accepted", "cutback", "penalty_increase"]
AdaptiveTerminationReason = Literal[
    "converged",
    "minimum_step",
    "maximum_attempts",
]


@dataclass(frozen=True, slots=True)
class AdaptiveContactAttempt:
    """One accepted, cut-back, or penalty-escalated continuation attempt."""

    attempt: int
    start_load_factor: float
    target_load_factor: float
    step_size: float
    action: AdaptiveAttemptAction
    inner_termination_reason: str
    augmentations: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    maximum_penetration: float
    penalties_before: tuple[float, ...]
    penalties_after: tuple[float, ...]
    path_values: tuple[tuple[str, float], ...] = ()
    prescribed_dofs: tuple[int, ...] = ()
    prescribed_values: tuple[float, ...] = ()
    effective_load_norm: float = 0.0
    reaction_norm: float = 0.0
    normalized_equilibrium_residual: float = 0.0
    normalized_maximum_penetration: float = 0.0
    interface_penetrations: tuple[float, ...] = ()
    normalized_interface_penetrations: tuple[float, ...] = ()
    penalty_ratios_before: tuple[float, ...] = ()
    penalty_ratios_after: tuple[float, ...] = ()
    penalty_update_reasons: tuple[str, ...] = ()

    @property
    def start_parameter(self) -> float:
        return self.start_load_factor

    @property
    def target_parameter(self) -> float:
        return self.target_load_factor


@dataclass(frozen=True, slots=True)
class AdaptiveAcceptedStep:
    """One committed path state, equilibrium solution, and constrained reaction."""

    path_state: CoupledPathState
    result: AugmentedContactResult
    reaction: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.reaction, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("accepted-step reaction must be a finite flat vector")
        force = self.path_state.effective_force
        if len(force) and values.shape != force.shape:
            raise ValueError("accepted-step reaction must match the global force vector")
        object.__setattr__(self, "reaction", values.copy())

    @property
    def parameter(self) -> float:
        return self.path_state.parameter

    @property
    def reaction_norm(self) -> float:
        return float(np.linalg.norm(self.reaction))


@dataclass(frozen=True, slots=True)
class AdaptiveContactResult:
    """Accepted adaptive path and all rejected/retried attempts."""

    problem: CoupledEquilibriumProblem
    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    load_factor: float
    converged: bool
    termination_reason: AdaptiveTerminationReason
    accepted_results: tuple[AugmentedContactResult, ...]
    attempts: tuple[AdaptiveContactAttempt, ...]
    accepted_steps: tuple[AdaptiveAcceptedStep, ...] = ()

    @property
    def accepted_step_count(self) -> int:
        return len(self.accepted_results)

    @property
    def cutback_count(self) -> int:
        return sum(attempt.action == "cutback" for attempt in self.attempts)

    @property
    def penalty_update_count(self) -> int:
        return sum(attempt.action == "penalty_increase" for attempt in self.attempts)

    @property
    def final_path_state(self) -> CoupledPathState | None:
        return self.accepted_steps[-1].path_state if self.accepted_steps else None
