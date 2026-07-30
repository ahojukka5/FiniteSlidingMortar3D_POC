"""Immutable adaptive-contact continuation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .coupled import AugmentedContactResult, CoupledEquilibriumProblem
from .enforcement_state import AugmentedLagrangeState
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

    @property
    def accepted_step_count(self) -> int:
        return len(self.accepted_results)

    @property
    def cutback_count(self) -> int:
        return sum(attempt.action == "cutback" for attempt in self.attempts)

    @property
    def penalty_update_count(self) -> int:
        return sum(attempt.action == "penalty_increase" for attempt in self.attempts)
