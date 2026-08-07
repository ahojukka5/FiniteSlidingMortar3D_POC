"""Adaptive continuation options and immutable result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..coupling import CoupledEquilibriumProblem
from ..mechanics import FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from ..scaling import ScaleAwareConvergenceOptions
from .results import AugmentedContactOptions, AugmentedContactResult

if TYPE_CHECKING:
    from ..load_path import CoupledPathState

AdaptiveAttemptAction = Literal["accepted", "cutback", "penalty_increase"]
AdaptiveTerminationReason = Literal[
    "converged",
    "minimum_step",
    "maximum_attempts",
]


@dataclass(frozen=True, slots=True)
class AdaptiveLoadOptions:
    """Step-size policy for monotone load-factor continuation."""

    initial_step: float = 0.25
    minimum_step: float = 1.0 / 1024.0
    maximum_step: float = 0.5
    cutback_factor: float = 0.5
    growth_factor: float = 1.5
    easy_newton_iterations: int = 8
    maximum_attempts: int = 100

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_step", self.initial_step),
            ("minimum_step", self.minimum_step),
            ("maximum_step", self.maximum_step),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_step > self.initial_step or self.initial_step > self.maximum_step:
            raise ValueError("load steps must satisfy minimum <= initial <= maximum")
        if not 0.0 < self.cutback_factor < 1.0:
            raise ValueError("cutback_factor must lie between zero and one")
        if not np.isfinite(self.growth_factor) or self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be finite and greater than one")
        if self.easy_newton_iterations < 0:
            raise ValueError("easy_newton_iterations must be nonnegative")
        if self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")


@dataclass(frozen=True, slots=True)
class AdaptivePenaltyOptions:
    """Escalate only under-resolved interface penalties within physical bounds."""

    enabled: bool = True
    increase_factor: float = 4.0
    maximum_penalty: float = 1.0e9
    maximum_updates_per_step: int = 4
    penetration_target: float | None = None
    normalized_penetration_target: float | None = None
    interface_local: bool = True
    minimum_scale_factor: float = 0.25
    maximum_scale_factor: float = 1.0e4

    def __post_init__(self) -> None:
        if not np.isfinite(self.increase_factor) or self.increase_factor <= 1.0:
            raise ValueError("increase_factor must be finite and greater than one")
        if not np.isfinite(self.maximum_penalty) or self.maximum_penalty <= 0.0:
            raise ValueError("maximum_penalty must be finite and positive")
        if self.maximum_updates_per_step < 0:
            raise ValueError("maximum_updates_per_step must be nonnegative")
        if self.penetration_target is not None and (
            not np.isfinite(self.penetration_target) or self.penetration_target < 0.0
        ):
            raise ValueError("penetration_target must be finite and nonnegative")
        if self.normalized_penetration_target is not None and (
            not np.isfinite(self.normalized_penetration_target)
            or self.normalized_penetration_target < 0.0
        ):
            raise ValueError(
                "normalized_penetration_target must be finite and nonnegative"
            )
        if not np.isfinite(self.minimum_scale_factor) or self.minimum_scale_factor <= 0.0:
            raise ValueError("minimum_scale_factor must be finite and positive")
        if (
            not np.isfinite(self.maximum_scale_factor)
            or self.maximum_scale_factor < self.minimum_scale_factor
        ):
            raise ValueError(
                "maximum_scale_factor must be finite and no smaller than the minimum"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveContactOptions:
    """Combined continuation, penalty, inner solve, and scaling settings."""

    load: AdaptiveLoadOptions = field(default_factory=AdaptiveLoadOptions)
    penalty: AdaptivePenaltyOptions = field(default_factory=AdaptivePenaltyOptions)
    augmented: AugmentedContactOptions = field(default_factory=AugmentedContactOptions)
    scaling: ScaleAwareConvergenceOptions = field(
        default_factory=ScaleAwareConvergenceOptions
    )


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
    """Accepted adaptive path and all rejected or retried attempts."""

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


__all__ = [
    "AdaptiveAcceptedStep",
    "AdaptiveAttemptAction",
    "AdaptiveContactAttempt",
    "AdaptiveContactOptions",
    "AdaptiveContactResult",
    "AdaptiveLoadOptions",
    "AdaptivePenaltyOptions",
    "AdaptiveTerminationReason",
]
