"""Adaptive load continuation and penalty control for coupled mortar contact."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coupled import AugmentedContactOptions
from .scaling import ScaleAwareConvergenceOptions


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
