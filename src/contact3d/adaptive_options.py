"""Adaptive load continuation and penalty control for coupled mortar contact."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coupled import AugmentedContactOptions


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
    """Escalate normal penalties after equilibrated but under-resolved AL attempts."""

    enabled: bool = True
    increase_factor: float = 4.0
    maximum_penalty: float = 1.0e9
    maximum_updates_per_step: int = 4
    penetration_target: float | None = None

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


@dataclass(frozen=True, slots=True)
class AdaptiveContactOptions:
    """Combined continuation, penalty, and inner augmented-contact settings."""

    load: AdaptiveLoadOptions = field(default_factory=AdaptiveLoadOptions)
    penalty: AdaptivePenaltyOptions = field(default_factory=AdaptivePenaltyOptions)
    augmented: AugmentedContactOptions = field(default_factory=AugmentedContactOptions)
