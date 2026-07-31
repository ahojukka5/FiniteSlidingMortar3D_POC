"""Result models for event-localized coupled contact solvers."""

from __future__ import annotations

from dataclasses import dataclass

from .coupled import (
    AugmentationIteration,
    CoupledEquilibriumEvaluation,
    CoupledNewtonIteration,
    CoupledNewtonResult,
    CoupledTerminationReason,
)
from .enforcement_state import AugmentedLagrangeState
from .model import FloatArray
from .topology_events import ContactTopologyEventBatch


@dataclass(frozen=True, slots=True)
class EventAwareCoupledNewtonResult:
    """One fixed-multiplier equilibrium result with localized event history."""

    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: CoupledTerminationReason
    evaluation: CoupledEquilibriumEvaluation
    history: tuple[CoupledNewtonIteration, ...]
    events: tuple[ContactTopologyEventBatch, ...]

    @property
    def iteration_count(self) -> int:
        return len(self.history)

    @property
    def contact_event_restarts(self) -> int:
        return len(self.events)

    def legacy_result(self) -> CoupledNewtonResult:
        """Drop event details while preserving the established solver result API."""

        return CoupledNewtonResult(
            self.displacement,
            self.load_factor,
            self.converged,
            self.termination_reason,
            self.evaluation,
            self.history,
            self.contact_event_restarts,
        )


@dataclass(frozen=True, slots=True)
class EventAwareAugmentedContactResult:
    """Outer augmented-Lagrange result retaining every localized Newton event."""

    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: EventAwareCoupledNewtonResult
    equilibria: tuple[EventAwareCoupledNewtonResult, ...]
    history: tuple[AugmentationIteration, ...]

    @property
    def events(self) -> tuple[ContactTopologyEventBatch, ...]:
        return tuple(event for result in self.equilibria for event in result.events)
