"""Result models for event-localized coupled contact solvers."""

from __future__ import annotations

from dataclasses import dataclass

from ...coupling import CoupledEquilibriumEvaluation
from ...mechanics import FloatArray
from ...mortar.enforcement import AugmentedLagrangeState
from ...topology_events import ContactTopologyEventBatch
from ..linear import LinearSolveDiagnostics
from ..results import (
    AugmentationIteration,
    CoupledNewtonIteration,
    CoupledNewtonResult,
    CoupledTerminationReason,
)
from .multiplier_transport import MultiplierTransportRecord


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
    linear_solve_failure: LinearSolveDiagnostics | None = None
    states: tuple[AugmentedLagrangeState, ...] = ()
    multiplier_transports: tuple[MultiplierTransportRecord, ...] = ()

    @property
    def iteration_count(self) -> int:
        return len(self.history)

    @property
    def contact_event_restarts(self) -> int:
        return len(self.events)

    @property
    def multiplier_transport_count(self) -> int:
        return len(self.multiplier_transports)

    def multiplier_transport_rows(self) -> tuple[dict[str, object], ...]:
        """Return machine-readable support-transport rows."""

        return tuple(record.as_dict() for record in self.multiplier_transports)

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
            self.linear_solve_failure,
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

    @property
    def multiplier_transports(self) -> tuple[MultiplierTransportRecord, ...]:
        return tuple(
            record
            for result in self.equilibria
            for record in result.multiplier_transports
        )


__all__ = [
    "EventAwareAugmentedContactResult",
    "EventAwareCoupledNewtonResult",
]
