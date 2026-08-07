"""Event-aware adaptive continuation and absolute path event records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...coupling import CoupledEquilibriumProblem
from ...load_path import CoupledLoadPath
from ...mechanics import FloatArray
from ...mortar.enforcement import AugmentedLagrangeState
from ...topology_events import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    TopologyEventLocalizationOptions,
)
from ..continuation import (
    AdaptiveAcceptedStep,
    AdaptiveAttemptAction,
    AdaptiveContactAttempt,
    AdaptiveContactOptions,
    AdaptiveContactResult,
    solve_adaptive_contact_path,
)
from .augmentation import solve_event_aware_augmented_contact
from .scaling import solve_event_aware_scale_aware_augmented_contact

EventAwareAdaptiveSolver = Callable[..., object]


@dataclass(frozen=True, slots=True)
class AdaptiveTopologyEventBatch:
    """One localized Newton event annotated by its adaptive path attempt."""

    attempt: int
    action: AdaptiveAttemptAction
    start_parameter: float
    target_parameter: float
    continuation_parameter: float
    solver_load_factor: float
    path_values: tuple[tuple[str, float], ...]
    augmentation: int
    batch_index: int
    batch: ContactTopologyEventBatch

    def __post_init__(self) -> None:
        for name in (
            "start_parameter",
            "target_parameter",
            "continuation_parameter",
            "solver_load_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.attempt <= 0:
            raise ValueError("attempt must be positive")
        if self.augmentation < 0 or self.batch_index < 0:
            raise ValueError("event indices must be nonnegative")
        if self.target_parameter < self.start_parameter:
            raise ValueError("target parameter must not precede the attempt start")
        object.__setattr__(self, "path_values", tuple(self.path_values))

    @property
    def events(self) -> tuple[ContactTopologyEvent, ...]:
        return self.batch.events


@dataclass(frozen=True, slots=True)
class EventAwareAdaptiveContactResult:
    """Adaptive continuation result with absolute-path topology history."""

    adaptive: AdaptiveContactResult
    event_batches: tuple[AdaptiveTopologyEventBatch, ...]
    attempt_results: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        batches = tuple(self.event_batches)
        attempts = tuple(self.attempt_results)
        if attempts and len(attempts) != len(self.adaptive.attempts):
            raise ValueError("attempt results must align with adaptive attempts")
        object.__setattr__(self, "event_batches", batches)
        object.__setattr__(self, "attempt_results", attempts)

    @property
    def problem(self) -> CoupledEquilibriumProblem:
        return self.adaptive.problem

    @property
    def displacement(self) -> FloatArray:
        return self.adaptive.displacement

    @property
    def states(self) -> tuple[AugmentedLagrangeState, ...]:
        return self.adaptive.states

    @property
    def load_factor(self) -> float:
        return self.adaptive.load_factor

    @property
    def converged(self) -> bool:
        return self.adaptive.converged

    @property
    def termination_reason(self) -> str:
        return self.adaptive.termination_reason

    @property
    def accepted_results(self) -> tuple[object, ...]:
        return self.adaptive.accepted_results

    @property
    def attempts(self) -> tuple[AdaptiveContactAttempt, ...]:
        return self.adaptive.attempts

    @property
    def accepted_steps(self) -> tuple[AdaptiveAcceptedStep, ...]:
        return self.adaptive.accepted_steps

    @property
    def accepted_step_count(self) -> int:
        return self.adaptive.accepted_step_count

    @property
    def cutback_count(self) -> int:
        return self.adaptive.cutback_count

    @property
    def penalty_update_count(self) -> int:
        return self.adaptive.penalty_update_count

    @property
    def contact_event_restarts(self) -> int:
        return len(self.event_batches)

    @property
    def events(self) -> tuple[ContactTopologyEvent, ...]:
        return tuple(event for record in self.event_batches for event in record.events)

    def event_rows(self) -> tuple[dict[str, object], ...]:
        """Flatten batches into deterministic machine-readable atomic event rows."""

        rows: list[dict[str, object]] = []
        for record in self.event_batches:
            batch = record.batch
            for event_index, event in enumerate(batch.events):
                rows.append(
                    {
                        "attempt": record.attempt,
                        "action": record.action,
                        "start_parameter": record.start_parameter,
                        "target_parameter": record.target_parameter,
                        "continuation_parameter": record.continuation_parameter,
                        "solver_load_factor": record.solver_load_factor,
                        "augmentation": record.augmentation,
                        "batch": record.batch_index,
                        "event": event_index,
                        "kind": event.kind,
                        "interface": event.interface,
                        "entity": ":".join(str(value) for value in event.entity),
                        "left_newton_fraction": batch.left_fraction,
                        "event_newton_fraction": batch.event_fraction,
                        "right_newton_fraction": batch.right_fraction,
                        "selected_newton_fraction": batch.selected_fraction,
                        "selected_branch": batch.selected_branch,
                        "path_values": ";".join(
                            f"{name}={value:.17g}" for name, value in record.path_values
                        ),
                        "detail": event.detail,
                    }
                )
        return tuple(rows)


def _event_records(
    attempt: AdaptiveContactAttempt,
    result: object,
) -> tuple[AdaptiveTopologyEventBatch, ...]:
    records: list[AdaptiveTopologyEventBatch] = []
    equilibria = tuple(getattr(result, "equilibria", ()))
    for augmentation, equilibrium in enumerate(equilibria):
        solver_load_factor = float(getattr(equilibrium, "load_factor", 0.0))
        for batch_index, batch in enumerate(tuple(getattr(equilibrium, "events", ()))):
            records.append(
                AdaptiveTopologyEventBatch(
                    attempt=attempt.attempt,
                    action=attempt.action,
                    start_parameter=attempt.start_parameter,
                    target_parameter=attempt.target_parameter,
                    continuation_parameter=attempt.target_parameter,
                    solver_load_factor=solver_load_factor,
                    path_values=attempt.path_values,
                    augmentation=augmentation,
                    batch_index=batch_index,
                    batch=batch,
                )
            )
    return tuple(records)


def solve_event_aware_adaptive_contact_path(
    problem: CoupledEquilibriumProblem,
    target_load_factor: float,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    initial_load_factor: float = 0.0,
    path: CoupledLoadPath | None = None,
    options: AdaptiveContactOptions | None = None,
    event_options: TopologyEventLocalizationOptions | None = None,
    tolerance: float = 1.0e-12,
    _solver: EventAwareAdaptiveSolver | None = None,
) -> EventAwareAdaptiveContactResult:
    """Advance an adaptive path while retaining every localized topology event."""

    settings = AdaptiveContactOptions() if options is None else options
    attempt_results: list[object] = []

    def production_solver(
        candidate_problem: object,
        displacement: FloatArray,
        states: tuple[AugmentedLagrangeState, ...],
        *,
        load_factor: float,
        options: object,
        tolerance: float,
    ) -> object:
        if settings.scaling.enabled:
            return solve_event_aware_scale_aware_augmented_contact(
                candidate_problem,
                displacement,
                states,
                load_factor=load_factor,
                options=options,
                scaling=settings.scaling,
                event_options=event_options,
                tolerance=tolerance,
            )
        return solve_event_aware_augmented_contact(
            candidate_problem,
            displacement,
            states,
            load_factor=load_factor,
            options=options,
            event_options=event_options,
            tolerance=tolerance,
        )

    selected_solver = production_solver if _solver is None else _solver

    def recorded_solver(*args: object, **kwargs: object) -> object:
        result = selected_solver(*args, **kwargs)
        attempt_results.append(result)
        return result

    adaptive = solve_adaptive_contact_path(
        problem,
        target_load_factor,
        initial_displacement,
        initial_states,
        initial_load_factor=initial_load_factor,
        path=path,
        options=settings,
        tolerance=tolerance,
        _solver=recorded_solver,
    )
    if len(attempt_results) != len(adaptive.attempts):
        raise RuntimeError("adaptive event recording lost an attempted solve")
    batches = tuple(
        record
        for attempt, result in zip(adaptive.attempts, attempt_results, strict=True)
        for record in _event_records(attempt, result)
    )
    return EventAwareAdaptiveContactResult(
        adaptive,
        batches,
        tuple(attempt_results),
    )
