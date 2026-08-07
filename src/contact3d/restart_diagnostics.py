"""Restart, cutback, and repeated-localization diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .solvers.continuation import AdaptiveContactAttempt
from .solvers.events.adaptive import (
    AdaptiveTopologyEventBatch,
    EventAwareAdaptiveContactResult,
)
from .topology_model import EventKind

RestartTerminationReason = Literal["restart_loop"]


@dataclass(frozen=True, slots=True)
class RestartDiagnosticOptions:
    """Controls for detecting repeated localization without accepted progress."""

    parameter_tolerance: float = 1.0e-10
    repetition_limit: int = 3

    def __post_init__(self) -> None:
        tolerance = float(self.parameter_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("parameter_tolerance must be finite and positive")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least two")
        object.__setattr__(self, "parameter_tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class RestartEventRecord:
    """One atomic topology event annotated with continuation progress."""

    attempt: int
    action: str
    committed_step: int
    continuation_parameter: float
    augmentation: int
    batch: int
    event: int
    interface: int
    kind: EventKind
    entity: tuple[int, ...]
    selected_branch: str
    selected_signature: str


@dataclass(frozen=True, slots=True)
class RestartCount:
    """Event count for one committed-step/augmentation/interface/kind bucket."""

    committed_step: int
    augmentation: int
    interface: int
    kind: EventKind
    count: int


@dataclass(frozen=True, slots=True)
class RestartAttemptDiagnostic:
    """Restart and cutback counts aligned with one adaptive attempt."""

    attempt: int
    action: str
    committed_step: int
    event_batches: int
    atomic_events: int
    line_search_cutbacks: int
    line_search_iterations_with_cutback: int


@dataclass(frozen=True, slots=True)
class RestartLoopDiagnostic:
    """First repeated event sequence that fails to make accepted progress."""

    termination_reason: RestartTerminationReason
    repetition_count: int
    committed_step: int
    first_attempt: int
    last_attempt: int
    continuation_parameter: float
    interface: int
    kind: EventKind
    entity: tuple[int, ...]
    selected_branch: str
    selected_signature: str


@dataclass(frozen=True, slots=True)
class RestartDiagnostics:
    """Complete restart classification for one adaptive continuation result."""

    events: tuple[RestartEventRecord, ...]
    counts: tuple[RestartCount, ...]
    attempts: tuple[RestartAttemptDiagnostic, ...]
    event_restart_batches: int
    atomic_event_count: int
    line_search_cutbacks: int
    line_search_iterations_with_cutback: int
    adaptive_cutbacks: int
    penalty_retries: int
    loop: RestartLoopDiagnostic | None

    @property
    def termination_reason(self) -> RestartTerminationReason | None:
        return None if self.loop is None else self.loop.termination_reason

    @property
    def healthy(self) -> bool:
        return self.loop is None

    def summary(self) -> dict[str, object]:
        """Return a strict-JSON-compatible summary for benchmark artifacts."""

        loop = None
        if self.loop is not None:
            loop = {
                "termination_reason": self.loop.termination_reason,
                "repetition_count": self.loop.repetition_count,
                "committed_step": self.loop.committed_step,
                "first_attempt": self.loop.first_attempt,
                "last_attempt": self.loop.last_attempt,
                "continuation_parameter": self.loop.continuation_parameter,
                "interface": self.loop.interface,
                "kind": self.loop.kind,
                "entity": list(self.loop.entity),
                "selected_branch": self.loop.selected_branch,
                "selected_signature": self.loop.selected_signature,
            }
        return {
            "healthy": self.healthy,
            "termination_reason": self.termination_reason,
            "event_restart_batches": self.event_restart_batches,
            "atomic_event_count": self.atomic_event_count,
            "line_search_cutbacks": self.line_search_cutbacks,
            "line_search_iterations_with_cutback": (
                self.line_search_iterations_with_cutback
            ),
            "adaptive_cutbacks": self.adaptive_cutbacks,
            "penalty_retries": self.penalty_retries,
            "restart_loop": loop,
        }

    def count_rows(self) -> tuple[dict[str, object], ...]:
        """Return deterministic event-count rows for CSV output."""

        return tuple(
            {
                "committed_step": row.committed_step,
                "augmentation": row.augmentation,
                "interface": row.interface,
                "kind": row.kind,
                "count": row.count,
            }
            for row in self.counts
        )

    def attempt_rows(self) -> tuple[dict[str, object], ...]:
        """Return attempt-level restart and retry classifications."""

        return tuple(
            {
                "attempt": row.attempt,
                "action": row.action,
                "committed_step": row.committed_step,
                "event_batches": row.event_batches,
                "atomic_events": row.atomic_events,
                "line_search_cutbacks": row.line_search_cutbacks,
                "line_search_iterations_with_cutback": (
                    row.line_search_iterations_with_cutback
                ),
            }
            for row in self.attempts
        )


def _selected_signature(record: AdaptiveTopologyEventBatch) -> str:
    signatures = record.batch.selected.signatures
    assert signatures is not None
    payload = []
    for signature in signatures:
        payload.append(
            {
                "facet_pairs": [list(pair) for pair in signature.facet_pairs],
                "active_rows": list(signature.active_rows),
                "supported_rows": list(signature.supported_rows),
                "geometry_tokens": [
                    list(token) for token in getattr(signature, "geometry_tokens", ())
                ],
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _committed_steps(
    attempts: tuple[AdaptiveContactAttempt, ...],
) -> dict[int, int]:
    committed = 0
    result: dict[int, int] = {}
    for attempt in attempts:
        if attempt.action == "accepted":
            committed += 1
        result[attempt.attempt] = committed
    return result


def _line_search_counts(result: object | None) -> tuple[int, int]:
    if result is None:
        return 0, 0
    equilibria = tuple(getattr(result, "equilibria", ()))
    if not equilibria:
        equilibrium = getattr(result, "equilibrium", None)
        equilibria = () if equilibrium is None else (equilibrium,)
    cutbacks = 0
    affected = 0
    for equilibrium in equilibria:
        for iteration in tuple(getattr(equilibrium, "history", ())):
            value = int(getattr(iteration, "line_search_iterations", 0))
            if value < 0:
                raise ValueError("line-search iteration count must be nonnegative")
            cutbacks += value
            affected += value > 0
    return cutbacks, affected


def _event_records(
    batches: tuple[AdaptiveTopologyEventBatch, ...],
    committed_steps: dict[int, int],
) -> tuple[RestartEventRecord, ...]:
    rows: list[RestartEventRecord] = []
    for record in batches:
        if record.attempt not in committed_steps:
            raise ValueError("event batch refers to an unknown adaptive attempt")
        signature = _selected_signature(record)
        for event_index, event in enumerate(record.events):
            rows.append(
                RestartEventRecord(
                    attempt=record.attempt,
                    action=record.action,
                    committed_step=committed_steps[record.attempt],
                    continuation_parameter=record.continuation_parameter,
                    augmentation=record.augmentation,
                    batch=record.batch_index,
                    event=event_index,
                    interface=event.interface,
                    kind=event.kind,
                    entity=tuple(event.entity),
                    selected_branch=event.selected_branch,
                    selected_signature=signature,
                )
            )
    return tuple(rows)


def _restart_counts(events: tuple[RestartEventRecord, ...]) -> tuple[RestartCount, ...]:
    counts = Counter(
        (event.committed_step, event.augmentation, event.interface, event.kind)
        for event in events
    )
    return tuple(
        RestartCount(step, augmentation, interface, kind, count)
        for (step, augmentation, interface, kind), count in sorted(
            counts.items(),
            key=lambda item: item[0],
        )
    )


def _restart_loop(
    events: tuple[RestartEventRecord, ...],
    options: RestartDiagnosticOptions,
) -> RestartLoopDiagnostic | None:
    previous: dict[tuple[object, ...], tuple[RestartEventRecord, int, int]] = {}
    for event in events:
        identity = (
            event.interface,
            event.kind,
            event.entity,
            event.selected_branch,
            event.selected_signature,
        )
        state = previous.get(identity)
        if state is None:
            previous[identity] = (event, 1, event.attempt)
            continue
        prior, count, first_attempt = state
        unchanged = (
            event.committed_step == prior.committed_step
            and abs(event.continuation_parameter - prior.continuation_parameter)
            <= options.parameter_tolerance
        )
        if unchanged:
            count += 1
        else:
            count = 1
            first_attempt = event.attempt
        previous[identity] = (event, count, first_attempt)
        if count >= options.repetition_limit:
            return RestartLoopDiagnostic(
                termination_reason="restart_loop",
                repetition_count=count,
                committed_step=event.committed_step,
                first_attempt=first_attempt,
                last_attempt=event.attempt,
                continuation_parameter=event.continuation_parameter,
                interface=event.interface,
                kind=event.kind,
                entity=event.entity,
                selected_branch=event.selected_branch,
                selected_signature=event.selected_signature,
            )
    return None


def analyze_restart_diagnostics(
    result: EventAwareAdaptiveContactResult,
    *,
    options: RestartDiagnosticOptions | None = None,
) -> RestartDiagnostics:
    """Classify progress, retries, and repeated event localization."""

    settings = RestartDiagnosticOptions() if options is None else options
    attempts = tuple(result.attempts)
    batches = tuple(result.event_batches)
    attempt_results = tuple(getattr(result, "attempt_results", ()))
    if attempt_results and len(attempt_results) != len(attempts):
        raise ValueError("attempt results must align with adaptive attempts")
    if not attempt_results:
        attempt_results = (None,) * len(attempts)

    committed_steps = _committed_steps(attempts)
    events = _event_records(batches, committed_steps)
    attempt_rows: list[RestartAttemptDiagnostic] = []
    total_line_search_cutbacks = 0
    total_affected_iterations = 0
    for attempt, attempt_result in zip(attempts, attempt_results, strict=True):
        line_search_cutbacks, affected_iterations = _line_search_counts(attempt_result)
        attempt_batches = tuple(
            batch for batch in batches if batch.attempt == attempt.attempt
        )
        atomic_events = sum(len(batch.events) for batch in attempt_batches)
        total_line_search_cutbacks += line_search_cutbacks
        total_affected_iterations += affected_iterations
        attempt_rows.append(
            RestartAttemptDiagnostic(
                attempt=attempt.attempt,
                action=attempt.action,
                committed_step=committed_steps[attempt.attempt],
                event_batches=len(attempt_batches),
                atomic_events=atomic_events,
                line_search_cutbacks=line_search_cutbacks,
                line_search_iterations_with_cutback=affected_iterations,
            )
        )

    return RestartDiagnostics(
        events=events,
        counts=_restart_counts(events),
        attempts=tuple(attempt_rows),
        event_restart_batches=len(batches),
        atomic_event_count=len(events),
        line_search_cutbacks=total_line_search_cutbacks,
        line_search_iterations_with_cutback=total_affected_iterations,
        adaptive_cutbacks=sum(attempt.action == "cutback" for attempt in attempts),
        penalty_retries=sum(
            attempt.action == "penalty_increase" for attempt in attempts
        ),
        loop=_restart_loop(events, settings),
    )
