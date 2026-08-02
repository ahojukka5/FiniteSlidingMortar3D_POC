from __future__ import annotations

from types import SimpleNamespace

import pytest

from contact3d.adaptive import AdaptiveContactAttempt
from contact3d.event_adaptive import (
    AdaptiveTopologyEventBatch,
    EventAwareAdaptiveContactResult,
)
from contact3d.event_solver import (
    RestartDiagnosticOptions,
    analyze_restart_diagnostics,
)
from contact3d.topology_events import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    ContactTopologySignature,
    TopologyObservation,
)


def _attempt(index: int, action: str, target: float) -> AdaptiveContactAttempt:
    return AdaptiveContactAttempt(
        index,
        0.0,
        target,
        target,
        action,
        "converged",
        1,
        2,
        1,
        0.0,
        0.0,
        (3200.0,),
        (3200.0,),
    )


def _event_batch(
    attempt: int,
    action: str,
    parameter: float,
    *,
    kind: str = "pair_entry",
    interface: int = 0,
    entity: tuple[int, ...] = (2, 3),
    augmentation: int = 0,
) -> AdaptiveTopologyEventBatch:
    signature = ContactTopologySignature(
        ((2, 3),),
        (True, False),
        (True, True),
        ((2, 3, 4, 4, 1),),
    )
    selected = TopologyObservation.valid(0.6, (signature,))
    event = ContactTopologyEvent(
        kind,
        interface,
        entity,
        0.5,
        "right",
        "synthetic transition",
    )
    batch = ContactTopologyEventBatch(
        "localized",
        0.4,
        0.5,
        0.6,
        0.6,
        "right",
        (event,),
        selected,
    )
    return AdaptiveTopologyEventBatch(
        attempt,
        action,
        0.0,
        parameter,
        parameter,
        1.0,
        (),
        augmentation,
        0,
        batch,
    )


def _attempt_result(*line_search_iterations: int) -> SimpleNamespace:
    history = tuple(
        SimpleNamespace(line_search_iterations=value)
        for value in line_search_iterations
    )
    return SimpleNamespace(equilibria=(SimpleNamespace(history=history),))


def _result(
    attempts: tuple[AdaptiveContactAttempt, ...],
    batches: tuple[AdaptiveTopologyEventBatch, ...],
    attempt_results: tuple[object, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        attempts=attempts,
        event_batches=batches,
        attempt_results=attempt_results,
    )


def test_progressing_events_do_not_trigger_restart_loop() -> None:
    attempts = (
        _attempt(1, "cutback", 0.4),
        _attempt(2, "accepted", 0.4),
        _attempt(3, "accepted", 0.6),
    )
    batches = (
        _event_batch(1, "cutback", 0.4),
        _event_batch(2, "accepted", 0.4),
        _event_batch(3, "accepted", 0.6),
    )
    diagnostics = analyze_restart_diagnostics(
        _result(
            attempts,
            batches,
            (_attempt_result(1), _attempt_result(0), _attempt_result(2)),
        ),
        options=RestartDiagnosticOptions(repetition_limit=2),
    )

    assert diagnostics.healthy
    assert diagnostics.termination_reason is None
    assert diagnostics.event_restart_batches == 3
    assert diagnostics.atomic_event_count == 3
    assert diagnostics.adaptive_cutbacks == 1
    assert diagnostics.penalty_retries == 0
    assert diagnostics.line_search_cutbacks == 3
    assert diagnostics.line_search_iterations_with_cutback == 2
    assert [event.committed_step for event in diagnostics.events] == [0, 1, 2]


def test_repeated_same_state_event_reports_restart_loop() -> None:
    attempts = tuple(_attempt(index, "cutback", 0.5) for index in (1, 2, 3))
    batches = tuple(
        _event_batch(index, "cutback", 0.5) for index in (1, 2, 3)
    )
    diagnostics = analyze_restart_diagnostics(
        _result(
            attempts,
            batches,
            (_attempt_result(2, 0), _attempt_result(1), _attempt_result(0)),
        ),
        options=RestartDiagnosticOptions(
            parameter_tolerance=1.0e-9,
            repetition_limit=3,
        ),
    )

    assert not diagnostics.healthy
    assert diagnostics.termination_reason == "restart_loop"
    assert diagnostics.loop is not None
    assert diagnostics.loop.repetition_count == 3
    assert diagnostics.loop.first_attempt == 1
    assert diagnostics.loop.last_attempt == 3
    assert diagnostics.loop.committed_step == 0
    assert diagnostics.loop.kind == "pair_entry"
    assert diagnostics.loop.entity == (2, 3)
    assert '"facet_pairs":[[2,3]]' in diagnostics.loop.selected_signature
    assert diagnostics.summary()["restart_loop"] is not None


def test_counts_and_attempt_rows_separate_retry_classes() -> None:
    attempts = (
        _attempt(1, "cutback", 0.25),
        _attempt(2, "penalty_increase", 0.25),
        _attempt(3, "accepted", 0.25),
    )
    batches = (
        _event_batch(1, "cutback", 0.25, kind="pair_entry"),
        _event_batch(
            2,
            "penalty_increase",
            0.25,
            kind="support_activation",
            interface=1,
            entity=(4,),
            augmentation=2,
        ),
    )
    diagnostics = analyze_restart_diagnostics(
        _result(
            attempts,
            batches,
            (_attempt_result(1), _attempt_result(0), _attempt_result(0)),
        )
    )

    assert diagnostics.adaptive_cutbacks == 1
    assert diagnostics.penalty_retries == 1
    assert diagnostics.count_rows() == (
        {
            "committed_step": 0,
            "augmentation": 0,
            "interface": 0,
            "kind": "pair_entry",
            "count": 1,
        },
        {
            "committed_step": 0,
            "augmentation": 2,
            "interface": 1,
            "kind": "support_activation",
            "count": 1,
        },
    )
    rows = diagnostics.attempt_rows()
    assert [row["action"] for row in rows] == [
        "cutback",
        "penalty_increase",
        "accepted",
    ]
    assert [row["event_batches"] for row in rows] == [1, 1, 0]


def test_event_result_retains_aligned_attempt_results() -> None:
    attempts = (_attempt(1, "accepted", 0.25),)
    adaptive = SimpleNamespace(attempts=attempts)
    attempt_result = _attempt_result(0)
    result = EventAwareAdaptiveContactResult(adaptive, (), (attempt_result,))

    assert result.attempt_results == (attempt_result,)
    with pytest.raises(ValueError, match="align"):
        EventAwareAdaptiveContactResult(adaptive, (), (attempt_result, attempt_result))


def test_restart_diagnostic_validation() -> None:
    with pytest.raises(ValueError, match="parameter_tolerance"):
        RestartDiagnosticOptions(parameter_tolerance=0.0)
    with pytest.raises(ValueError, match="repetition_limit"):
        RestartDiagnosticOptions(repetition_limit=1)

    attempts = (_attempt(1, "accepted", 0.25),)
    foreign_batch = _event_batch(2, "accepted", 0.25)
    with pytest.raises(ValueError, match="unknown adaptive attempt"):
        analyze_restart_diagnostics(
            _result(attempts, (foreign_batch,), (_attempt_result(0),))
        )
