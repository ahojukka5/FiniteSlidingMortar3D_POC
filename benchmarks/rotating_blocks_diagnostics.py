"""Runtime and solver diagnostics for the rotating-blocks benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SCHEMA = "contact3d-rotating-blocks-diagnostics/v1"


@dataclass(frozen=True, slots=True)
class RotatingBlocksDiagnostics:
    """Attempt-level rows and a stable aggregate solver-work summary."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]


def _equilibria(result: object | None) -> tuple[object, ...]:
    return () if result is None else tuple(getattr(result, "equilibria", ()))


def _linear_records(result: object | None) -> tuple[object, ...]:
    records: list[object] = []
    for equilibrium in _equilibria(result):
        for iteration in tuple(getattr(equilibrium, "history", ())):
            diagnostics = getattr(iteration, "linear_solve", None)
            if diagnostics is not None:
                records.append(diagnostics)
        failure = getattr(equilibrium, "linear_solve_failure", None)
        if failure is not None:
            records.append(failure)
    return tuple(records)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _event_counts(result: object | None) -> tuple[int, int]:
    batches = tuple(
        batch
        for equilibrium in _equilibria(result)
        for batch in tuple(getattr(equilibrium, "events", ()))
    )
    return len(batches), sum(len(tuple(getattr(batch, "events", ()))) for batch in batches)


def attempt_diagnostic_rows(result: object) -> tuple[dict[str, object], ...]:
    """Return complete deterministic and provenance fields for every attempt."""

    attempts = tuple(getattr(result, "attempts", ()))
    attempt_results = tuple(getattr(result, "attempt_results", ()))
    if not attempt_results:
        attempt_results = (None,) * len(attempts)
    if len(attempt_results) != len(attempts):
        raise ValueError("attempt results must align with adaptive attempts")

    rows: list[dict[str, object]] = []
    for attempt, attempt_result in zip(attempts, attempt_results, strict=True):
        linear = _linear_records(attempt_result)
        event_batches, event_count = _event_counts(attempt_result)
        backends = _unique(str(item.backend) for item in linear)
        requested = _unique(str(item.requested_backend) for item in linear)
        preconditioners = _unique(str(item.preconditioner) for item in linear)
        rows.append(
            {
                "attempt": int(attempt.attempt),
                "action": str(attempt.action),
                "start_parameter": float(attempt.start_parameter),
                "target_parameter": float(attempt.target_parameter),
                "inner_termination_reason": str(attempt.inner_termination_reason),
                "diagnostics_complete": attempt_result is not None,
                "augmentation_iterations": int(attempt.augmentations),
                "newton_iterations": int(attempt.newton_iterations),
                "line_search_iterations": sum(
                    int(iteration.line_search_iterations)
                    for equilibrium in _equilibria(attempt_result)
                    for iteration in tuple(getattr(equilibrium, "history", ()))
                ),
                "linear_solve_count": len(linear),
                "linear_iterations": sum(int(item.iterations) for item in linear),
                "linear_failures": sum(not bool(item.converged) for item in linear),
                "contact_event_restarts": int(attempt.contact_event_restarts),
                "event_localization_batches": event_batches,
                "event_localization_events": event_count,
                "requested_backends": ",".join(requested),
                "selected_backends": ",".join(backends),
                "preconditioners": ",".join(preconditioners),
                "maximum_matrix_nnz": max(
                    (int(item.matrix_nnz) for item in linear),
                    default=0,
                ),
                "dense_materializations": sum(
                    bool(item.materialized_dense) for item in linear
                ),
                "linear_setup_seconds": sum(float(item.setup_seconds) for item in linear),
                "linear_solve_seconds": sum(float(item.solve_seconds) for item in linear),
            }
        )
    return tuple(rows)


def _work_key(row: dict[str, object]) -> tuple[int, int, int, int]:
    return (
        int(row["newton_iterations"]),
        int(row["linear_iterations"]),
        int(row["event_localization_batches"]),
        int(row["attempt"]),
    )


def _worst(rows: Iterable[dict[str, object]]) -> dict[str, object] | None:
    values = tuple(rows)
    if not values:
        return None
    row = max(values, key=_work_key)
    return {
        "attempt": int(row["attempt"]),
        "action": str(row["action"]),
        "inner_termination_reason": str(row["inner_termination_reason"]),
        "newton_iterations": int(row["newton_iterations"]),
        "linear_iterations": int(row["linear_iterations"]),
        "event_localization_batches": int(row["event_localization_batches"]),
    }


def summarize_diagnostics(
    rows: tuple[dict[str, object], ...],
    *,
    profile: str,
    wall_seconds: float,
) -> dict[str, object]:
    """Aggregate counts while keeping measured times provenance-only."""

    accepted = tuple(row for row in rows if row["action"] == "accepted")
    failed = tuple(row for row in rows if row["action"] != "accepted")
    linear_setup = sum(float(row["linear_setup_seconds"]) for row in rows)
    linear_solve = sum(float(row["linear_solve_seconds"]) for row in rows)
    dense_materializations = sum(int(row["dense_materializations"]) for row in rows)
    selected_backends = _unique(
        backend
        for row in rows
        for backend in str(row["selected_backends"]).split(",")
    )
    deterministic = {
        "attempts": len(rows),
        "accepted_attempts": len(accepted),
        "rejected_or_retried_attempts": len(failed),
        "augmentation_iterations": sum(
            int(row["augmentation_iterations"]) for row in rows
        ),
        "newton_iterations": sum(int(row["newton_iterations"]) for row in rows),
        "line_search_iterations": sum(
            int(row["line_search_iterations"]) for row in rows
        ),
        "linear_solves": sum(int(row["linear_solve_count"]) for row in rows),
        "linear_iterations": sum(int(row["linear_iterations"]) for row in rows),
        "linear_failures": sum(int(row["linear_failures"]) for row in rows),
        "event_localization_batches": sum(
            int(row["event_localization_batches"]) for row in rows
        ),
        "event_localization_events": sum(
            int(row["event_localization_events"]) for row in rows
        ),
        "dense_materializations": dense_materializations,
        "maximum_matrix_nnz": max(
            (int(row["maximum_matrix_nnz"]) for row in rows),
            default=0,
        ),
    }
    return {
        "schema_version": SCHEMA,
        "diagnostics_complete": all(bool(row["diagnostics_complete"]) for row in rows),
        "selected_backends": selected_backends,
        "sparse_profile_avoided_dense_materialization": (
            profile != "full" or dense_materializations == 0
        ),
        "deterministic_counts": deterministic,
        "worst_accepted_attempt": _worst(accepted),
        "worst_failed_attempt": _worst(failed),
        "provenance_timings": {
            "wall_seconds": float(wall_seconds),
            "linear_setup_seconds": linear_setup,
            "linear_solve_seconds": linear_solve,
            "unattributed_seconds": max(
                0.0,
                float(wall_seconds) - linear_setup - linear_solve,
            ),
        },
        "timings_used_for_acceptance": False,
    }


def collect_solver_diagnostics(
    result: object,
    *,
    profile: str,
    wall_seconds: float,
) -> RotatingBlocksDiagnostics:
    """Collect one complete diagnostics bundle for a production run."""

    rows = attempt_diagnostic_rows(result)
    return RotatingBlocksDiagnostics(
        rows,
        summarize_diagnostics(rows, profile=profile, wall_seconds=wall_seconds),
    )
