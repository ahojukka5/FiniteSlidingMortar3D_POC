#!/usr/bin/env python3
"""Compare rotating-blocks continuation histories under path refinement."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from rotating_blocks_profiles import (
    RotatingBlocksExecutionProfile,
    rotating_blocks_execution_profile,
)
from rotating_blocks_solver import RotatingBlocksSolverRun
from rotating_blocks_solver import run as run_solver
from svg_plots import write_line_chart

SCHEMA = "contact3d-rotating-blocks-refinement/v1"
Runner = Callable[[RotatingBlocksExecutionProfile], RotatingBlocksSolverRun]
FIELDS = (
    "reaction_x",
    "reaction_y",
    "reaction_z",
    "maximum_pressure",
    "overlap_area",
)


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    """One requested path resolution and its completed production solve."""

    requested_steps: int
    run: RotatingBlocksSolverRun

    @property
    def cutbacks(self) -> int:
        return int(self.run.summary["cutback_count"])


@dataclass(frozen=True, slots=True)
class RotatingBlocksRefinement:
    """Three-level convergence evidence on one shared comparison grid."""

    profile: RotatingBlocksExecutionProfile
    levels: tuple[RefinementLevel, ...]
    comparison_parameters: tuple[float, ...]
    comparison_rows: tuple[dict[str, object], ...]
    event_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def _refined_profile(
    profile: RotatingBlocksExecutionProfile,
    requested_steps: int,
) -> RotatingBlocksExecutionProfile:
    increment = 1.0 / requested_steps
    return replace(
        profile,
        requested_path_steps=requested_steps,
        initial_step=increment,
        maximum_step=increment,
        minimum_step=min(profile.minimum_step, increment / 64.0),
        maximum_attempts=max(profile.maximum_attempts, 8 * requested_steps),
        refinement_steps=tuple(sorted(set((*profile.refinement_steps, requested_steps)))),
    )


def _interpolate(
    rows: tuple[dict[str, object], ...],
    parameters: np.ndarray,
    field: str,
) -> np.ndarray:
    source_parameters = np.asarray([float(row["parameter"]) for row in rows])
    source_values = np.asarray([float(row[field]) for row in rows])
    if len(source_parameters) == 0:
        raise ValueError("refinement run contains no accepted states")
    order = np.argsort(source_parameters)
    source_parameters = source_parameters[order]
    source_values = source_values[order]
    unique, indices = np.unique(source_parameters, return_index=True)
    return np.interp(parameters, unique, source_values[indices])


def _relative_error(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(reference), 1.0e-14)
    return np.abs(value - reference) / scale


def _comparison_rows(
    levels: tuple[RefinementLevel, ...],
    parameters: np.ndarray,
) -> tuple[dict[str, object], ...]:
    medium = levels[-2].run.accepted_rows
    fine = levels[-1].run.accepted_rows
    interpolated = {
        field: (
            _interpolate(medium, parameters, field),
            _interpolate(fine, parameters, field),
        )
        for field in FIELDS
    }
    rows: list[dict[str, object]] = []
    for index, parameter in enumerate(parameters):
        row: dict[str, object] = {"parameter": float(parameter)}
        for field, (medium_values, fine_values) in interpolated.items():
            row[f"medium_{field}"] = float(medium_values[index])
            row[f"fine_{field}"] = float(fine_values[index])
            row[f"absolute_error_{field}"] = float(
                abs(medium_values[index] - fine_values[index])
            )
            row[f"relative_error_{field}"] = float(
                _relative_error(fine_values, medium_values)[index]
            )
        rows.append(row)
    return tuple(rows)


def _event_key(row: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("kind", "")),
        str(row.get("entity", "")),
        int(row.get("interface", 0)),
    )


def _event_positions(
    rows: tuple[dict[str, object], ...],
) -> dict[tuple[str, str, int], list[float]]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault(_event_key(row), []).append(
            float(row["continuation_parameter"])
        )
    for values in grouped.values():
        values.sort()
    return grouped


def _event_comparison_rows(
    medium: RotatingBlocksSolverRun,
    fine: RotatingBlocksSolverRun,
) -> tuple[dict[str, object], ...]:
    medium_events = _event_positions(medium.event_rows)
    fine_events = _event_positions(fine.event_rows)
    keys = sorted(set(medium_events) | set(fine_events))
    rows: list[dict[str, object]] = []
    for key in keys:
        medium_values = medium_events.get(key, [])
        fine_values = fine_events.get(key, [])
        count = max(len(medium_values), len(fine_values))
        for occurrence in range(count):
            medium_value = (
                medium_values[occurrence]
                if occurrence < len(medium_values)
                else None
            )
            fine_value = (
                fine_values[occurrence] if occurrence < len(fine_values) else None
            )
            error = (
                abs(medium_value - fine_value)
                if medium_value is not None and fine_value is not None
                else None
            )
            rows.append(
                {
                    "kind": key[0],
                    "entity": key[1],
                    "interface": key[2],
                    "occurrence": occurrence,
                    "medium_parameter": medium_value,
                    "fine_parameter": fine_value,
                    "absolute_error": error,
                }
            )
    return tuple(rows)


def _summary(
    profile: RotatingBlocksExecutionProfile,
    levels: tuple[RefinementLevel, ...],
    comparison_rows: tuple[dict[str, object], ...],
    event_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    field_errors = {
        field: max(float(row[f"relative_error_{field}"]) for row in comparison_rows)
        for field in FIELDS
    }
    event_errors = [
        float(row["absolute_error"])
        for row in event_rows
        if row["absolute_error"] is not None
    ]
    event_counts_match = all(
        row["medium_parameter"] is not None and row["fine_parameter"] is not None
        for row in event_rows
    )
    final_parameters = [float(level.run.summary["final_parameter"]) for level in levels]
    final_states = [level.run.accepted_rows[-1] for level in levels]
    event_tolerance = 2.0 / levels[-2].requested_steps
    criteria = {
        "all_runs_passed": all(level.run.passed for level in levels),
        "final_motion_reached": all(
            np.isclose(value, 1.0, rtol=0.0, atol=1.0e-12)
            for value in final_parameters
        ),
        "final_contact_state_matches": all(
            int(row["active_rows"]) == int(final_states[-1]["active_rows"])
            and int(row["supported_rows"])
            == int(final_states[-1]["supported_rows"])
            and int(row["facet_pairs"]) == int(final_states[-1]["facet_pairs"])
            for row in final_states[:-1]
        ),
        "medium_fine_fields_converged": max(field_errors.values()) <= 5.0e-2,
        "event_counts_match": event_counts_match,
        "event_locations_converged": (
            max(event_errors, default=0.0) <= event_tolerance
        ),
    }
    return {
        "schema_version": SCHEMA,
        "profile": profile.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "requested_steps": [level.requested_steps for level in levels],
        "adaptive_cutbacks": [level.cutbacks for level in levels],
        "maximum_relative_field_errors": field_errors,
        "maximum_event_location_error": max(event_errors, default=0.0),
    }


def run(
    profile: str | RotatingBlocksExecutionProfile = "full",
    *,
    raise_on_failure: bool = True,
    _runner: Runner | None = None,
) -> RotatingBlocksRefinement:
    """Run and compare all configured path resolutions."""

    selected = rotating_blocks_execution_profile(profile)
    runner = (
        (lambda item: run_solver(item, raise_on_failure=True))
        if _runner is None
        else _runner
    )
    levels = tuple(
        RefinementLevel(steps, runner(_refined_profile(selected, steps)))
        for steps in selected.refinement_steps
    )
    finest_steps = levels[-1].requested_steps
    parameters = np.linspace(0.0, 1.0, finest_steps + 1)
    comparison_rows = _comparison_rows(levels, parameters)
    event_rows = _event_comparison_rows(levels[-2].run, levels[-1].run)
    summary = _summary(selected, levels, comparison_rows, event_rows)
    completed = RotatingBlocksRefinement(
        selected,
        levels,
        tuple(float(value) for value in parameters),
        comparison_rows,
        event_rows,
        summary,
    )
    if raise_on_failure and not completed.passed:
        criteria = summary["criteria"]
        assert isinstance(criteria, dict)
        failed = [name for name, passed in criteria.items() if not passed]
        raise RuntimeError("rotating-blocks refinement failed: " + ", ".join(failed))
    return completed


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results(output: Path, result: RotatingBlocksRefinement) -> None:
    """Write machine-readable convergence tables and one overview plot."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_csv(output / "field-comparison.csv", result.comparison_rows)
    _write_csv(output / "event-comparison.csv", result.event_rows)
    parameters = np.asarray(result.comparison_parameters)
    write_line_chart(
        output / "refinement-error.svg",
        title="Rotating-blocks medium/fine path refinement",
        x_label="continuation parameter",
        y_label="relative error",
        x_values=parameters,
        series=tuple(
            (
                np.asarray(
                    [
                        float(row[f"relative_error_{field}"])
                        for row in result.comparison_rows
                    ]
                ),
                field,
            )
            for field in FIELDS
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    completed = run(arguments.profile)
    write_results(arguments.output, completed)
    print(json.dumps(completed.summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
