"""Profile-aware acceptance gate for the rotating-blocks benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rotating_blocks_model import build_rotating_blocks_model
from rotating_blocks_profiles import (
    RotatingBlocksExecutionProfile,
    rotating_blocks_execution_profile,
)

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.repetition import RepetitionTolerances, compare_kinematic_topology_scans
from contact3d.topology_scan import scan_kinematic_contact_path

SCHEMA = "contact3d-rotating-blocks-acceptance-gate/v1"
CRITERION_SCHEMA = "contact3d-rotating-blocks-acceptance-criteria/v1"


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    """Versioned numerical limits for one execution profile."""

    profile: str
    final_parameter_tolerance: float
    minimum_rotation_overlap_area: float
    minimum_rotation_supported_rows: int
    maximum_normalized_equilibrium_residual: float
    maximum_normalized_penetration: float
    maximum_normalized_force_error: float
    maximum_normalized_moment_error: float
    maximum_refinement_relative_error: float
    maximum_event_location_error: float
    repetition_absolute_tolerance: float
    repetition_relative_tolerance: float

    def __post_init__(self) -> None:
        if self.profile not in ("quick", "full"):
            raise ValueError("acceptance profile must be 'quick' or 'full'")
        numeric = (
            self.final_parameter_tolerance,
            self.minimum_rotation_overlap_area,
            self.maximum_normalized_equilibrium_residual,
            self.maximum_normalized_penetration,
            self.maximum_normalized_force_error,
            self.maximum_normalized_moment_error,
            self.maximum_refinement_relative_error,
            self.maximum_event_location_error,
            self.repetition_absolute_tolerance,
            self.repetition_relative_tolerance,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("acceptance thresholds must be finite and nonnegative")
        if self.minimum_rotation_supported_rows <= 0:
            raise ValueError("minimum supported-row count must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


QUICK_THRESHOLDS = AcceptanceThresholds(
    profile="quick",
    final_parameter_tolerance=1.0e-12,
    minimum_rotation_overlap_area=1.0e-12,
    minimum_rotation_supported_rows=1,
    maximum_normalized_equilibrium_residual=1.0e-8,
    maximum_normalized_penetration=1.0e-7,
    maximum_normalized_force_error=1.0e-7,
    maximum_normalized_moment_error=1.0e-7,
    maximum_refinement_relative_error=5.0e-2,
    maximum_event_location_error=1.25e-1,
    repetition_absolute_tolerance=1.0e-12,
    repetition_relative_tolerance=1.0e-10,
)
FULL_THRESHOLDS = AcceptanceThresholds(
    profile="full",
    final_parameter_tolerance=1.0e-12,
    minimum_rotation_overlap_area=1.0e-12,
    minimum_rotation_supported_rows=1,
    maximum_normalized_equilibrium_residual=1.0e-8,
    maximum_normalized_penetration=1.0e-7,
    maximum_normalized_force_error=1.0e-7,
    maximum_normalized_moment_error=1.0e-7,
    maximum_refinement_relative_error=5.0e-2,
    maximum_event_location_error=3.125e-2,
    repetition_absolute_tolerance=1.0e-12,
    repetition_relative_tolerance=1.0e-10,
)
THRESHOLDS = {
    QUICK_THRESHOLDS.profile: QUICK_THRESHOLDS,
    FULL_THRESHOLDS.profile: FULL_THRESHOLDS,
}


@dataclass(frozen=True, slots=True)
class RotatingBlocksAcceptanceGate:
    """Complete criterion rows and their aggregate assessment."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def acceptance_thresholds(
    profile: str | RotatingBlocksExecutionProfile,
) -> AcceptanceThresholds:
    selected = rotating_blocks_execution_profile(profile)
    return THRESHOLDS[selected.name]


def run_event_determinism(
    profile: str | RotatingBlocksExecutionProfile,
) -> dict[str, object]:
    """Repeat the kinematic topology oracle from two clean model instances."""

    selected = rotating_blocks_execution_profile(profile)
    limits = acceptance_thresholds(selected)
    parameters = np.linspace(0.0, 1.0, selected.topology_samples)
    first_model = build_rotating_blocks_model(selected.model_profile)
    second_model = build_rotating_blocks_model(selected.model_profile)
    first = scan_kinematic_contact_path(first_model.problem, first_model.path, parameters)
    second = scan_kinematic_contact_path(second_model.problem, second_model.path, parameters)
    comparison = compare_kinematic_topology_scans(
        first,
        second,
        tolerances=RepetitionTolerances(
            absolute=limits.repetition_absolute_tolerance,
            relative=limits.repetition_relative_tolerance,
        ),
    )
    return {
        "schema_version": "contact3d-rotating-blocks-gate-determinism/v1",
        "profile": selected.name,
        "sample_count": selected.topology_samples,
        **comparison.summary(),
    }


def _criterion(
    name: str,
    category: str,
    observed: object,
    relation: str,
    limit: object,
    passed: bool,
) -> dict[str, object]:
    return {
        "criterion": name,
        "category": category,
        "observed": observed,
        "relation": relation,
        "limit": limit,
        "passed": bool(passed),
        "message": (
            f"{name}: observed={observed!r}; required {relation} {limit!r}"
        ),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _maximum(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if field in row]
    return max(values, default=float("inf"))


def _maximum_prefixed(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
) -> float:
    values = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key.startswith(prefix) and isinstance(value, (int, float, np.generic))
    ]
    return max(values, default=float("inf"))


def _refinement_field_error(
    refinement: object,
    summary: Mapping[str, object],
) -> float:
    recorded = _mapping(summary.get("maximum_relative_field_errors"))
    if recorded:
        return max(float(value) for value in recorded.values())
    rows = tuple(getattr(refinement, "comparison_rows", ()))
    return _maximum_prefixed(rows, "relative_error_")


def _refinement_event_error(
    refinement: object,
    summary: Mapping[str, object],
) -> float:
    if "maximum_event_location_error" in summary:
        return float(summary["maximum_event_location_error"])
    values = [
        float(row["absolute_error"])
        for row in tuple(getattr(refinement, "event_rows", ()))
        if row.get("absolute_error") is not None
    ]
    return max(values, default=0.0)


def _balance_error(summary: Mapping[str, object], kind: str) -> float:
    maxima = _mapping(summary.get("maximum_normalized_errors"))
    values = [float(value) for key, value in maxima.items() if kind in key]
    return max(values, default=float("inf"))


def evaluate_acceptance_gate(
    completed: object,
    refinement: object,
    balance_summary: Mapping[str, object],
    pressure_summary: Mapping[str, object],
    *,
    determinism: Mapping[str, object] | None = None,
) -> RotatingBlocksAcceptanceGate:
    """Evaluate every required benchmark criterion without short-circuiting."""

    profile = rotating_blocks_execution_profile(completed.profile)
    limits = acceptance_thresholds(profile)
    solver_summary = _mapping(completed.summary)
    solver_criteria = _mapping(solver_summary.get("criteria"))
    accepted = tuple(getattr(completed, "accepted_rows", ()))
    attempts = tuple(getattr(completed, "attempt_rows", ()))
    accepted_attempts = tuple(row for row in attempts if row.get("action") == "accepted")
    rotation = tuple(row for row in accepted if int(row.get("phase_index", -1)) == 1)

    final_parameter = float(
        solver_summary.get(
            "final_parameter",
            accepted[-1]["parameter"] if accepted else float("-inf"),
        )
    )
    maximum_residual = float(
        solver_summary.get(
            "maximum_normalized_equilibrium_residual",
            _maximum(accepted_attempts, "normalized_equilibrium_residual"),
        )
    )
    maximum_penetration = float(
        solver_summary.get(
            "maximum_normalized_penetration",
            _maximum(accepted_attempts, "normalized_maximum_penetration"),
        )
    )
    minimum_overlap = min(
        (float(row["overlap_area"]) for row in rotation),
        default=0.0,
    )
    minimum_supported = min(
        (int(row["supported_rows"]) for row in rotation),
        default=0,
    )
    refinement_summary = _mapping(refinement.summary)
    refinement_criteria = _mapping(refinement_summary.get("criteria"))
    field_error = _refinement_field_error(refinement, refinement_summary)
    event_error = _refinement_event_error(refinement, refinement_summary)
    force_error = _balance_error(balance_summary, "force")
    moment_error = _balance_error(balance_summary, "moment")
    deterministic = (
        run_event_determinism(profile) if determinism is None else dict(determinism)
    )
    deterministic_passed = bool(deterministic.get("passed", False))
    deterministic_absolute = float(
        deterministic.get("maximum_absolute_error", float("inf"))
    )
    deterministic_relative = float(
        deterministic.get("maximum_relative_error", float("inf"))
    )

    rows = (
        _criterion(
            "solver_converged",
            "convergence",
            bool(solver_criteria.get("solver_converged", completed.passed)),
            "==",
            True,
            bool(solver_criteria.get("solver_converged", completed.passed)),
        ),
        _criterion(
            "final_motion_reached",
            "convergence",
            final_parameter,
            f"within ±{limits.final_parameter_tolerance:g} of",
            1.0,
            abs(final_parameter - 1.0) <= limits.final_parameter_tolerance,
        ),
        _criterion(
            "rotation_overlap_retained",
            "contact_retention",
            minimum_overlap,
            ">=",
            limits.minimum_rotation_overlap_area,
            bool(rotation) and minimum_overlap >= limits.minimum_rotation_overlap_area,
        ),
        _criterion(
            "rotation_support_retained",
            "contact_retention",
            minimum_supported,
            ">=",
            limits.minimum_rotation_supported_rows,
            bool(rotation) and minimum_supported >= limits.minimum_rotation_supported_rows,
        ),
        _criterion(
            "normalized_equilibrium_residual",
            "kkt",
            maximum_residual,
            "<=",
            limits.maximum_normalized_equilibrium_residual,
            maximum_residual <= limits.maximum_normalized_equilibrium_residual,
        ),
        _criterion(
            "normalized_penetration",
            "kkt",
            maximum_penetration,
            "<=",
            limits.maximum_normalized_penetration,
            maximum_penetration <= limits.maximum_normalized_penetration,
        ),
        _criterion(
            "normalized_force_balance",
            "balance",
            force_error,
            "<=",
            limits.maximum_normalized_force_error,
            force_error <= limits.maximum_normalized_force_error,
        ),
        _criterion(
            "normalized_moment_balance",
            "balance",
            moment_error,
            "<=",
            limits.maximum_normalized_moment_error,
            moment_error <= limits.maximum_normalized_moment_error,
        ),
        _criterion(
            "event_history_deterministic",
            "determinism",
            deterministic_passed,
            "==",
            True,
            deterministic_passed,
        ),
        _criterion(
            "determinism_absolute_error",
            "determinism",
            deterministic_absolute,
            "<=",
            limits.repetition_absolute_tolerance,
            deterministic_absolute <= limits.repetition_absolute_tolerance,
        ),
        _criterion(
            "determinism_relative_error",
            "determinism",
            deterministic_relative,
            "<=",
            limits.repetition_relative_tolerance,
            deterministic_relative <= limits.repetition_relative_tolerance,
        ),
        _criterion(
            "refinement_field_agreement",
            "refinement",
            field_error,
            "<=",
            limits.maximum_refinement_relative_error,
            field_error <= limits.maximum_refinement_relative_error,
        ),
        _criterion(
            "refinement_event_locations",
            "refinement",
            event_error,
            "<=",
            limits.maximum_event_location_error,
            event_error <= limits.maximum_event_location_error,
        ),
        _criterion(
            "refinement_event_counts_match",
            "refinement",
            bool(refinement_criteria.get("event_counts_match", refinement.passed)),
            "==",
            True,
            bool(refinement_criteria.get("event_counts_match", refinement.passed)),
        ),
        _criterion(
            "pressure_redistribution_passed",
            "pressure",
            bool(pressure_summary.get("passed", False)),
            "==",
            True,
            bool(pressure_summary.get("passed", False)),
        ),
    )
    failed = tuple(row for row in rows if not row["passed"])
    summary = {
        "schema_version": SCHEMA,
        "profile": profile.name,
        "passed": not failed,
        "thresholds": limits.as_dict(),
        "criterion_count": len(rows),
        "failed_count": len(failed),
        "criteria": list(rows),
        "failed_criteria": [str(row["criterion"]) for row in failed],
        "failure_messages": [str(row["message"]) for row in failed],
        "determinism": deterministic,
    }
    return RotatingBlocksAcceptanceGate(rows, summary)


def write_gate_artifacts(
    writer: BenchmarkArtifactWriter,
    gate: RotatingBlocksAcceptanceGate,
) -> tuple[str, ...]:
    """Write the complete criterion table and aggregate result."""

    paths = ("tables/acceptance-gate.csv", "acceptance-gate.json")
    writer.write_csv(paths[0], gate.rows, schema=CRITERION_SCHEMA)
    writer.write_json(paths[1], gate.summary, schema=SCHEMA)
    return paths


def acceptance_failure_message(summary: Mapping[str, object]) -> str:
    """Format all failed metrics with their observed values and limits."""

    messages = tuple(str(value) for value in summary.get("failure_messages", ()))
    if not messages:
        return "rotating-blocks acceptance gate failed without criterion details"
    return "rotating-blocks acceptance gate failed:\n- " + "\n- ".join(messages)


def write_standalone_summary(path: Path, gate: RotatingBlocksAcceptanceGate) -> None:
    """Write a standalone strict JSON result for focused gate invocations."""

    import json

    Path(path).write_text(
        json.dumps(gate.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
