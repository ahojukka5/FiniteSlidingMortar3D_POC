#!/usr/bin/env python3
"""Run rotating blocks through the production adaptive contact solver."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rotating_blocks_model import build_rotating_blocks_model
from rotating_blocks_profiles import (
    RotatingBlocksExecutionProfile,
    rotating_blocks_execution_profile,
)

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    ScaleAwareConvergenceOptions,
)
from contact3d.coupled import AugmentedContactOptions
from contact3d.equilibrium import NewtonOptions
from contact3d.event_solver import (
    analyze_restart_diagnostics,
    solve_event_aware_adaptive_contact_path,
)

SCHEMA = "contact3d-rotating-blocks-solver/v1"
Solver = Callable[..., object]


@dataclass(frozen=True, slots=True)
class RotatingBlocksSolverRun:
    """Production solve plus stable rows for later artifact and refinement work."""

    profile: RotatingBlocksExecutionProfile
    result: object
    summary: dict[str, object]
    accepted_rows: tuple[dict[str, object], ...]
    attempt_rows: tuple[dict[str, object], ...]
    event_rows: tuple[dict[str, object], ...]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def solver_options(profile: RotatingBlocksExecutionProfile) -> AdaptiveContactOptions:
    """Map a benchmark profile onto production adaptive solver controls."""

    scaling = ScaleAwareConvergenceOptions(
        enabled=True,
        equilibrium_tolerance=1.0e-8,
        gap_tolerance=1.0e-7,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-7,
        multiplier_tolerance=1.0e-7,
    )
    augmented = AugmentedContactOptions(
        maximum_augmentations=16,
        gap_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-8,
        projection_tolerance=1.0e-8,
        multiplier_tolerance=1.0e-8,
        event_policy="restart",
        newton=NewtonOptions(
            maximum_iterations=40 if profile.name == "quick" else 50,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
        ),
    )
    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=profile.initial_step,
            minimum_step=profile.minimum_step,
            maximum_step=profile.maximum_step,
            maximum_attempts=profile.maximum_attempts,
        ),
        penalty=AdaptivePenaltyOptions(
            enabled=True,
            normalized_penetration_target=scaling.gap_tolerance,
            maximum_updates_per_step=4,
            interface_local=True,
        ),
        augmented=augmented,
        scaling=scaling,
    )


def _accepted_rows(result: object) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, step in enumerate(tuple(getattr(result, "accepted_steps", ())), start=1):
        state = step.path_state
        value = getattr(state, "value")
        rows.append(
            {
                "accepted_step": index,
                "parameter": float(step.parameter),
                "phase_index": int(round(float(value("phase_index")))),
                "phase_parameter": float(value("phase_parameter")),
                "rotation_angle": float(value("rotation_angle")),
                "reaction_norm": float(step.reaction_norm),
                "inner_converged": bool(step.result.converged),
                "inner_termination_reason": str(step.result.termination_reason),
            }
        )
    return tuple(rows)


def _attempt_rows(result: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "attempt": int(attempt.attempt),
            "start_parameter": float(attempt.start_parameter),
            "target_parameter": float(attempt.target_parameter),
            "step_size": float(attempt.step_size),
            "action": str(attempt.action),
            "inner_termination_reason": str(attempt.inner_termination_reason),
            "augmentations": int(attempt.augmentations),
            "newton_iterations": int(attempt.newton_iterations),
            "contact_event_restarts": int(attempt.contact_event_restarts),
            "normalized_equilibrium_residual": float(
                attempt.normalized_equilibrium_residual
            ),
            "normalized_maximum_penetration": float(
                attempt.normalized_maximum_penetration
            ),
        }
        for attempt in tuple(getattr(result, "attempts", ()))
    )


def _event_counts(event_rows: tuple[dict[str, object], ...]) -> dict[str, int]:
    kinds = tuple(str(row["kind"]) for row in event_rows)
    return {
        "event_count": len(kinds),
        "pair_entries": kinds.count("pair_entry"),
        "pair_exits": kinds.count("pair_exit"),
        "support_activations": kinds.count("support_activation"),
        "support_releases": kinds.count("support_release"),
    }


def _summary(
    profile: RotatingBlocksExecutionProfile,
    result: object,
    accepted_rows: tuple[dict[str, object], ...],
    attempt_rows: tuple[dict[str, object], ...],
    event_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    diagnostics = analyze_restart_diagnostics(result)
    accepted_attempts = tuple(row for row in attempt_rows if row["action"] == "accepted")
    max_residual = max(
        (float(row["normalized_equilibrium_residual"]) for row in accepted_attempts),
        default=float("inf"),
    )
    max_penetration = max(
        (float(row["normalized_maximum_penetration"]) for row in accepted_attempts),
        default=float("inf"),
    )
    final_parameter = float(getattr(result, "load_factor", 0.0))
    event_counts = _event_counts(event_rows)
    criteria = {
        "solver_converged": bool(getattr(result, "converged", False)),
        "final_motion_reached": bool(
            np.isclose(final_parameter, 1.0, rtol=0.0, atol=1.0e-12)
        ),
        "accepted_states_present": bool(accepted_rows),
        "accepted_inner_solves_converged": all(
            bool(row["inner_converged"]) for row in accepted_rows
        ),
        "scale_aware_residuals_satisfied": max_residual <= 1.0e-8,
        "scale_aware_penetration_satisfied": max_penetration <= 1.0e-7,
        "restart_history_healthy": diagnostics.healthy,
        "repeated_pair_entries": event_counts["pair_entries"] >= 2,
        "repeated_pair_exits": event_counts["pair_exits"] >= 2,
    }
    return {
        "schema_version": SCHEMA,
        "profile": profile.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "termination_reason": str(getattr(result, "termination_reason", "unknown")),
        "final_parameter": final_parameter,
        "accepted_step_count": len(accepted_rows),
        "attempt_count": len(attempt_rows),
        "cutback_count": int(getattr(result, "cutback_count", 0)),
        "penalty_update_count": int(getattr(result, "penalty_update_count", 0)),
        "maximum_normalized_equilibrium_residual": max_residual,
        "maximum_normalized_penetration": max_penetration,
        "events": event_counts,
        "restart_diagnostics": diagnostics.summary(),
    }


def run(
    profile: str | RotatingBlocksExecutionProfile = "quick",
    *,
    raise_on_failure: bool = True,
    _solver: Solver = solve_event_aware_adaptive_contact_path,
) -> RotatingBlocksSolverRun:
    """Solve one rotating-blocks profile through the production solver stack."""

    selected = rotating_blocks_execution_profile(profile)
    model = build_rotating_blocks_model(selected.model_profile)
    result = _solver(
        model.problem,
        model.path.end_parameter,
        path=model.path,
        options=solver_options(selected),
    )
    accepted_rows = _accepted_rows(result)
    attempt_rows = _attempt_rows(result)
    event_rows = tuple(getattr(result, "event_rows")())
    summary = _summary(selected, result, accepted_rows, attempt_rows, event_rows)
    completed = RotatingBlocksSolverRun(
        selected,
        result,
        summary,
        accepted_rows,
        attempt_rows,
        event_rows,
    )
    if raise_on_failure and not completed.passed:
        criteria = summary["criteria"]
        assert isinstance(criteria, dict)
        failed = [name for name, passed in criteria.items() if not passed]
        raise RuntimeError("rotating-blocks solver criteria failed: " + ", ".join(failed))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()
    completed = run(arguments.profile)
    encoded = json.dumps(completed.summary, indent=2, sort_keys=True, allow_nan=False)
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
