#!/usr/bin/env python3
"""Run rotating blocks through the production adaptive contact solver."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from rotating_blocks_diagnostics import (
    RotatingBlocksDiagnostics,
    collect_solver_diagnostics,
)
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
from contact3d.linear_solver import LinearSolverOptions

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
    diagnostic_rows: tuple[dict[str, object], ...]

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
    linear = LinearSolverOptions(
        backend="auto" if profile.name == "quick" else "sparse_lu",
        dense_threshold=96,
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
            linear_solver=linear,
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


def _contact_metrics(step: object) -> dict[str, object]:
    result = step.result
    equilibrium = getattr(result, "equilibrium", None)
    evaluation = getattr(equilibrium, "evaluation", None)
    contacts = tuple(getattr(evaluation, "contacts", ()))
    if not contacts:
        return {
            "maximum_pressure": 0.0,
            "overlap_area": 0.0,
            "active_rows": 0,
            "supported_rows": 0,
            "facet_pairs": 0,
        }
    contact = contacts[0]
    weights = contact.raw.contact.weights
    return {
        "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
        "overlap_area": float(weights.total_area),
        "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
        "supported_rows": int(np.count_nonzero(contact.signature.supported_rows)),
        "facet_pairs": len(contact.signature.facet_pairs),
    }


def _controlled_reaction(step: object, controlled_nodes: np.ndarray) -> np.ndarray:
    reaction = getattr(step, "reaction", None)
    if reaction is None:
        return np.zeros(3)
    nodal = np.asarray(reaction, dtype=float).reshape((-1, 3))
    return np.sum(nodal[np.asarray(controlled_nodes, dtype=np.int64)], axis=0)


def _accepted_rows(
    result: object,
    controlled_nodes: np.ndarray,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, step in enumerate(tuple(getattr(result, "accepted_steps", ())), start=1):
        state = step.path_state
        value = state.value
        reaction = _controlled_reaction(step, controlled_nodes)
        rows.append(
            {
                "accepted_step": index,
                "parameter": float(step.parameter),
                "phase_index": int(round(float(value("phase_index")))),
                "phase_parameter": float(value("phase_parameter")),
                "rotation_angle": float(value("rotation_angle")),
                "reaction_norm": float(step.reaction_norm),
                "reaction_x": float(reaction[0]),
                "reaction_y": float(reaction[1]),
                "reaction_z": float(reaction[2]),
                "inner_converged": bool(step.result.converged),
                "inner_termination_reason": str(step.result.termination_reason),
                **_contact_metrics(step),
            }
        )
    return tuple(rows)


def _attempt_rows(
    result: object,
    diagnostics: RotatingBlocksDiagnostics,
) -> tuple[dict[str, object], ...]:
    diagnostic_by_attempt = {
        int(row["attempt"]): row for row in diagnostics.rows
    }
    rows: list[dict[str, object]] = []
    for attempt in tuple(getattr(result, "attempts", ())):
        diagnostic = diagnostic_by_attempt[int(attempt.attempt)]
        rows.append(
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
                **{
                    key: value
                    for key, value in diagnostic.items()
                    if key
                    not in {
                        "attempt",
                        "action",
                        "start_parameter",
                        "target_parameter",
                        "inner_termination_reason",
                        "augmentation_iterations",
                        "newton_iterations",
                        "contact_event_restarts",
                    }
                },
            }
        )
    return tuple(rows)


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
    diagnostics: RotatingBlocksDiagnostics,
) -> dict[str, object]:
    restart_diagnostics = analyze_restart_diagnostics(result)
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
    solver_diagnostics = diagnostics.summary
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
        "restart_history_healthy": restart_diagnostics.healthy,
        "repeated_pair_entries": event_counts["pair_entries"] >= 2,
        "repeated_pair_exits": event_counts["pair_exits"] >= 2,
        "solver_diagnostics_complete": bool(
            solver_diagnostics["diagnostics_complete"]
        ),
        "sparse_backend_policy_satisfied": bool(
            solver_diagnostics["sparse_profile_avoided_dense_materialization"]
        ),
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
        "restart_diagnostics": restart_diagnostics.summary(),
        "solver_diagnostics": solver_diagnostics,
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
    started = perf_counter()
    result = _solver(
        model.problem,
        model.path.end_parameter,
        path=model.path,
        options=solver_options(selected),
    )
    wall_seconds = perf_counter() - started
    diagnostics = collect_solver_diagnostics(
        result,
        profile=selected.name,
        wall_seconds=wall_seconds,
    )
    accepted_rows = _accepted_rows(result, model.controlled_nodes)
    attempt_rows = _attempt_rows(result, diagnostics)
    event_rows = tuple(result.event_rows())
    summary = _summary(
        selected,
        result,
        accepted_rows,
        attempt_rows,
        event_rows,
        diagnostics,
    )
    completed = RotatingBlocksSolverRun(
        selected,
        result,
        summary,
        accepted_rows,
        attempt_rows,
        event_rows,
        diagnostics.rows,
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
