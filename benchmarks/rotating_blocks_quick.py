#!/usr/bin/env python3
"""Run the bounded rotating-blocks production profile and write its evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from rotating_blocks_balance import audit_accepted_states, write_balance_plots
from rotating_blocks_bundle_data import select_checkpoint_regimes
from rotating_blocks_bundle_output import write_checkpoint, write_plots
from rotating_blocks_model import build_rotating_blocks_model
from rotating_blocks_profiles import rotating_blocks_execution_profile
from rotating_blocks_retention import (
    audit_contact_retention,
    retention_thresholds,
    write_retention_artifacts,
)
from rotating_blocks_solver import RotatingBlocksSolverRun, solver_options
from rotating_blocks_solver import run as run_solver

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter

SCHEMA = "contact3d-rotating-blocks-quick/v1"
GATE_SCHEMA = "contact3d-rotating-blocks-quick-gate/v1"


def _csv(
    writer: BenchmarkArtifactWriter,
    path: str,
    rows: Iterable[Mapping[str, object]],
    schema: str,
) -> str:
    writer.write_csv(path, rows, schema=schema)
    return path


def _quick_gate(
    completed: RotatingBlocksSolverRun,
    *,
    balance_passed: bool,
    retention_passed: bool,
    checkpoint_regimes_complete: bool,
    final_checkpoint_present: bool,
) -> dict[str, object]:
    criteria = {
        "solver_passed": completed.passed,
        "balance_passed": balance_passed,
        "contact_retention_passed": retention_passed,
        "checkpoint_regimes_identified": checkpoint_regimes_complete,
        "final_checkpoint_present": final_checkpoint_present,
    }
    return {
        "schema_version": GATE_SCHEMA,
        "profile": completed.profile.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
    }


def _augmentation_rows(result: object) -> list[dict[str, object]]:
    rows = []
    for row in tuple(getattr(result, "history", ())):
        rows.append(
            {
                "augmentation": int(row.augmentation),
                "newton_iterations": int(row.newton_iterations),
                "contact_event_restarts": int(row.contact_event_restarts),
                "normalized_equilibrium_residual": float(
                    row.normalized_equilibrium_residual
                ),
                "normalized_maximum_penetration": float(
                    row.normalized_maximum_penetration
                ),
                "normalized_maximum_complementarity": float(
                    row.normalized_maximum_complementarity
                ),
                "normalized_maximum_projection_residual": float(
                    row.normalized_maximum_projection_residual
                ),
                "normalized_maximum_multiplier_increment": float(
                    row.normalized_maximum_multiplier_increment
                ),
                "active_rows": int(row.active_rows),
                "normalized_maximum_pressure": float(
                    row.normalized_maximum_pressure
                ),
            }
        )
    return rows


def _report_solver_failure(completed: RotatingBlocksSolverRun) -> None:
    criteria = completed.summary["criteria"]
    assert isinstance(criteria, dict)
    failed = [name for name, passed in criteria.items() if not passed]
    attempt_results = tuple(getattr(completed.result, "attempt_results", ()))
    payload = {
        "summary": completed.summary,
        "final_attempts": list(completed.attempt_rows[-12:]),
        "final_augmentation_histories": [
            _augmentation_rows(result) for result in attempt_results[-2:]
        ],
    }
    print(
        "ROTATING_BLOCKS_QUICK_FAILURE\n"
        + json.dumps(payload, indent=2, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    raise RuntimeError("rotating-blocks solver criteria failed: " + ", ".join(failed))


def write_quick_bundle(
    output: Path,
    completed: RotatingBlocksSolverRun,
) -> dict[str, object]:
    """Write bounded evidence without refinement or repetition solves."""

    profile = rotating_blocks_execution_profile(completed.profile)
    if profile.name != "quick":
        raise ValueError("the bounded rotating-blocks writer requires the quick profile")
    model = build_rotating_blocks_model(profile.model_profile)
    balance = audit_accepted_states(model, completed)
    retention = audit_contact_retention(completed)
    selection = select_checkpoint_regimes(model, completed)
    final = next(
        (checkpoint for checkpoint in selection.checkpoints if checkpoint.name == "final"),
        None,
    )

    writer = BenchmarkArtifactWriter(
        output,
        "rotating-blocks",
        seed=0,
        solver_settings={
            "profile": profile,
            "solver": solver_options(profile),
            "retention_thresholds": retention_thresholds(profile),
            "optional_refinement_executed": False,
            "optional_repetition_executed": False,
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    required = [
        _csv(
            writer,
            "tables/accepted-steps.csv",
            completed.accepted_rows,
            "contact3d-rotating-blocks-accepted-steps/v1",
        ),
        _csv(
            writer,
            "tables/attempts.csv",
            completed.attempt_rows,
            "contact3d-rotating-blocks-attempts/v1",
        ),
        _csv(
            writer,
            "tables/solver-diagnostics.csv",
            completed.diagnostic_rows,
            "contact3d-rotating-blocks-solver-diagnostics/v1",
        ),
        _csv(
            writer,
            "tables/events.csv",
            completed.event_rows,
            "contact3d-rotating-blocks-events/v1",
        ),
        _csv(
            writer,
            "tables/force-moment-balance.csv",
            balance.rows,
            "contact3d-rotating-blocks-balance/v1",
        ),
    ]
    required.extend(write_balance_plots(writer, output, balance))
    required.extend(write_retention_artifacts(writer, output, retention))

    overlap_rows: tuple[dict[str, object], ...] = ()
    if final is not None:
        checkpoint_paths, overlap_rows, overlays = write_checkpoint(
            writer,
            model,
            final,
            0,
        )
        required.extend(checkpoint_paths)
        required.extend(write_plots(writer, output, completed, (final,), overlays))
        required.append(
            _csv(
                writer,
                "tables/final-overlap-regions.csv",
                overlap_rows,
                "contact3d-rotating-blocks-overlap-regions/v1",
            )
        )

    gate = _quick_gate(
        completed,
        balance_passed=balance.passed,
        retention_passed=retention.passed,
        checkpoint_regimes_complete=selection.complete,
        final_checkpoint_present=final is not None,
    )
    writer.write_json("acceptance-gate.json", gate, schema=GATE_SCHEMA)
    required.append("acceptance-gate.json")
    summary = {
        "schema_version": SCHEMA,
        "profile": profile.name,
        "passed": bool(gate["passed"]),
        "acceptance_gate": gate,
        "solver_summary": completed.summary,
        "balance_summary": balance.summary,
        "retention_summary": retention.summary,
        "optional_evidence": {
            "refinement_executed": False,
            "repetition_executed": False,
        },
        "checkpoint_selection": list(selection.requests),
        "table_row_counts": {
            "accepted_steps": len(completed.accepted_rows),
            "attempts": len(completed.attempt_rows),
            "solver_diagnostics": len(completed.diagnostic_rows),
            "events": len(completed.event_rows),
            "force_moment_balance": len(balance.rows),
            "contact_retention": len(retention.rows),
            "checkpoint_requests": len(selection.requests),
            "checkpoint_exports": int(final is not None),
            "overlap_regions": len(overlap_rows),
        },
    }
    writer.write_json("summary.json", summary, schema=SCHEMA)
    required.append("summary.json")
    writer.finalize(required=required)
    if not summary["passed"]:
        failed = gate["failed_criteria"]
        assert isinstance(failed, list)
        raise RuntimeError("rotating-blocks quick gate failed: " + ", ".join(failed))
    return summary


def run(output: Path) -> dict[str, object]:
    """Execute one quick production solve and write bounded evidence."""

    completed = run_solver("quick", raise_on_failure=False)
    if not completed.passed:
        _report_solver_failure(completed)
    return write_quick_bundle(output, completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
