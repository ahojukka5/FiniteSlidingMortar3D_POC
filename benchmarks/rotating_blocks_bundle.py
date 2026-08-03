#!/usr/bin/env python3
"""Write the complete rotating-blocks result bundle and visualizations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
from rotating_blocks_balance import (
    RotatingBlocksBalance,
    audit_accepted_states,
    write_balance_plots,
)
from rotating_blocks_bundle_data import (
    CHECKPOINT_NAMES,
    CheckpointSelection,
    checkpoint_selection_rows,
    interface_rows,
    pair_rows,
    select_checkpoint_regimes,
)
from rotating_blocks_bundle_output import write_checkpoint, write_plots
from rotating_blocks_gate import (
    acceptance_failure_message,
    acceptance_thresholds,
    evaluate_acceptance_gate,
    write_gate_artifacts,
)
from rotating_blocks_mesh_quality import (
    RotatingBlocksMeshQuality,
    evaluate_mesh_quality,
    mesh_quality_thresholds,
    write_mesh_quality_artifacts,
)
from rotating_blocks_mesh_quality_gate import include_mesh_quality_in_gate
from rotating_blocks_model import RotatingBlocksModel, build_rotating_blocks_model
from rotating_blocks_pressure import PressureArtifacts, write_pressure_artifacts
from rotating_blocks_refinement import RotatingBlocksRefinement
from rotating_blocks_refinement import run as run_refinement
from rotating_blocks_retention import (
    RotatingBlocksRetention,
    audit_contact_retention,
    retention_thresholds,
    write_retention_artifacts,
)
from rotating_blocks_retention_gate import include_retention_in_gate
from rotating_blocks_solver import RotatingBlocksSolverRun, solver_options
from rotating_blocks_solver import run as run_solver

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter

SCHEMA = "contact3d-rotating-blocks-bundle/v1"
SolverRunner = Callable[[object], RotatingBlocksSolverRun]
RefinementRunner = Callable[[object], RotatingBlocksRefinement]


def _csv(
    writer: BenchmarkArtifactWriter,
    path: str,
    rows: Iterable[Mapping[str, object]],
    schema: str,
) -> str:
    writer.write_csv(path, rows, schema=schema)
    return path


def _write_tables(
    writer: BenchmarkArtifactWriter,
    completed: RotatingBlocksSolverRun,
    refinement: RotatingBlocksRefinement,
    balance: RotatingBlocksBalance,
    selection: CheckpointSelection,
) -> tuple[list[str], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    checkpoints = selection.checkpoints
    selected_rows = checkpoint_selection_rows(selection)
    contact_rows = interface_rows(checkpoints)
    pairs = pair_rows(checkpoints)
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
        _csv(
            writer,
            "tables/checkpoints.csv",
            selected_rows,
            "contact3d-rotating-blocks-checkpoints/v2",
        ),
        _csv(
            writer,
            "tables/interface-rows.csv",
            contact_rows,
            "contact3d-rotating-blocks-interface-rows/v1",
        ),
        _csv(
            writer,
            "tables/facet-pairs.csv",
            pairs,
            "contact3d-rotating-blocks-facet-pairs/v1",
        ),
        _csv(
            writer,
            "tables/refinement-fields.csv",
            refinement.comparison_rows,
            "contact3d-rotating-blocks-refinement-fields/v1",
        ),
        _csv(
            writer,
            "tables/refinement-events.csv",
            refinement.event_rows,
            "contact3d-rotating-blocks-refinement-events/v1",
        ),
    ]
    return required, contact_rows, pairs


def write_bundle(
    output: Path,
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
    refinement: RotatingBlocksRefinement,
    *,
    balance: RotatingBlocksBalance | None = None,
    retention: RotatingBlocksRetention | None = None,
    mesh_quality: RotatingBlocksMeshQuality | None = None,
    determinism: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write and manifest-validate one complete result directory."""

    output = Path(output)
    assessed_balance = (
        audit_accepted_states(model, completed) if balance is None else balance
    )
    assessed_retention = (
        audit_contact_retention(completed) if retention is None else retention
    )
    assessed_quality = (
        evaluate_mesh_quality(model, completed, refinement)
        if mesh_quality is None
        else mesh_quality
    )
    writer = BenchmarkArtifactWriter(
        output,
        "rotating-blocks",
        seed=0,
        solver_settings={
            "profile": completed.profile,
            "solver": solver_options(completed.profile),
            "refinement_steps": refinement.summary["requested_steps"],
            "balance_force_tolerance": assessed_balance.summary["force_tolerance"],
            "balance_moment_tolerance": assessed_balance.summary["moment_tolerance"],
            "retention_thresholds": retention_thresholds(completed.profile),
            "mesh_quality_thresholds": mesh_quality_thresholds(completed.profile),
            "acceptance_thresholds": acceptance_thresholds(completed.profile),
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    selection = select_checkpoint_regimes(model, completed)
    checkpoints = selection.checkpoints
    required, contact_rows, pairs = _write_tables(
        writer,
        completed,
        refinement,
        assessed_balance,
        selection,
    )
    overlap_rows: list[dict[str, object]] = []
    checkpoint_files: list[dict[str, object]] = []
    final_overlays: tuple[tuple[np.ndarray, str], ...] = ()
    for ordinal, checkpoint in enumerate(checkpoints):
        paths, rows, overlays = write_checkpoint(writer, model, checkpoint, ordinal)
        required.extend(paths)
        overlap_rows.extend(rows)
        checkpoint_files.append(
            {
                "checkpoint": checkpoint.name,
                "selection_rule": checkpoint.selection_rule,
                "target_parameter": checkpoint.target,
                "accepted_step": checkpoint.accepted_step,
                "selected_parameter": checkpoint.parameter,
                "selection_error": checkpoint.selection_error,
                "files": list(paths),
            }
        )
        if checkpoint.name == "final":
            final_overlays = overlays
    required.append(
        _csv(
            writer,
            "tables/overlap-regions.csv",
            overlap_rows,
            "contact3d-rotating-blocks-overlap-regions/v1",
        )
    )
    required.extend(
        write_plots(writer, output, completed, checkpoints, final_overlays)
    )
    required.extend(write_balance_plots(writer, output, assessed_balance))
    pressure: PressureArtifacts = write_pressure_artifacts(
        writer,
        output,
        model,
        completed,
        refinement,
    )
    required.extend(pressure.required)
    required.extend(
        write_retention_artifacts(writer, output, assessed_retention)
    )
    quality_paths = write_mesh_quality_artifacts(
        writer,
        output,
        assessed_quality,
    )
    for path in quality_paths:
        if path.endswith(".svg"):
            writer.register(path, "svg")
    required.extend(quality_paths)
    gate = evaluate_acceptance_gate(
        completed,
        refinement,
        assessed_balance.summary,
        pressure.summary,
        determinism=determinism,
    )
    gate = include_retention_in_gate(gate, assessed_retention)
    gate = include_mesh_quality_in_gate(gate, assessed_quality)
    required.extend(write_gate_artifacts(writer, gate))
    names = tuple(checkpoint.name for checkpoint in checkpoints)
    final_checkpoint = next(
        (checkpoint for checkpoint in checkpoints if checkpoint.name == "final"),
        None,
    )
    rotation_count = sum(
        int(row.get("phase_index", -1)) == 1 for row in completed.accepted_rows
    )
    criteria = {
        "acceptance_gate_passed": gate.passed,
        "solver_passed": completed.passed,
        "refinement_passed": refinement.passed,
        "force_moment_balance_passed": assessed_balance.passed,
        "balance_evidence_complete": len(assessed_balance.rows)
        == len(completed.accepted_rows),
        "pressure_redistribution_passed": bool(pressure.summary["passed"]),
        "pressure_evidence_complete": pressure.row_counts["pressure_aggregates"]
        == len(completed.accepted_rows),
        "contact_retention_passed": assessed_retention.passed,
        "contact_retention_evidence_complete": len(assessed_retention.rows)
        == rotation_count,
        "mesh_quality_passed": assessed_quality.passed,
        "mesh_quality_evidence_complete": len(assessed_quality.rows)
        == len(completed.accepted_rows),
        "checkpoint_regimes_complete": selection.complete
        and names == CHECKPOINT_NAMES,
        "final_checkpoint_reached": final_checkpoint is not None
        and bool(
            np.isclose(
                final_checkpoint.parameter,
                model.path.end_parameter,
                rtol=0.0,
                atol=max(float(completed.profile.minimum_step), 1.0e-12),
            )
        ),
        "interface_fields_complete": all(
            {
                "normal_gap",
                "pressure",
                "multiplier",
                "supported",
                "active",
                "contact_force_x",
            }
            <= set(row)
            for row in contact_rows
        ),
        "projected_overlap_regions_present": bool(overlap_rows),
    }
    summary = {
        "schema_version": SCHEMA,
        "profile": completed.profile.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "acceptance_gate": gate.summary,
        "solver_summary": completed.summary,
        "refinement_summary": refinement.summary,
        "balance_summary": assessed_balance.summary,
        "pressure_summary": pressure.summary,
        "retention_summary": assessed_retention.summary,
        "mesh_quality_summary": assessed_quality.summary,
        "checkpoint_selection": checkpoint_selection_rows(selection),
        "checkpoints": checkpoint_files,
        "table_row_counts": {
            "accepted_steps": len(completed.accepted_rows),
            "attempts": len(completed.attempt_rows),
            "solver_diagnostics": len(completed.diagnostic_rows),
            "events": len(completed.event_rows),
            "acceptance_criteria": len(gate.rows),
            "force_moment_balance": len(assessed_balance.rows),
            "contact_retention": len(assessed_retention.rows),
            "mesh_quality": len(assessed_quality.rows),
            "refinement_mesh_quality": len(assessed_quality.refinement.rows),
            "checkpoint_requests": len(selection.requests),
            "checkpoint_exports": len(checkpoints),
            "interface_rows": len(contact_rows),
            "facet_pairs": len(pairs),
            "overlap_regions": len(overlap_rows),
            "refinement_fields": len(refinement.comparison_rows),
            "refinement_events": len(refinement.event_rows),
            **pressure.row_counts,
        },
    }
    writer.write_json("summary.json", summary, schema=SCHEMA)
    required.append("summary.json")
    writer.finalize(required=required)
    return summary


def run(
    output: Path,
    profile: str = "full",
    *,
    _solver_runner: SolverRunner = run_solver,
    _refinement_runner: RefinementRunner = run_refinement,
) -> dict[str, object]:
    """Execute the complete benchmark and enforce every acceptance criterion."""

    completed = _solver_runner(profile)
    refinement = _refinement_runner(profile)
    model = build_rotating_blocks_model(completed.profile.model_profile)
    summary = write_bundle(output, model, completed, refinement)
    if not summary["passed"]:
        gate_summary = summary["acceptance_gate"]
        assert isinstance(gate_summary, Mapping)
        if not gate_summary["passed"]:
            raise RuntimeError(acceptance_failure_message(gate_summary))
        failed = [name for name, passed in summary["criteria"].items() if not passed]
        raise RuntimeError("rotating-blocks bundle failed: " + ", ".join(failed))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("results/rotating-blocks"))
    arguments = parser.parse_args()
    summary = run(arguments.output, arguments.profile)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
