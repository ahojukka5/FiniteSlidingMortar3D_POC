#!/usr/bin/env python3
"""Write the complete rotating-blocks result bundle and visualizations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
from rotating_blocks_bundle_data import (
    checkpoint_rows,
    interface_rows,
    pair_rows,
    select_checkpoints,
)
from rotating_blocks_bundle_output import write_checkpoint, write_plots
from rotating_blocks_model import RotatingBlocksModel, build_rotating_blocks_model
from rotating_blocks_refinement import RotatingBlocksRefinement
from rotating_blocks_refinement import run as run_refinement
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
    checkpoints: tuple[object, ...],
) -> tuple[list[str], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    selected_rows = checkpoint_rows(checkpoints)
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
            "tables/checkpoints.csv",
            selected_rows,
            "contact3d-rotating-blocks-checkpoints/v1",
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
) -> dict[str, object]:
    """Write and manifest-validate one complete result directory."""

    output = Path(output)
    writer = BenchmarkArtifactWriter(
        output,
        "rotating-blocks",
        seed=0,
        solver_settings={
            "profile": completed.profile,
            "solver": solver_options(completed.profile),
            "refinement_steps": refinement.summary["requested_steps"],
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    checkpoints = select_checkpoints(model, completed)
    required, contact_rows, pairs = _write_tables(
        writer,
        completed,
        refinement,
        checkpoints,
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
                "parameter": checkpoint.parameter,
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
    names = tuple(checkpoint.name for checkpoint in checkpoints)
    criteria = {
        "solver_passed": completed.passed,
        "refinement_passed": refinement.passed,
        "checkpoint_regimes_complete": names
        == ("pre-contact", "compressed", "mid-rotation", "final"),
        "final_checkpoint_reached": bool(
            np.isclose(checkpoints[-1].parameter, model.path.end_parameter)
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
        "solver_summary": completed.summary,
        "refinement_summary": refinement.summary,
        "checkpoints": checkpoint_files,
        "table_row_counts": {
            "accepted_steps": len(completed.accepted_rows),
            "attempts": len(completed.attempt_rows),
            "solver_diagnostics": len(completed.diagnostic_rows),
            "events": len(completed.event_rows),
            "interface_rows": len(contact_rows),
            "facet_pairs": len(pairs),
            "overlap_regions": len(overlap_rows),
            "refinement_fields": len(refinement.comparison_rows),
            "refinement_events": len(refinement.event_rows),
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
    """Execute production, refinement, and complete artifact export."""

    completed = _solver_runner(profile)
    refinement = _refinement_runner(profile)
    model = build_rotating_blocks_model(completed.profile.model_profile)
    return write_bundle(output, model, completed, refinement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("results/rotating-blocks"))
    arguments = parser.parse_args()
    summary = run(arguments.output, arguments.profile)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
