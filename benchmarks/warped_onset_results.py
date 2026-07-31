"""Result artifacts and summary metrics for warped nonmatching contact onset."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import build_facet_overlap
from svg_plots import write_line_chart

try:
    from .warped_onset_analysis import OnsetHistories
    from .warped_onset_model import BenchmarkModel, _minimum_reference_determinant
    from .warped_onset_reporting import (
        _write_csv,
        _write_deformation,
        _write_overlap,
        _write_pressure,
    )
except ImportError:  # Direct script execution from the repository root.
    from warped_onset_analysis import OnsetHistories
    from warped_onset_model import BenchmarkModel, _minimum_reference_determinant
    from warped_onset_reporting import (
        _write_csv,
        _write_deformation,
        _write_overlap,
        _write_pressure,
    )


def _write_tables(output: Path, histories: OnsetHistories) -> None:
    _write_csv(output / "accepted-steps.csv", histories.step_rows)
    _write_csv(output / "interface-rows.csv", histories.interface_rows)
    _write_csv(output / "attempt-history.csv", histories.attempt_rows)
    _write_csv(output / "tangent-checks.csv", histories.tangent_rows)


def _write_final_state_plots(
    output: Path,
    benchmark: BenchmarkModel,
    result: object,
    histories: OnsetHistories,
) -> None:
    final_step = result.accepted_steps[-1]
    final_contact = final_step.result.equilibrium.evaluation.contacts[0]
    final_displacement = final_step.result.displacement.reshape((-1, 3))
    current = benchmark.problem.mesh.reference_nodes + final_displacement
    _write_deformation(
        output / "deformation.svg",
        benchmark.problem.mesh.reference_nodes,
        current,
        benchmark.problem.mesh.elements,
    )
    final_pressure_rows = [
        row
        for row in histories.interface_rows
        if row["accepted_step"] == len(result.accepted_steps)
    ]
    _write_pressure(output / "pressure.svg", final_pressure_rows)

    interface = final_step.path_state.problem.interfaces[0]
    slave_current = interface.pair.slave.current_nodes(final_displacement[interface.slave_nodes])
    master_current = interface.pair.master.current_nodes(final_displacement[interface.master_nodes])
    polygons: list[tuple[np.ndarray, str]] = []
    for pair_index, (slave_index, master_index) in enumerate(
        final_contact.signature.facet_pairs
    ):
        slave_facet = interface.pair.slave.facets[slave_index]
        master_facet = interface.pair.master.facets[master_index]
        overlap = build_facet_overlap(
            slave_current[slave_facet],
            master_current[master_facet],
        )
        if pair_index == 0:
            polygons.append((overlap.slave_polygon, "slave QUAD4"))
        polygons.append((overlap.master_polygon, f"master TRI3 {master_index}"))
        polygons.append((overlap.intersection_polygon, f"intersection {master_index}"))
    _write_overlap(output / "overlap.svg", polygons)


def _write_history_plots(output: Path, step_rows: list[dict[str, object]]) -> None:
    parameters = np.asarray([float(row["parameter"]) for row in step_rows])
    write_line_chart(
        output / "reaction.svg",
        title="Warped nonmatching contact reactions",
        x_label="continuation parameter",
        y_label="summed constrained reaction",
        x_values=parameters,
        series=(
            (
                np.asarray([float(row["tool_reaction_z"]) for row in step_rows]),
                "tool reaction z",
            ),
            (
                np.asarray([float(row["support_reaction_z"]) for row in step_rows]),
                "support reaction z",
            ),
        ),
    )
    write_line_chart(
        output / "residual.svg",
        title="Scale-aware equilibrium and penetration histories",
        x_label="continuation parameter",
        y_label="normalized residual",
        x_values=parameters,
        series=(
            (
                np.asarray([float(row["normalized_residual"]) for row in step_rows]),
                "equilibrium",
            ),
            (
                np.asarray([float(row["normalized_penetration"]) for row in step_rows]),
                "penetration",
            ),
        ),
    )


def _metrics(result: object, histories: OnsetHistories) -> dict[str, object]:
    final_step = result.accepted_steps[-1]
    final_evaluation = final_step.result.equilibrium.evaluation
    final_contact = final_evaluation.contacts[0]
    step_rows = histories.step_rows
    maximum_tangent_error = max(
        float(row["relative_error"]) for row in histories.tangent_rows
    )
    return {
        "converged": result.converged,
        "accepted_steps": result.accepted_step_count,
        "attempts": len(result.attempts),
        "cutbacks": result.cutback_count,
        "penalty_updates": result.penalty_update_count,
        "contact_onset_parameter": float(histories.contacting_steps[0].parameter),
        "final_facet_pairs": len(final_contact.signature.facet_pairs),
        "final_overlap_area": final_contact.raw.contact.weights.total_area,
        "final_active_rows": int(np.count_nonzero(final_contact.signature.active_rows)),
        "final_supported_rows": int(np.count_nonzero(final_contact.signature.supported_rows)),
        "final_maximum_pressure": float(np.max(final_contact.pressure, initial=0.0)),
        "final_tool_reaction_z": float(step_rows[-1]["tool_reaction_z"]),
        "maximum_global_reaction_balance": max(
            float(row["global_reaction_balance"]) for row in step_rows
        ),
        "final_contact_force_balance": final_contact.raw.contact.force_balance_norm,
        "final_maximum_penetration": final_contact.diagnostics.maximum_penetration,
        "final_normalized_penetration": (
            final_step.result.scales.interfaces[0]
            .normalize_kkt(final_contact.diagnostics)
            .maximum_penetration
        ),
        "final_free_residual": final_evaluation.free_residual_norm,
        "final_normalized_residual": (
            final_evaluation.free_residual_norm / final_step.result.scales.force
        ),
        "minimum_element_jacobian": min(
            float(row["minimum_jacobian"]) for row in step_rows
        ),
        "maximum_partition_error": max(
            float(row["partition_error"]) for row in step_rows
        ),
        "maximum_directional_tangent_error": maximum_tangent_error,
        "total_contact_event_restarts": sum(
            int(row["contact_event_restarts"]) for row in step_rows
        ),
    }


def write_results(
    output: Path,
    benchmark: BenchmarkModel,
    result: object,
    histories: OnsetHistories,
) -> dict[str, object]:
    """Write all deterministic benchmark tables, plots, and summary metadata."""

    _write_tables(output, histories)
    _write_final_state_plots(output, benchmark, result, histories)
    _write_history_plots(output, histories.step_rows)
    summary = {
        "geometry": {
            "slave": "one warped QUAD4",
            "master": "two warped TRI3 facets",
            "initial_separation": benchmark.initial_separation,
            "reference_minimum_tet_determinant": _minimum_reference_determinant(
                benchmark.problem.mesh.reference_nodes,
                benchmark.problem.mesh.elements,
            ),
        },
        "path": {"tool_x": 0.04, "tool_z": -0.09, "dead_load_x": 0.50},
        "metrics": _metrics(result, histories),
        "tangent_checks": histories.tangent_rows,
        "accepted_steps": histories.step_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    for svg in output.glob("*.svg"):
        ElementTree.parse(svg)
    return summary
