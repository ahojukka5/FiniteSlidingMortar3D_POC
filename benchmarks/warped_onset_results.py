"""Result artifacts and summary metrics for warped nonmatching contact onset."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import build_facet_overlap
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from svg_plots import write_line_chart

try:
    from .warped_onset_analysis import OnsetHistories
    from .warped_onset_model import BenchmarkModel, _minimum_reference_determinant
    from .warped_onset_reporting import (
        _write_deformation,
        _write_overlap,
        _write_pressure,
    )
except ImportError:  # Direct script execution from the repository root.
    from warped_onset_analysis import OnsetHistories
    from warped_onset_model import BenchmarkModel, _minimum_reference_determinant
    from warped_onset_reporting import (
        _write_deformation,
        _write_overlap,
        _write_pressure,
    )


def _write_tables(
    artifacts: BenchmarkArtifactWriter,
    histories: OnsetHistories,
) -> None:
    artifacts.write_csv(
        "accepted-steps.csv",
        histories.step_rows,
        schema="contact3d-warped-onset-steps/v1",
    )
    artifacts.write_csv(
        "interface-rows.csv",
        histories.interface_rows,
        schema="contact3d-contact-path-nodes/v1",
    )
    artifacts.write_csv(
        "attempt-history.csv",
        histories.attempt_rows,
        schema="contact3d-continuation-attempts/v1",
    )
    artifacts.write_csv(
        "tangent-checks.csv",
        histories.tangent_rows,
        schema="contact3d-directional-tangent-checks/v1",
    )


def _overlap_data(
    result: object,
) -> tuple[
    list[tuple[np.ndarray, str]],
    list[tuple[np.ndarray, int, int]],
]:
    final_step = result.accepted_steps[-1]
    final_contact = final_step.result.equilibrium.evaluation.contacts[0]
    final_displacement = final_step.result.displacement.reshape((-1, 3))
    interface = final_step.path_state.problem.interfaces[0]
    slave_current = interface.pair.slave.current_nodes(
        final_displacement[interface.slave_nodes]
    )
    master_current = interface.pair.master.current_nodes(
        final_displacement[interface.master_nodes]
    )
    polygons: list[tuple[np.ndarray, str]] = []
    vtk_polygons: list[tuple[np.ndarray, int, int]] = []
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
            vtk_polygons.append((overlap.slave_polygon, 0, pair_index))
        polygons.append((overlap.master_polygon, f"master TRI3 {master_index}"))
        polygons.append((overlap.intersection_polygon, f"intersection {master_index}"))
        vtk_polygons.append((overlap.master_polygon, 1, pair_index))
        vtk_polygons.append((overlap.intersection_polygon, 2, pair_index))
    return polygons, vtk_polygons


def _write_final_state_plots(
    output: Path,
    benchmark: BenchmarkModel,
    result: object,
    histories: OnsetHistories,
    polygons: list[tuple[np.ndarray, str]],
) -> None:
    final_step = result.accepted_steps[-1]
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


def _polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _write_vtk(
    artifacts: BenchmarkArtifactWriter,
    benchmark: BenchmarkModel,
    result: object,
    vtk_polygons: list[tuple[np.ndarray, int, int]],
) -> None:
    final_step = result.accepted_steps[-1]
    final_problem = final_step.path_state.problem
    final_result = final_step.result
    evaluation = final_result.equilibrium.evaluation
    contact = evaluation.contacts[0]
    interface = final_problem.interfaces[0]
    total_dofs = 3 * final_problem.mesh.node_count
    global_contact_force = np.zeros(total_dofs)
    np.add.at(global_contact_force, interface.dofs, contact.residual)
    element_count = final_problem.mesh.element_count
    body_id = np.concatenate(
        [
            np.zeros(element_count // 2, dtype=np.int64),
            np.ones(element_count - element_count // 2, dtype=np.int64),
        ]
    )
    artifacts.write_tet4_vtu(
        "deformed.vtu",
        final_problem.mesh.reference_nodes,
        final_problem.mesh.elements,
        final_result.displacement,
        point_data={
            "reaction": final_step.reaction.reshape((-1, 3)),
            "effective_load": final_step.path_state.effective_force.reshape((-1, 3)),
            "contact_force": global_contact_force.reshape((-1, 3)),
        },
        cell_data={
            "body_id": body_id,
            "jacobian": np.asarray(
                [item.jacobian for item in evaluation.bulk.element_evaluations]
            ),
            "energy_density": np.asarray(
                [item.energy_density for item in evaluation.bulk.element_evaluations]
            ),
        },
    )

    displacement = final_result.displacement.reshape((-1, 3))
    local_contact_force = contact.residual.reshape((-1, 3))
    slave_count = len(interface.slave_nodes)
    artifacts.write_surface_vtp(
        "slave-contact.vtp",
        interface.pair.slave.reference_nodes,
        interface.pair.slave.facets,
        displacement[interface.slave_nodes],
        point_data={
            "normal_gap": contact.normal_gaps,
            "pressure": contact.pressure,
            "multiplier": final_result.states[0].multipliers,
            "supported": np.asarray(contact.signature.supported_rows, dtype=np.int64),
            "active": np.asarray(contact.signature.active_rows, dtype=np.int64),
            "contact_force": local_contact_force[:slave_count],
        },
    )
    master_overlap = np.zeros(len(interface.pair.master.facets), dtype=float)
    for (_, master_index), area in zip(
        contact.signature.facet_pairs,
        contact.raw.contact.weights.overlap_areas,
        strict=True,
    ):
        master_overlap[master_index] += area
    artifacts.write_surface_vtp(
        "master-contact.vtp",
        interface.pair.master.reference_nodes,
        interface.pair.master.facets,
        displacement[interface.master_nodes],
        point_data={"contact_force": local_contact_force[slave_count:]},
        cell_data={"overlap_area": master_overlap},
    )

    projected_points: list[np.ndarray] = []
    projected_facets: list[np.ndarray] = []
    region_kind: list[int] = []
    pair_indices: list[int] = []
    projected_areas: list[float] = []
    for polygon, kind, pair_index in vtk_polygons:
        start = len(projected_points)
        projected_points.extend(np.column_stack([polygon, np.zeros(len(polygon))]))
        projected_facets.append(
            np.arange(start, start + len(polygon), dtype=np.int64)
        )
        region_kind.append(kind)
        pair_indices.append(pair_index)
        projected_areas.append(_polygon_area(polygon))
    artifacts.write_surface_vtp(
        "projected-overlap.vtp",
        np.asarray(projected_points),
        tuple(projected_facets),
        cell_data={
            "region_kind": np.asarray(region_kind, dtype=np.int64),
            "pair_index": np.asarray(pair_indices, dtype=np.int64),
            "projected_area": np.asarray(projected_areas),
        },
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
        "final_supported_rows": int(
            np.count_nonzero(final_contact.signature.supported_rows)
        ),
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
    artifacts: BenchmarkArtifactWriter,
    benchmark: BenchmarkModel,
    result: object,
    histories: OnsetHistories,
) -> dict[str, object]:
    """Write all deterministic benchmark tables, plots, and summary metadata."""

    output = artifacts.output
    _write_tables(artifacts, histories)
    polygons, vtk_polygons = _overlap_data(result)
    _write_final_state_plots(output, benchmark, result, histories, polygons)
    _write_history_plots(output, histories.step_rows)
    _write_vtk(artifacts, benchmark, result, vtk_polygons)
    summary = {
        "schema_version": "contact3d-warped-contact-onset/v1",
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
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-warped-contact-onset/v1",
    )
    for svg in output.glob("*.svg"):
        ElementTree.parse(svg)
        artifacts.register(svg.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "accepted-steps.csv",
            "interface-rows.csv",
            "attempt-history.csv",
            "tangent-checks.csv",
            "deformed.vtu",
            "slave-contact.vtp",
            "master-contact.vtp",
            "projected-overlap.vtp",
            "deformation.svg",
            "pressure.svg",
            "overlap.svg",
            "reaction.svg",
            "residual.svg",
        )
    )
    return summary
