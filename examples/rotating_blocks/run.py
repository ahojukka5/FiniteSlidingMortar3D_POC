"""Solve rotating blocks and write compact user-facing results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from contact3d.benchmark_artifacts import write_tet4_vtu
from contact3d.benchmark_plots import (
    write_line_chart,
    write_mesh_projection_overlay,
)
from contact3d.solvers.events import solve_event_aware_adaptive_contact_path

from .model import RotatingBlocksModel, build_model, solver_options

SUMMARY_SCHEMA = "contact3d-rotating-blocks-example/v1"
EQUILIBRIUM_TOLERANCE = 1.0e-8
PENETRATION_TOLERANCE = 1.0e-7
FORCE_BALANCE_TOLERANCE = 1.0e-7


def _event_parameter(row: Mapping[str, object]) -> float:
    value = row.get("continuation_parameter", row.get("parameter", 0.0))
    return float(value)


def _history(
    model: RotatingBlocksModel,
    result: object,
    event_rows: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tiny = np.finfo(float).tiny
    for accepted_step, step in enumerate(result.accepted_steps, start=1):
        solved = step.result
        evaluation = solved.equilibrium.evaluation
        contact = evaluation.contacts[0]
        normalized = solved.scales.interfaces[0].normalize_kkt(contact.diagnostics)
        reaction = np.asarray(step.reaction, dtype=float).reshape((-1, 3))
        applied = np.asarray(
            step.path_state.effective_force,
            dtype=float,
        ).reshape((-1, 3))
        controlled_reaction = np.sum(reaction[model.controlled_nodes], axis=0)
        external = reaction + applied
        contact_force = np.asarray(contact.residual, dtype=float).reshape((-1, 3))
        force_scale = max(
            float(np.sum(np.linalg.norm(reaction, axis=1))),
            float(np.sum(np.linalg.norm(applied, axis=1))),
            float(np.sum(np.linalg.norm(contact_force, axis=1))),
            tiny,
        )
        parameter = float(step.parameter)
        rows.append(
            {
                "accepted_step": accepted_step,
                "parameter": parameter,
                "phase_index": int(
                    round(float(step.path_state.value("phase_index")))
                ),
                "phase_parameter": float(
                    step.path_state.value("phase_parameter")
                ),
                "rotation_angle": float(
                    step.path_state.value("rotation_angle")
                ),
                "reaction_x": float(controlled_reaction[0]),
                "reaction_y": float(controlled_reaction[1]),
                "reaction_z": float(controlled_reaction[2]),
                "reaction_norm": float(np.linalg.norm(controlled_reaction)),
                "normalized_equilibrium_residual": float(
                    evaluation.free_residual_norm / solved.scales.force
                ),
                "normalized_penetration": float(
                    normalized.maximum_penetration
                ),
                "force_balance_relative": float(
                    np.linalg.norm(np.sum(external, axis=0)) / force_scale
                ),
                "active_rows": int(
                    np.count_nonzero(contact.signature.active_rows)
                ),
                "supported_rows": int(
                    np.count_nonzero(contact.signature.supported_rows)
                ),
                "facet_pairs": int(len(contact.signature.facet_pairs)),
                "facet_pair_signature": [
                    [int(slave), int(master)]
                    for slave, master in contact.signature.facet_pairs
                ],
                "overlap_area": float(contact.raw.contact.weights.total_area),
                "maximum_pressure": float(
                    np.max(contact.pressure, initial=0.0)
                ),
                "event_count": sum(
                    _event_parameter(event) <= parameter + 1.0e-12
                    for event in event_rows
                ),
                "minimum_element_jacobian": float(
                    evaluation.bulk.minimum_jacobian
                ),
            }
        )
    return rows


def _summary(
    model: RotatingBlocksModel,
    result: object,
    history: list[dict[str, object]],
    event_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not history:
        raise RuntimeError("rotating-blocks solve produced no accepted states")
    final = history[-1]
    contacting = [row for row in history if int(row["active_rows"]) > 0]
    overlap_values = np.asarray(
        [float(row["overlap_area"]) for row in contacting],
        dtype=float,
    )
    facet_pair_values = np.asarray(
        [int(row["facet_pairs"]) for row in contacting],
        dtype=np.int64,
    )
    topology_signatures = {
        tuple(
            tuple(int(value) for value in pair)
            for pair in row["facet_pair_signature"]
        )
        for row in contacting
    }
    maximum_residual = max(
        float(row["normalized_equilibrium_residual"]) for row in history
    )
    maximum_penetration = max(
        float(row["normalized_penetration"]) for row in history
    )
    maximum_force_balance = max(
        float(row["force_balance_relative"]) for row in history
    )
    minimum_jacobian = min(
        float(row["minimum_element_jacobian"]) for row in history
    )
    changing_overlap_topology = len(topology_signatures) > 1
    checks = {
        "solver_converged": bool(result.converged),
        "reached_final_parameter": (
            abs(float(final["parameter"]) - 1.0) <= 1.0e-12
        ),
        "contact_established": bool(contacting),
        "final_contact_supported": int(final["supported_rows"]) > 0,
        "changing_overlap_topology": changing_overlap_topology,
        "topology_events": len(event_rows) >= 2,
        "equilibrium_residual": maximum_residual <= EQUILIBRIUM_TOLERANCE,
        "contact_penetration": maximum_penetration <= PENETRATION_TOLERANCE,
        "force_balance": maximum_force_balance <= FORCE_BALANCE_TOLERANCE,
        "positive_element_jacobian": minimum_jacobian > 0.0,
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "example": "rotating blocks with changing mortar overlap topology",
        "formulation": {
            "contact": "biased single-pass standard mortar",
            "enforcement": "projected augmented Lagrange",
            "bulk": "finite-strain neo-Hookean TET4",
            "surfaces": "nonmatching QUAD4/QUAD4",
            "path": "compression, translation, and 90-degree rotation",
        },
        "geometry": {
            "nodes": int(model.problem.mesh.node_count),
            "elements": int(model.problem.mesh.element_count),
            "slave_nodes": int(len(model.slave_nodes)),
            "master_nodes": int(len(model.master_nodes)),
            "initial_separation": float(model.geometry.initial_separation),
            "final_angle": float(model.geometry.final_angle),
        },
        "metrics": {
            "converged": bool(result.converged),
            "final_parameter": float(final["parameter"]),
            "accepted_steps": int(len(history)),
            "attempts": int(len(result.attempts)),
            "cutbacks": int(result.cutback_count),
            "penalty_updates": int(result.penalty_update_count),
            "event_count": int(len(event_rows)),
            "contact_onset_parameter": float(contacting[0]["parameter"]),
            "final_rotation_angle": float(final["rotation_angle"]),
            "final_reaction_x": float(final["reaction_x"]),
            "final_reaction_y": float(final["reaction_y"]),
            "final_reaction_z": float(final["reaction_z"]),
            "final_reaction_norm": float(final["reaction_norm"]),
            "final_active_rows": int(final["active_rows"]),
            "final_supported_rows": int(final["supported_rows"]),
            "final_facet_pairs": int(final["facet_pairs"]),
            "final_overlap_area": float(final["overlap_area"]),
            "overlap_area_range": float(np.ptp(overlap_values)),
            "facet_pair_count_range": int(np.ptp(facet_pair_values)),
            "unique_overlap_topologies": int(len(topology_signatures)),
            "final_maximum_pressure": float(final["maximum_pressure"]),
            "maximum_normalized_equilibrium_residual": maximum_residual,
            "maximum_normalized_penetration": maximum_penetration,
            "maximum_force_balance_relative": maximum_force_balance,
            "minimum_element_jacobian": minimum_jacobian,
        },
        "tolerances": {
            "normalized_equilibrium_residual": EQUILIBRIUM_TOLERANCE,
            "normalized_penetration": PENETRATION_TOLERANCE,
            "relative_force_balance": FORCE_BALANCE_TOLERANCE,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "history": history,
        "events": list(event_rows),
    }


def _write_vtk(
    path: Path,
    model: RotatingBlocksModel,
    step: object,
) -> None:
    problem = step.path_state.problem
    solved = step.result
    evaluation = solved.equilibrium.evaluation
    contact = evaluation.contacts[0]
    interface = problem.interfaces[0]
    node_count = problem.mesh.node_count

    pressure = np.zeros(node_count)
    gap = np.zeros(node_count)
    active = np.zeros(node_count, dtype=np.int64)
    supported = np.zeros(node_count, dtype=np.int64)
    pressure[interface.slave_nodes] = contact.pressure
    gap[interface.slave_nodes] = contact.normal_gaps
    active[interface.slave_nodes] = np.asarray(
        contact.signature.active_rows,
        dtype=np.int64,
    )
    supported[interface.slave_nodes] = np.asarray(
        contact.signature.supported_rows,
        dtype=np.int64,
    )
    contact_force = np.zeros(3 * node_count)
    np.add.at(contact_force, interface.dofs, contact.residual)
    body_id = np.concatenate(
        (
            np.zeros(len(model.lower_elements), dtype=np.int64),
            np.ones(len(model.upper_elements), dtype=np.int64),
        )
    )
    write_tet4_vtu(
        path,
        problem.mesh.reference_nodes,
        problem.mesh.elements,
        solved.displacement,
        point_data={
            "reaction": step.reaction.reshape((-1, 3)),
            "effective_load": step.path_state.effective_force.reshape((-1, 3)),
            "contact_force": contact_force.reshape((-1, 3)),
            "contact_pressure": pressure,
            "normal_gap": gap,
            "contact_active": active,
            "contact_supported": supported,
        },
        cell_data={"body_id": body_id},
    )


def _write_vtk_states(
    output: Path,
    model: RotatingBlocksModel,
    result: object,
) -> dict[str, float]:
    steps = tuple(result.accepted_steps)
    if len(steps) < 3:
        raise RuntimeError("rotating-blocks solve produced too few VTK states")
    compression = min(
        steps,
        key=lambda step: abs(
            float(step.parameter) - model.geometry.compression_end
        ),
    )
    mid_rotation = min(
        steps,
        key=lambda step: abs(
            float(step.path_state.value("rotation_angle"))
            - 0.5 * model.geometry.final_angle
        ),
    )
    final = steps[-1]
    selected = {
        "compression.vtu": compression,
        "mid-rotation.vtu": mid_rotation,
        "final.vtu": final,
    }
    if len({id(step) for step in selected.values()}) != len(selected):
        raise RuntimeError("rotating-blocks VTK selections are not distinct")
    for filename, step in selected.items():
        _write_vtk(output / filename, model, step)
    return {
        filename: float(step.parameter)
        for filename, step in selected.items()
    }


def _write_plots(
    output: Path,
    model: RotatingBlocksModel,
    result: object,
    history: list[dict[str, object]],
) -> None:
    parameters = np.asarray([float(row["parameter"]) for row in history])
    reaction = np.asarray([float(row["reaction_norm"]) for row in history])
    write_line_chart(
        output / "reaction-path.svg",
        title="Rotating blocks reaction path",
        x_label="path parameter",
        y_label="controlled-block reaction norm",
        x_values=parameters,
        series=((reaction, "reaction norm"),),
        show_markers=True,
    )
    final = result.accepted_steps[-1].result.displacement.reshape((-1, 3))
    nodes = model.problem.mesh.reference_nodes
    write_mesh_projection_overlay(
        output / "deformed.svg",
        title="Rotating blocks: reference and final x-y mesh",
        reference_nodes=nodes,
        current_nodes=nodes + final,
        elements=model.problem.mesh.elements,
        axes=(0, 1),
    )


def run(output: Path) -> dict[str, object]:
    """Run the production solver and write inspectable result files."""

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    model = build_model()
    result = solve_event_aware_adaptive_contact_path(
        model.problem,
        model.path.end_parameter,
        path=model.path,
        options=solver_options(),
    )
    if not result.converged:
        raise RuntimeError(
            f"rotating-blocks path failed: {result.termination_reason}"
        )
    event_rows = tuple(result.event_rows())
    history = _history(model, result, event_rows)
    summary = _summary(model, result, history, event_rows)
    vtk_states = _write_vtk_states(target, model, result)
    summary["artifacts"] = {
        "vtk_states": vtk_states,
        "files": (
            "compression.vtu",
            "mid-rotation.vtu",
            "final.vtu",
            "deformed.svg",
            "reaction-path.svg",
            "summary.json",
        ),
    }
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_plots(target, model, result, history)
    if not summary["passed"]:
        failed = [name for name, passed in summary["checks"].items() if not passed]
        raise RuntimeError("rotating-blocks checks failed: " + ", ".join(failed))
    return summary
