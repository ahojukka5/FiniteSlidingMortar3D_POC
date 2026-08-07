"""Solve the contact patch and write its compact user-facing results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contact3d.benchmark_artifacts import write_tet4_vtu
from contact3d.benchmark_plots import write_line_chart
from contact3d.solvers import solve_adaptive_contact_path

from .model import ContactPatchModel, build_model, solver_options

SUMMARY_SCHEMA = "contact3d-contact-patch-example/v1"
EQUILIBRIUM_TOLERANCE = 1.0e-8
PENETRATION_TOLERANCE = 2.0e-7
PARTITION_TOLERANCE = 1.0e-10


def _step_contact(step: object) -> object:
    return step.result.equilibrium.evaluation.contacts[0]


def _summary(model: ContactPatchModel, result: object) -> dict[str, object]:
    steps = tuple(result.accepted_steps)
    if not steps:
        raise RuntimeError("contact-patch solve produced no accepted states")
    contacting = tuple(
        step for step in steps if any(_step_contact(step).signature.active_rows)
    )
    if not contacting:
        raise RuntimeError("contact-patch path never established active contact")

    final_step = steps[-1]
    final_result = final_step.result
    evaluation = final_result.equilibrium.evaluation
    contact = evaluation.contacts[0]
    normalized = final_result.scales.interfaces[0].normalize_kkt(contact.diagnostics)
    reaction = final_step.reaction.reshape((-1, 3))
    applied = final_step.path_state.effective_force.reshape((-1, 3)).sum(axis=0)
    reaction_balance = reaction.sum(axis=0) + applied
    tool_reaction = reaction[model.tool_nodes].sum(axis=0)
    support_reaction = reaction[model.support_nodes].sum(axis=0)
    maximum_partition_error = max(
        float(_step_contact(step).raw.contact.weights.consistency_error)
        for step in steps
    )
    minimum_jacobian = min(
        float(step.result.equilibrium.evaluation.bulk.minimum_jacobian)
        for step in steps
    )
    final_parameter = float(final_step.parameter)
    normalized_residual = float(evaluation.free_residual_norm / final_result.scales.force)
    normalized_penetration = float(normalized.maximum_penetration)
    checks = {
        "solver_converged": bool(result.converged),
        "reached_final_parameter": abs(final_parameter - 1.0) <= 1.0e-12,
        "active_contact": bool(np.count_nonzero(contact.signature.active_rows)),
        "equilibrium_residual": normalized_residual <= EQUILIBRIUM_TOLERANCE,
        "contact_penetration": normalized_penetration <= PENETRATION_TOLERANCE,
        "mortar_partition": maximum_partition_error <= PARTITION_TOLERANCE,
        "positive_element_jacobian": minimum_jacobian > 0.0,
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "example": "nonmatching frictionless contact patch",
        "formulation": {
            "contact": "biased single-pass standard mortar",
            "enforcement": "projected augmented Lagrange",
            "bulk": "finite-strain neo-Hookean TET4",
            "slave_surface": "one warped QUAD4",
            "master_surface": "two warped TRI3 facets",
        },
        "geometry": {
            "nodes": int(model.problem.mesh.node_count),
            "elements": int(model.problem.mesh.element_count),
            "slave_nodes": int(len(model.slave_nodes)),
            "master_nodes": int(len(model.master_nodes)),
            "initial_separation": float(model.initial_separation),
        },
        "metrics": {
            "converged": bool(result.converged),
            "final_parameter": final_parameter,
            "accepted_steps": int(len(steps)),
            "attempts": int(len(result.attempts)),
            "cutbacks": int(result.cutback_count),
            "penalty_updates": int(result.penalty_update_count),
            "contact_onset_parameter": float(contacting[0].parameter),
            "final_active_rows": int(np.count_nonzero(contact.signature.active_rows)),
            "final_supported_rows": int(
                np.count_nonzero(contact.signature.supported_rows)
            ),
            "final_facet_pairs": int(len(contact.signature.facet_pairs)),
            "final_overlap_area": float(contact.raw.contact.weights.total_area),
            "final_maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
            "final_tool_reaction_z": float(tool_reaction[2]),
            "final_support_reaction_z": float(support_reaction[2]),
            "global_reaction_balance": float(np.linalg.norm(reaction_balance)),
            "final_contact_force_balance": float(
                contact.raw.contact.force_balance_norm
            ),
            "final_normalized_residual": normalized_residual,
            "final_normalized_penetration": normalized_penetration,
            "maximum_partition_error": maximum_partition_error,
            "minimum_element_jacobian": minimum_jacobian,
        },
        "tolerances": {
            "normalized_equilibrium_residual": EQUILIBRIUM_TOLERANCE,
            "normalized_penetration": PENETRATION_TOLERANCE,
            "mortar_partition_error": PARTITION_TOLERANCE,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_final_vtk(output: Path, model: ContactPatchModel, result: object) -> None:
    final_step = result.accepted_steps[-1]
    final_problem = final_step.path_state.problem
    final_result = final_step.result
    evaluation = final_result.equilibrium.evaluation
    contact = evaluation.contacts[0]
    interface = final_problem.interfaces[0]
    node_count = final_problem.mesh.node_count

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

    global_contact_force = np.zeros(3 * node_count)
    np.add.at(global_contact_force, interface.dofs, contact.residual)
    element_count = final_problem.mesh.element_count
    body_id = np.concatenate(
        [
            np.zeros(element_count // 2, dtype=np.int64),
            np.ones(element_count - element_count // 2, dtype=np.int64),
        ]
    )
    write_tet4_vtu(
        output / "final.vtu",
        final_problem.mesh.reference_nodes,
        final_problem.mesh.elements,
        final_result.displacement,
        point_data={
            "reaction": final_step.reaction.reshape((-1, 3)),
            "effective_load": final_step.path_state.effective_force.reshape((-1, 3)),
            "contact_force": global_contact_force.reshape((-1, 3)),
            "contact_pressure": pressure,
            "normal_gap": gap,
            "contact_active": active,
            "contact_supported": supported,
        },
        cell_data={"body_id": body_id},
    )


def _write_pressure_plot(output: Path, result: object) -> None:
    contact = _step_contact(result.accepted_steps[-1])
    write_line_chart(
        output / "pressure.svg",
        title="Final mortar pressure",
        x_label="slave node row",
        y_label="normal pressure",
        x_values=np.arange(len(contact.pressure), dtype=float),
        series=((contact.pressure, "pressure"),),
        show_markers=True,
    )


def run(output: Path) -> dict[str, object]:
    """Solve the example and write summary JSON, final VTU, and one SVG plot."""

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    model = build_model()
    result = solve_adaptive_contact_path(
        model.problem,
        1.0,
        path=model.path,
        options=solver_options(),
    )
    if not result.converged:
        raise RuntimeError(f"contact-patch path failed: {result.termination_reason}")

    summary = _summary(model, result)
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_final_vtk(target, model, result)
    _write_pressure_plot(target, result)
    if not summary["passed"]:
        failed = [name for name, passed in summary["checks"].items() if not passed]
        raise RuntimeError("contact-patch checks failed: " + ", ".join(failed))
    return summary
