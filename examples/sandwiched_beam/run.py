"""Solve the v0.1 sandwiched-beam contact example."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contact3d import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    AugmentedContactOptions,
    NewtonOptions,
    ScaleAwareConvergenceOptions,
    solve_adaptive_contact_path,
    solve_equilibrium,
)
from contact3d.adaptive_model import AdaptiveAcceptedStep
from contact3d.benchmark_artifacts import write_tet4_vtu
from contact3d.benchmark_plots import write_line_chart, write_mesh_projection_overlay
from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.linear_solver import LinearSolverOptions
from contact3d.scaled_solver import solve_scale_aware_augmented_contact

from .model import SandwichedBeamModel, build_model

SUMMARY_SCHEMA = "contact3d-sandwiched-beam-example/v1"
EQUILIBRIUM_TOLERANCE = 1.0e-8
PENETRATION_TOLERANCE = 2.0e-7
FORCE_BALANCE_TOLERANCE = 1.0e-8
MOMENT_BALANCE_TOLERANCE = 1.0e-7
INITIAL_STEP = 0.05


def solver_options() -> AdaptiveContactOptions:
    """Return bounded settings for the one coarse v0.1 beam model."""

    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=INITIAL_STEP,
            minimum_step=1.0 / 4096.0,
            maximum_step=0.10,
            cutback_factor=0.5,
            growth_factor=1.4,
            easy_newton_iterations=10,
            maximum_attempts=240,
        ),
        penalty=AdaptivePenaltyOptions(
            enabled=True,
            increase_factor=2.0,
            maximum_penalty=2.0e6,
            maximum_updates_per_step=4,
            normalized_penetration_target=PENETRATION_TOLERANCE,
            interface_local=True,
            minimum_scale_factor=0.25,
            maximum_scale_factor=1.0e3,
        ),
        augmented=AugmentedContactOptions(
            maximum_augmentations=28,
            gap_tolerance=1.0e-8,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-6,
            multiplier_tolerance=1.0e-8,
            event_policy="restart",
            newton=NewtonOptions(
                maximum_iterations=50,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
                maximum_line_search_iterations=24,
                minimum_step=2.0**-20,
                linear_solver=LinearSolverOptions(
                    backend="sparse_lu",
                    dense_threshold=96,
                ),
            ),
        ),
        scaling=ScaleAwareConvergenceOptions(
            enabled=True,
            equilibrium_tolerance=EQUILIBRIUM_TOLERANCE,
            gap_tolerance=PENETRATION_TOLERANCE,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-6,
            multiplier_tolerance=1.0e-8,
        ),
    )


def _preload_multiplier_state(
    model: SandwichedBeamModel,
) -> tuple[AugmentedLagrangeState, ...]:
    """Represent the full ambient preload on the coincident zero-gap interface."""

    return (
        AugmentedLagrangeState(
            np.full(
                len(model.slave_nodes),
                model.geometry.ambient_pressure,
                dtype=float,
            ),
        ),
    )


def _constrained_reaction(equilibrium: object) -> np.ndarray:
    evaluation = equilibrium.evaluation
    residual = np.asarray(evaluation.residual, dtype=float).reshape(-1)
    free = np.asarray(evaluation.free_dofs, dtype=np.int64)
    reaction = np.zeros_like(residual)
    constrained = np.ones(len(residual), dtype=bool)
    constrained[free] = False
    reaction[constrained] = residual[constrained]
    return reaction


def _preload_failure(result: object) -> str:
    equilibrium = result.equilibrium
    contact = equilibrium.evaluation.contacts[0]
    payload: dict[str, object] = {
        "termination_reason": str(result.termination_reason),
        "equilibrium_termination_reason": str(equilibrium.termination_reason),
        "equilibrium_iterations": int(equilibrium.iteration_count),
        "contact_event_restarts": int(equilibrium.contact_event_restarts),
        "free_residual_norm": float(equilibrium.evaluation.free_residual_norm),
        "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
        "supported_rows": int(np.count_nonzero(contact.signature.supported_rows)),
        "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
        "maximum_penetration": float(contact.diagnostics.maximum_penetration),
    }
    if equilibrium.history:
        last = equilibrium.history[-1]
        payload["last_newton"] = {
            "iteration": int(last.iteration),
            "residual_norm": float(last.residual_norm),
            "relative_residual": float(last.relative_residual),
            "accepted_step": float(last.accepted_step),
            "line_search_iterations": int(last.line_search_iterations),
            "contact_branch_changed": bool(last.contact_branch_changed),
            "minimum_jacobian": float(last.minimum_jacobian),
        }
    return json.dumps(payload, sort_keys=True)


def _continuation_failure(result: object) -> str:
    payload: dict[str, object] = {
        "termination_reason": str(result.termination_reason),
        "accepted_parameter": float(result.load_factor),
        "accepted_steps": int(len(result.accepted_steps)),
        "attempts": int(len(result.attempts)),
        "cutbacks": int(result.cutback_count),
        "penalty_updates": int(result.penalty_update_count),
    }
    if result.attempts:
        last = result.attempts[-1]
        payload["last_attempt"] = {
            "attempt": int(last.attempt),
            "start_parameter": float(last.start_parameter),
            "target_parameter": float(last.target_parameter),
            "step_size": float(last.step_size),
            "action": str(last.action),
            "inner_termination_reason": str(last.inner_termination_reason),
            "augmentations": int(last.augmentations),
            "newton_iterations": int(last.newton_iterations),
            "contact_event_restarts": int(last.contact_event_restarts),
            "normalized_equilibrium_residual": float(
                last.normalized_equilibrium_residual
            ),
            "normalized_maximum_penetration": float(
                last.normalized_maximum_penetration
            ),
            "penalties_before": last.penalties_before,
            "penalties_after": last.penalties_after,
        }
    return json.dumps(payload, sort_keys=True)


def _solve_contact_path(model: SandwichedBeamModel) -> object:
    """Establish the physical preload directly, then continue through bending."""

    settings = solver_options()
    preload_parameter = model.geometry.compression_end
    preload_state = model.path.evaluate(model.problem, preload_parameter)
    preload = solve_scale_aware_augmented_contact(
        preload_state.problem,
        np.zeros(3 * preload_state.problem.mesh.node_count),
        _preload_multiplier_state(model),
        load_factor=preload_state.solver_load_factor,
        options=settings.augmented,
        scaling=settings.scaling,
    )
    if not preload.converged:
        raise RuntimeError(
            "sandwiched-beam preload failed: " + _preload_failure(preload)
        )

    preload_step = AdaptiveAcceptedStep(
        preload_state,
        preload,
        _constrained_reaction(preload.equilibrium),
    )
    continuation = solve_adaptive_contact_path(
        preload_state.problem,
        1.0,
        initial_displacement=preload.displacement,
        initial_states=preload.states,
        initial_load_factor=preload_parameter,
        path=model.path,
        options=settings,
    )
    if not continuation.converged:
        raise RuntimeError(
            "sandwiched-beam bending path failed: "
            + _continuation_failure(continuation)
        )

    return SimpleNamespace(
        converged=True,
        termination_reason="converged",
        load_factor=continuation.load_factor,
        accepted_steps=(preload_step, *continuation.accepted_steps),
        attempts=continuation.attempts,
        cutback_count=continuation.cutback_count,
        penalty_update_count=continuation.penalty_update_count,
    )


def _end_response(
    nodes: np.ndarray,
    displacement: np.ndarray,
    interface_z: float,
) -> tuple[float, float]:
    """Return mean transverse end displacement and section rotation."""

    reference = np.asarray(nodes, dtype=float)
    values = np.asarray(displacement, dtype=float).reshape((-1, 3))
    end_nodes = np.flatnonzero(np.isclose(reference[:, 0], np.max(reference[:, 0])))
    lever = reference[end_nodes, 2] - interface_z
    centered = lever - float(np.mean(lever))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).tiny:
        raise RuntimeError("beam end section cannot represent a rotation")
    axial = values[end_nodes, 0]
    rotation = float(np.dot(centered, axial - np.mean(axial)) / denominator)
    transverse = float(np.mean(values[end_nodes, 2]))
    return transverse, rotation


def _resultant_moment(
    nodes: np.ndarray,
    force: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(force, dtype=float).reshape((-1, 3))
    return np.sum(np.cross(np.asarray(nodes, dtype=float) - origin, vectors), axis=0)


def _contact_history(model: SandwichedBeamModel, result: object) -> list[dict[str, object]]:
    nodes = model.problem.mesh.reference_nodes
    origin = np.asarray((0.0, 0.0, model.geometry.interface_z))
    force_scale = max(
        model.geometry.ambient_pressure * model.geometry.loaded_area,
        np.finfo(float).tiny,
    )
    moment_scale = max(model.geometry.end_moment, np.finfo(float).tiny)
    rows: list[dict[str, object]] = []
    for step in result.accepted_steps:
        solved = step.result
        evaluation = solved.equilibrium.evaluation
        contact = evaluation.contacts[0]
        normalized = solved.scales.interfaces[0].normalize_kkt(contact.diagnostics)
        reaction = step.reaction.reshape((-1, 3))
        applied = step.path_state.effective_force.reshape((-1, 3))
        force_balance = np.sum(reaction + applied, axis=0)
        moment_balance = _resultant_moment(nodes, reaction + applied, origin)
        transverse, rotation = _end_response(
            nodes,
            solved.displacement,
            model.geometry.interface_z,
        )
        rows.append(
            {
                "parameter": float(step.parameter),
                "phase": model.path.phase_name(step.parameter),
                "pressure_scale": float(step.path_state.value("pressure_scale")),
                "moment_scale": float(step.path_state.value("moment_scale")),
                "applied_moment": (
                    model.geometry.end_moment
                    * float(step.path_state.value("moment_scale"))
                ),
                "end_transverse_displacement": transverse,
                "end_rotation": rotation,
                "normalized_equilibrium_residual": float(
                    evaluation.free_residual_norm / solved.scales.force
                ),
                "normalized_penetration": float(normalized.maximum_penetration),
                "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
                "minimum_pressure": float(np.min(contact.pressure, initial=0.0)),
                "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
                "supported_rows": int(
                    np.count_nonzero(contact.signature.supported_rows)
                ),
                "overlap_area": float(contact.raw.contact.weights.total_area),
                "force_balance_relative": float(np.linalg.norm(force_balance) / force_scale),
                "moment_balance_relative": float(
                    np.linalg.norm(moment_balance) / moment_scale
                ),
                "minimum_element_jacobian": float(evaluation.bulk.minimum_jacobian),
            }
        )
    return rows


def _reference_parameters(compression_end: float) -> np.ndarray:
    compression = np.linspace(compression_end / 5.0, compression_end, 5)
    bending = np.linspace(compression_end, 1.0, 13)[1:]
    return np.concatenate([compression, bending])


def _solve_reference(model: SandwichedBeamModel) -> list[dict[str, object]]:
    """Solve the conforming monolithic comparison along a modest fixed path."""

    options = NewtonOptions(
        maximum_iterations=60,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        maximum_line_search_iterations=24,
        minimum_step=2.0**-20,
        linear_solver=LinearSolverOptions(
            backend="sparse_lu",
            dense_threshold=96,
        ),
    )
    displacement: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    nodes = model.reference_problem.mesh.reference_nodes
    for parameter in _reference_parameters(model.geometry.compression_end):
        force = model.reference_path.force(float(parameter))
        problem = replace(model.reference_problem, load=model.reference_path.load(parameter))
        solved = solve_equilibrium(problem, displacement, options=options)
        if not solved.converged:
            raise RuntimeError(
                "sandwiched-beam reference solve failed at "
                f"{parameter:.6f}: {solved.termination_reason}"
            )
        displacement = solved.displacement.copy()
        transverse, rotation = _end_response(
            nodes,
            displacement,
            model.geometry.interface_z,
        )
        force_norm = max(float(np.linalg.norm(force)), np.finfo(float).tiny)
        _, moment_factor, _, _ = model.reference_path.scales(float(parameter))
        rows.append(
            {
                "parameter": float(parameter),
                "applied_moment": model.geometry.end_moment * moment_factor,
                "end_transverse_displacement": transverse,
                "end_rotation": rotation,
                "normalized_equilibrium_residual": float(
                    solved.evaluation.free_residual_norm / force_norm
                ),
                "minimum_element_jacobian": float(
                    solved.evaluation.bulk.minimum_jacobian
                ),
                "newton_iterations": int(solved.iteration_count),
            }
        )
    return rows


def _write_final_vtk(output: Path, model: SandwichedBeamModel, result: object) -> None:
    final_step = result.accepted_steps[-1]
    final_problem = final_step.path_state.problem
    solved = final_step.result
    evaluation = solved.equilibrium.evaluation
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
    body_id = np.concatenate(
        [
            np.zeros(len(model.lower_elements), dtype=np.int64),
            np.ones(len(model.upper_elements), dtype=np.int64),
        ]
    )
    write_tet4_vtu(
        output / "final.vtu",
        final_problem.mesh.reference_nodes,
        final_problem.mesh.elements,
        solved.displacement,
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


def _write_plots(
    output: Path,
    model: SandwichedBeamModel,
    result: object,
    contact_history: list[dict[str, object]],
    reference_history: list[dict[str, object]],
) -> None:
    final = result.accepted_steps[-1].result.displacement.reshape((-1, 3))
    reference_nodes = model.problem.mesh.reference_nodes
    write_mesh_projection_overlay(
        output / "deformed.svg",
        title="Sandwiched beam: reference and final x-z mesh",
        reference_nodes=reference_nodes,
        current_nodes=reference_nodes + final,
        elements=model.problem.mesh.elements,
        axes=(0, 2),
    )

    parameters = np.asarray([float(row["parameter"]) for row in contact_history])
    contact_rotation = np.asarray(
        [float(row["end_rotation"]) for row in contact_history]
    )
    reference_parameter = np.asarray(
        [float(row["parameter"]) for row in reference_history]
    )
    reference_rotation = np.asarray(
        [float(row["end_rotation"]) for row in reference_history]
    )
    interpolated_reference = np.interp(
        parameters,
        reference_parameter,
        reference_rotation,
    )
    write_line_chart(
        output / "moment-rotation.svg",
        title="Sandwiched beam moment-rotation response",
        x_label="path parameter",
        y_label="end-section rotation",
        x_values=parameters,
        series=(
            (contact_rotation, "nonmatching mortar"),
            (interpolated_reference, "conforming reference"),
        ),
        show_markers=True,
    )


def _summary(
    model: SandwichedBeamModel,
    result: object,
    contact_history: list[dict[str, object]],
    reference_history: list[dict[str, object]],
) -> dict[str, object]:
    if not contact_history:
        raise RuntimeError("sandwiched-beam solve produced no accepted states")
    final = contact_history[-1]
    reference_final = reference_history[-1]
    bending_rows = [
        row
        for row in contact_history
        if float(row["parameter"]) >= model.geometry.compression_end
    ]
    contact_rotation = float(final["end_rotation"])
    reference_rotation = float(reference_final["end_rotation"])
    rotation_scale = max(abs(reference_rotation), np.finfo(float).tiny)
    rotation_relative_difference = abs(contact_rotation - reference_rotation) / rotation_scale
    checks = {
        "solver_converged": bool(result.converged),
        "reached_final_parameter": abs(float(final["parameter"]) - 1.0) <= 1.0e-12,
        "contact_supported_during_bending": bool(bending_rows)
        and all(int(row["supported_rows"]) > 0 for row in bending_rows),
        "compressive_contact_during_bending": bool(bending_rows)
        and all(float(row["minimum_pressure"]) >= -1.0e-12 for row in bending_rows)
        and all(int(row["active_rows"]) > 0 for row in bending_rows),
        "equilibrium_residual": float(final["normalized_equilibrium_residual"])
        <= EQUILIBRIUM_TOLERANCE,
        "contact_penetration": float(final["normalized_penetration"])
        <= PENETRATION_TOLERANCE,
        "force_balance": float(final["force_balance_relative"])
        <= FORCE_BALANCE_TOLERANCE,
        "moment_balance": float(final["moment_balance_relative"])
        <= MOMENT_BALANCE_TOLERANCE,
        "positive_element_jacobian": min(
            float(row["minimum_element_jacobian"]) for row in contact_history
        )
        > 0.0,
        "reference_converged": float(reference_final["parameter"]) == 1.0,
        "nonzero_bending_response": abs(contact_rotation) > 1.0e-6
        and contact_rotation * reference_rotation > 0.0,
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "example": "nonmatching frictionless sandwiched-beam bending",
        "formulation": {
            "contact": "biased single-pass standard mortar",
            "enforcement": "projected augmented Lagrange",
            "bulk": "finite-strain neo-Hookean TET4",
            "model_level": model.level.name,
            "slave_side": model.slave_side,
            "reference": "conforming monolithic TET4 beam",
        },
        "geometry": {
            "length": model.geometry.length,
            "width": model.geometry.width,
            "beam_thickness": model.geometry.beam_thickness,
            "contact_nodes": int(model.problem.mesh.node_count),
            "contact_elements": int(model.problem.mesh.element_count),
            "reference_nodes": int(model.reference_problem.mesh.node_count),
            "reference_elements": int(model.reference_problem.mesh.element_count),
        },
        "loads": {
            "ambient_pressure": model.geometry.ambient_pressure,
            "end_moment": model.geometry.end_moment,
            "compression_end": model.geometry.compression_end,
            "preload_multiplier_predictor": model.geometry.ambient_pressure,
        },
        "metrics": {
            "converged": bool(result.converged),
            "final_parameter": float(final["parameter"]),
            "accepted_steps": int(len(result.accepted_steps)),
            "attempts": int(len(result.attempts)),
            "cutbacks": int(result.cutback_count),
            "penalty_updates": int(result.penalty_update_count),
            "final_end_transverse_displacement": float(
                final["end_transverse_displacement"]
            ),
            "final_end_rotation": contact_rotation,
            "reference_end_rotation": reference_rotation,
            "rotation_relative_difference": rotation_relative_difference,
            "final_maximum_pressure": float(final["maximum_pressure"]),
            "final_active_rows": int(final["active_rows"]),
            "final_supported_rows": int(final["supported_rows"]),
            "final_overlap_area": float(final["overlap_area"]),
            "final_normalized_equilibrium_residual": float(
                final["normalized_equilibrium_residual"]
            ),
            "final_normalized_penetration": float(final["normalized_penetration"]),
            "final_force_balance_relative": float(final["force_balance_relative"]),
            "final_moment_balance_relative": float(final["moment_balance_relative"]),
            "minimum_element_jacobian": min(
                float(row["minimum_element_jacobian"]) for row in contact_history
            ),
        },
        "tolerances": {
            "normalized_equilibrium_residual": EQUILIBRIUM_TOLERANCE,
            "normalized_penetration": PENETRATION_TOLERANCE,
            "relative_force_balance": FORCE_BALANCE_TOLERANCE,
            "relative_moment_balance": MOMENT_BALANCE_TOLERANCE,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "history": contact_history,
        "reference_history": reference_history,
    }


def run(output: Path) -> dict[str, object]:
    """Solve contact and conforming paths and write four inspectable files."""

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    model = build_model()
    result = _solve_contact_path(model)
    contact_history = _contact_history(model, result)
    reference_history = _solve_reference(model)
    summary = _summary(model, result, contact_history, reference_history)
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_final_vtk(target, model, result)
    _write_plots(target, model, result, contact_history, reference_history)
    if not summary["passed"]:
        failed = [name for name, passed in summary["checks"].items() if not passed]
        raise RuntimeError("sandwiched-beam checks failed: " + ", ".join(failed))
    return summary
