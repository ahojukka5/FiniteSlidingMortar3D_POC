#!/usr/bin/env python3
"""Solve one warped nonmatching contact-patch refinement case."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from contact3d import LinearBoundaryPath, MortarContactInterface
from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    ScaleAwareConvergenceOptions,
)
from contact3d.clipping import ClippingTopologyError
from contact3d.coupled import AugmentedContactOptions
from contact3d.equilibrium import NewtonOptions
from contact3d.event_solver import solve_event_aware_adaptive_contact_path
from contact3d.linear_solver import LinearSolverOptions
from contact3d.pallets import PalletTopologyError
from contact3d.parametric import InverseMapTopologyError

try:
    from .warped_patch_model import (
        WarpedPatchModel,
        WarpedPatchProfile,
        build_warped_patch_model,
        manufactured_displacement,
        reference_pressure,
        reference_vertical_reaction,
    )
except ImportError:  # Direct script execution from the repository root.
    from warped_patch_model import (
        WarpedPatchModel,
        WarpedPatchProfile,
        build_warped_patch_model,
        manufactured_displacement,
        reference_pressure,
        reference_vertical_reaction,
    )

CASE_SCHEMA = "contact3d-warped-patch-case/v1"
INTERFACE_SCHEMA = "contact3d-warped-patch-interface/v1"
ATTEMPT_SCHEMA = "contact3d-warped-patch-attempt/v1"
PRELOAD_FRACTION = 0.90
CONVERGENCE_PROFILES = {
    "coarse": WarpedPatchProfile("coarse", (2, 2, 1), (3, 3, 1)),
    "medium": WarpedPatchProfile("medium", (3, 3, 2), (4, 4, 2)),
    "fine": WarpedPatchProfile("fine", (4, 4, 2), (5, 5, 2)),
}
_TOPOLOGY_ERRORS = (
    ClippingTopologyError,
    PalletTopologyError,
    InverseMapTopologyError,
)


@dataclass(frozen=True, slots=True)
class InactiveSafeMortarContactInterface(MortarContactInterface):
    """Use exact inactive and branch-contained special-state tangents."""

    def _branch_contained_tangent(
        self,
        displacement: np.ndarray,
        state: object,
        evaluation: object,
        *,
        tolerance: float,
    ) -> np.ndarray:
        values = np.asarray(displacement, dtype=float)
        baseline = np.asarray(evaluation.residual, dtype=float)
        tangent = np.zeros((len(baseline), len(self.dofs)), dtype=float)
        base_step = 2.0e-7 * max(1.0, float(np.linalg.norm(values)))
        for column, dof in enumerate(self.dofs):
            derivative: np.ndarray | None = None
            for refinement in range(12):
                step = base_step * 0.5**refinement
                branches: dict[int, np.ndarray] = {}
                for sign in (-1, 1):
                    trial = values.copy()
                    trial[int(dof)] += sign * step
                    try:
                        candidate = self.evaluate(
                            trial,
                            state,
                            tolerance=tolerance,
                        )
                    except _TOPOLOGY_ERRORS:
                        continue
                    if candidate.signature == evaluation.signature:
                        branches[sign] = np.asarray(candidate.residual, dtype=float)
                if -1 in branches and 1 in branches:
                    derivative = (branches[1] - branches[-1]) / (2.0 * step)
                    break
                if 1 in branches:
                    derivative = (branches[1] - baseline) / step
                    break
                if -1 in branches:
                    derivative = (baseline - branches[-1]) / step
                    break
            if derivative is not None:
                tangent[:, column] = derivative
        return tangent

    def tangent(
        self,
        displacement: np.ndarray,
        state: object,
        evaluation: object,
        *,
        tolerance: float,
    ) -> np.ndarray:
        if not any(evaluation.signature.active_rows):
            size = len(self.dofs)
            return np.zeros((size, size), dtype=float)
        try:
            return MortarContactInterface.tangent(
                self,
                displacement,
                state,
                evaluation,
                tolerance=tolerance,
            )
        except _TOPOLOGY_ERRORS:
            return self._branch_contained_tangent(
                displacement,
                state,
                evaluation,
                tolerance=tolerance,
            )


@dataclass(frozen=True, slots=True)
class WarpedPatchCaseRun:
    """One converged level with stable rows for campaign reporting."""

    model: WarpedPatchModel
    result: object
    metrics: dict[str, object]
    interface_rows: tuple[dict[str, object], ...]
    attempt_rows: tuple[dict[str, object], ...]
    event_rows: tuple[dict[str, object], ...]

    @property
    def case_id(self) -> str:
        return str(self.metrics["case_id"])


def solver_options(*, publication: bool = False) -> AdaptiveContactOptions:
    """Return bounded production controls for the warped patch campaign."""

    scaling = ScaleAwareConvergenceOptions(
        enabled=True,
        equilibrium_tolerance=1.0e-8,
        gap_tolerance=2.0e-7,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-5,
        multiplier_tolerance=1.0e-8,
    )
    augmented = AugmentedContactOptions(
        maximum_augmentations=32 if publication else 24,
        gap_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-5,
        multiplier_tolerance=1.0e-8,
        event_policy="restart",
        newton=NewtonOptions(
            maximum_iterations=50 if publication else 40,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
            maximum_line_search_iterations=24,
            minimum_step=2.0**-20,
            linear_solver=LinearSolverOptions(
                backend="sparse_lu",
                dense_threshold=96,
            ),
        ),
    )
    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.10 if publication else 1.0,
            minimum_step=1.0 / 4096.0,
            maximum_step=0.20 if publication else 1.0,
            cutback_factor=0.5,
            growth_factor=1.5,
            easy_newton_iterations=10,
            maximum_attempts=260 if publication else 200,
        ),
        penalty=AdaptivePenaltyOptions(
            enabled=True,
            increase_factor=2.0,
            maximum_penalty=2.0e6,
            maximum_updates_per_step=4,
            normalized_penetration_target=scaling.gap_tolerance,
            interface_local=True,
            minimum_scale_factor=0.25,
            maximum_scale_factor=1.0e3,
        ),
        augmented=augmented,
        scaling=scaling,
    )


def _with_inactive_safe_interface(model: WarpedPatchModel) -> WarpedPatchModel:
    base = model.interface
    interface = InactiveSafeMortarContactInterface(
        base.pair,
        base.slave_nodes,
        base.master_nodes,
    )
    problem = replace(model.problem, interfaces=(interface,))
    return replace(model, problem=problem)


def _preloaded_contact_path(
    model: WarpedPatchModel,
) -> tuple[LinearBoundaryPath, np.ndarray]:
    """Return a contact-regime path and a compatible displacement predictor."""

    end = model.problem.constraints
    start = replace(end, values=PRELOAD_FRACTION * end.values)
    path = LinearBoundaryPath(
        start,
        end,
        model.problem.load,
        model.problem.load,
    )
    predictor = PRELOAD_FRACTION * manufactured_displacement(model)
    return path, predictor


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= np.finfo(float).tiny:
        return float(np.sqrt(np.mean(np.square(values))))
    return float(np.sqrt(np.sum(weights * np.square(values)) / total))


def _attempt_rows(result: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "attempt": int(item.attempt),
            "start_parameter": float(item.start_parameter),
            "target_parameter": float(item.target_parameter),
            "step_size": float(item.step_size),
            "action": str(item.action),
            "inner_termination_reason": str(item.inner_termination_reason),
            "augmentations": int(item.augmentations),
            "newton_iterations": int(item.newton_iterations),
            "contact_event_restarts": int(item.contact_event_restarts),
            "normalized_equilibrium_residual": float(
                item.normalized_equilibrium_residual
            ),
            "normalized_maximum_penetration": float(
                item.normalized_maximum_penetration
            ),
            "penalties_before": item.penalties_before,
            "penalties_after": item.penalties_after,
            "penalty_update_reasons": item.penalty_update_reasons,
        }
        for item in tuple(getattr(result, "attempts", ()))
    )


def _terminal_diagnostics(
    result: object,
    attempt_results: tuple[object, ...],
) -> dict[str, object]:
    """Reduce the last failed adaptive solve to a compact actionable record."""

    attempts = tuple(getattr(result, "attempts", ()))
    diagnostics: dict[str, object] = {
        "termination_reason": str(getattr(result, "termination_reason", "unknown")),
        "accepted_parameter": float(getattr(result, "load_factor", 0.0)),
        "accepted_steps": len(tuple(getattr(result, "accepted_steps", ()))),
        "attempts": len(attempts),
        "cutbacks": int(getattr(result, "cutback_count", 0)),
        "penalty_updates": int(getattr(result, "penalty_update_count", 0)),
    }
    if attempts:
        attempt = attempts[-1]
        diagnostics["last_attempt"] = {
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
            "penalties_before": attempt.penalties_before,
            "penalties_after": attempt.penalties_after,
            "penalty_update_reasons": attempt.penalty_update_reasons,
        }
    if attempt_results:
        terminal = attempt_results[-1]
        equilibrium = getattr(terminal, "equilibrium", None)
        diagnostics["last_result"] = {
            "termination_reason": str(
                getattr(terminal, "termination_reason", "unknown")
            ),
            "equilibrium_termination_reason": str(
                getattr(equilibrium, "termination_reason", "unknown")
            ),
            "equilibrium_iterations": int(
                getattr(equilibrium, "iteration_count", 0)
            ),
        }
        history = tuple(getattr(terminal, "history", ()))
        if history:
            row = history[-1]
            diagnostics["last_augmentation"] = {
                "augmentation": int(row.augmentation),
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
                "maximum_pressure": float(row.maximum_pressure),
            }
        linear = getattr(equilibrium, "linear_solve_failure", None)
        if linear is not None:
            diagnostics["linear_solve_failure"] = {
                "requested_backend": str(linear.requested_backend),
                "backend": str(linear.backend),
                "converged": bool(linear.converged),
                "failure_reason": linear.failure_reason,
                "matrix_shape": linear.matrix_shape,
                "matrix_nnz": int(linear.matrix_nnz),
                "residual_norm": float(linear.residual_norm),
                "relative_residual": float(linear.relative_residual),
            }
        newton_history = tuple(getattr(equilibrium, "history", ()))
        if newton_history:
            row = newton_history[-1]
            diagnostics["last_newton"] = {
                "iteration": int(row.iteration),
                "residual_norm": float(row.residual_norm),
                "relative_residual": float(row.relative_residual),
                "accepted_step": float(row.accepted_step),
                "line_search_iterations": int(row.line_search_iterations),
                "contact_branch_changed": bool(row.contact_branch_changed),
                "minimum_jacobian": float(row.minimum_jacobian),
            }
    return diagnostics


def _interface_rows(
    model: WarpedPatchModel,
    step: object,
) -> tuple[dict[str, object], ...]:
    contact = step.result.equilibrium.evaluation.contacts[0]
    weights = contact.raw.contact.weights
    current = model.problem.mesh.reference_nodes + step.result.displacement.reshape((-1, 3))
    slave_nodes = model.interface.slave_nodes
    rows: list[dict[str, object]] = []
    for row, node in enumerate(slave_nodes):
        rows.append(
            {
                "case_id": (
                    f"{model.surface_family.name}-{model.bias_side}-{model.profile.name}"
                ),
                "profile": model.profile.name,
                "surface_family": model.surface_family.name,
                "bias_side": model.bias_side,
                "slave_row": row,
                "global_node": int(node),
                "x": float(current[node, 0]),
                "y": float(current[node, 1]),
                "z": float(current[node, 2]),
                "row_area": float(weights.row_areas[row]),
                "normal_gap": float(contact.normal_gaps[row]),
                "pressure": float(contact.pressure[row]),
                "multiplier": float(step.result.states[0].multipliers[row]),
                "active": bool(contact.signature.active_rows[row]),
                "supported": bool(contact.signature.supported_rows[row]),
            }
        )
    return tuple(rows)


def _case_metrics(
    model: WarpedPatchModel,
    result: object,
    attempt_results: tuple[object, ...],
    elapsed_seconds: float,
) -> dict[str, object]:
    if not result.converged or not result.accepted_steps:
        raise RuntimeError(
            "warped patch case failed: "
            + json.dumps(
                _terminal_diagnostics(result, attempt_results),
                sort_keys=True,
            )
        )
    step = result.accepted_steps[-1]
    solved = step.result
    evaluation = solved.equilibrium.evaluation
    contact = evaluation.contacts[0]
    weights = contact.raw.contact.weights
    row_areas = np.asarray(weights.row_areas, dtype=float)
    reference = manufactured_displacement(model)
    free = evaluation.free_dofs
    displacement_error = solved.displacement[free] - reference[free]
    displacement_scale = max(
        float(np.linalg.norm(reference[free])),
        np.sqrt(max(1, len(free))) * np.finfo(float).eps,
    )
    pressure_reference = reference_pressure(model)
    reaction_reference = reference_vertical_reaction(model)
    pressure_error = _weighted_rms(contact.pressure - pressure_reference, row_areas)
    gap_rms = _weighted_rms(contact.normal_gaps, row_areas)
    normalized = solved.scales.interfaces[0].normalize_kkt(contact.diagnostics)
    nodal_reaction = step.reaction.reshape((-1, 3))
    controlled_reaction = abs(
        float(np.sum(nodal_reaction[model.controlled_nodes, 2]))
    )
    global_balance = float(np.linalg.norm(np.sum(nodal_reaction, axis=0)))
    contact_balance = float(
        np.linalg.norm(contact.residual.reshape((-1, 3)).sum(axis=0))
    )
    force_scale = max(reaction_reference, np.finfo(float).tiny)
    case_id = f"{model.surface_family.name}-{model.bias_side}-{model.profile.name}"
    topology_events = len(tuple(getattr(result, "events", ())))
    return {
        "schema_version": CASE_SCHEMA,
        "case_id": case_id,
        "profile": model.profile.name,
        "surface_family": model.surface_family.name,
        "bias_side": model.bias_side,
        "characteristic_size": model.characteristic_size,
        "warp_amplitude": model.warp_amplitude,
        "node_count": model.problem.mesh.node_count,
        "element_count": model.problem.mesh.element_count,
        "free_dofs": len(free),
        "slave_nodes": len(model.interface.slave_nodes),
        "master_nodes": len(model.interface.master_nodes),
        "preload_fraction": PRELOAD_FRACTION,
        "converged": bool(result.converged),
        "final_parameter": float(result.load_factor),
        "accepted_steps": len(result.accepted_steps),
        "attempts": len(result.attempts),
        "cutbacks": int(result.cutback_count),
        "penalty_updates": int(result.penalty_update_count),
        "topology_events": topology_events,
        "elapsed_seconds": elapsed_seconds,
        "displacement_relative_l2_error": (
            float(np.linalg.norm(displacement_error)) / displacement_scale
        ),
        "reaction": controlled_reaction,
        "reference_reaction": reaction_reference,
        "reaction_relative_error": abs(controlled_reaction - reaction_reference)
        / force_scale,
        "pressure_weighted_l2_error": pressure_error,
        "pressure_relative_l2_error": pressure_error
        / max(pressure_reference, np.finfo(float).tiny),
        "gap_weighted_l2": gap_rms,
        "gap_over_h": gap_rms / model.characteristic_size,
        "maximum_penetration": contact.diagnostics.maximum_penetration,
        "normalized_maximum_penetration": normalized.maximum_penetration,
        "maximum_complementarity": contact.diagnostics.maximum_complementarity,
        "normalized_maximum_complementarity": normalized.maximum_complementarity,
        "maximum_projection_residual": (
            contact.diagnostics.maximum_projection_residual
        ),
        "normalized_maximum_projection_residual": (
            normalized.maximum_projection_residual
        ),
        "normalized_equilibrium_residual": (
            evaluation.free_residual_norm / solved.scales.force
        ),
        "overlap_area": float(weights.total_area),
        "overlap_area_error": abs(float(weights.total_area) - 1.0),
        "partition_error": float(weights.consistency_error),
        "global_force_balance_relative": global_balance / force_scale,
        "contact_force_balance_relative": contact_balance / force_scale,
        "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
        "minimum_pressure": float(np.min(contact.pressure, initial=0.0)),
        "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
        "supported_rows": int(np.count_nonzero(contact.signature.supported_rows)),
        "facet_pairs": len(contact.signature.facet_pairs),
        "minimum_jacobian": evaluation.bulk.minimum_jacobian,
    }


def solve_warped_patch_case(
    profile: str,
    *,
    surface_family: str = "quad-quad",
    bias_side: str = "upper",
    publication: bool = False,
    _solver: object | None = None,
) -> WarpedPatchCaseRun:
    """Build, solve, and reduce one warped patch refinement case."""

    try:
        selected_profile = CONVERGENCE_PROFILES[profile]
    except KeyError as error:
        raise ValueError("warped patch profile must be coarse, medium, or fine") from error
    model = _with_inactive_safe_interface(
        build_warped_patch_model(
            selected_profile,
            surface_family=surface_family,
            bias_side=bias_side,
        )
    )
    settings = solver_options(publication=publication)
    path, initial_displacement = _preloaded_contact_path(model)
    selected_solver = (
        solve_event_aware_adaptive_contact_path if _solver is None else _solver
    )
    started = perf_counter()
    result = selected_solver(
        model.problem,
        1.0,
        initial_displacement,
        options=settings,
        path=path,
    )
    elapsed = perf_counter() - started
    attempt_results = tuple(getattr(result, "attempt_results", ()))
    metrics = _case_metrics(model, result, attempt_results, elapsed)
    final_step = result.accepted_steps[-1]
    event_rows_method = getattr(result, "event_rows", None)
    event_rows = () if event_rows_method is None else tuple(event_rows_method())
    return WarpedPatchCaseRun(
        model=model,
        result=result,
        metrics=metrics,
        interface_rows=_interface_rows(model, final_step),
        attempt_rows=_attempt_rows(result),
        event_rows=event_rows,
    )
