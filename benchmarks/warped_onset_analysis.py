"""State histories and smooth tangent checks for warped contact onset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact3d import evaluate_coupled_equilibrium


@dataclass(frozen=True, slots=True)
class OnsetHistories:
    step_rows: list[dict[str, object]]
    interface_rows: list[dict[str, object]]
    tangent_rows: list[dict[str, object]]
    attempt_rows: list[dict[str, object]]
    separated_steps: tuple[object, ...]
    contacting_steps: tuple[object, ...]


def directional_tangent_error(
    step: object,
    *,
    seed: int,
    increment: float = 2.0e-7,
) -> dict[str, object]:
    """Compare the analytical coupled tangent with a centered directional oracle."""

    problem = step.path_state.problem
    displacement = step.result.displacement
    states = step.result.states
    factor = step.path_state.solver_load_factor
    base = evaluate_coupled_equilibrium(
        problem,
        displacement,
        states,
        load_factor=factor,
    )
    if base.tangent is None:
        raise RuntimeError("coupled tangent was not assembled")

    rng = np.random.default_rng(seed)
    direction = np.zeros_like(displacement)
    direction[base.free_dofs] = rng.normal(size=len(base.free_dofs))
    direction /= np.linalg.norm(direction)
    plus = evaluate_coupled_equilibrium(
        problem,
        displacement + increment * direction,
        states,
        load_factor=factor,
        assemble_tangent=False,
    )
    minus = evaluate_coupled_equilibrium(
        problem,
        displacement - increment * direction,
        states,
        load_factor=factor,
        assemble_tangent=False,
    )
    if plus.signatures != base.signatures or minus.signatures != base.signatures:
        raise RuntimeError("directional tangent check crossed a discrete contact event")

    numerical = (plus.residual - minus.residual) / (2.0 * increment)
    analytical = base.tangent.matvec(direction)
    free = base.free_dofs
    denominator = max(float(np.linalg.norm(numerical[free])), np.finfo(float).tiny)
    error = float(np.linalg.norm(analytical[free] - numerical[free]) / denominator)
    return {
        "parameter": step.parameter,
        "relative_error": error,
        "increment": increment,
        "active_rows": int(np.count_nonzero(base.contacts[0].signature.active_rows)),
        "facet_pairs": len(base.contacts[0].signature.facet_pairs),
    }


def _step_histories(
    result: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    step_rows: list[dict[str, object]] = []
    interface_rows: list[dict[str, object]] = []
    for step_index, step in enumerate(result.accepted_steps, start=1):
        equilibrium = step.result.equilibrium
        evaluation = equilibrium.evaluation
        contact = evaluation.contacts[0]
        raw = contact.raw.contact
        nodal_reaction = step.reaction.reshape((-1, 3))
        support_reaction = np.sum(nodal_reaction[0:4], axis=0)
        tool_reaction = np.sum(nodal_reaction[13:17], axis=0)
        applied = step.path_state.effective_force.reshape((-1, 3)).sum(axis=0)
        reaction_balance = np.sum(nodal_reaction, axis=0) + applied
        scale = step.result.scales.interfaces[0]
        normalized = scale.normalize_kkt(contact.diagnostics)
        step_rows.append(
            {
                "accepted_step": step_index,
                "parameter": step.parameter,
                "tool_x": step.path_state.value("tool_x"),
                "tool_z": step.path_state.value("tool_z"),
                "tool_reaction_x": float(tool_reaction[0]),
                "tool_reaction_y": float(tool_reaction[1]),
                "tool_reaction_z": float(tool_reaction[2]),
                "support_reaction_x": float(support_reaction[0]),
                "support_reaction_y": float(support_reaction[1]),
                "support_reaction_z": float(support_reaction[2]),
                "reaction_norm": step.reaction_norm,
                "global_reaction_balance": float(np.linalg.norm(reaction_balance)),
                "free_residual": evaluation.free_residual_norm,
                "normalized_residual": (
                    evaluation.free_residual_norm / step.result.scales.force
                ),
                "minimum_jacobian": evaluation.bulk.minimum_jacobian,
                "maximum_penetration": contact.diagnostics.maximum_penetration,
                "normalized_penetration": normalized.maximum_penetration,
                "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
                "facet_pairs": len(contact.signature.facet_pairs),
                "overlap_area": raw.weights.total_area,
                "partition_error": raw.weights.consistency_error,
                "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
                "supported_rows": int(
                    np.count_nonzero(contact.signature.supported_rows)
                ),
                "contact_event_restarts": sum(
                    value.contact_event_restarts for value in step.result.equilibria
                ),
                "newton_iterations": sum(
                    value.iteration_count for value in step.result.equilibria
                ),
                "augmentations": len(step.result.history),
                "penalty": step.path_state.problem.interfaces[0].pair.normal_penalty,
            }
        )
        for row, (gap, pressure, active, supported) in enumerate(
            zip(
                contact.normal_gaps,
                contact.pressure,
                contact.signature.active_rows,
                contact.signature.supported_rows,
                strict=True,
            )
        ):
            interface_rows.append(
                {
                    "accepted_step": step_index,
                    "parameter": step.parameter,
                    "slave_row": row,
                    "normal_gap": float(gap),
                    "pressure": float(pressure),
                    "multiplier": float(step.result.states[0].multipliers[row]),
                    "active": bool(active),
                    "supported": bool(supported),
                }
            )
    return step_rows, interface_rows


def _classified_steps(
    result: object,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    separated = tuple(
        step
        for step in result.accepted_steps
        if not any(step.result.equilibrium.evaluation.contacts[0].signature.active_rows)
    )
    contacting = tuple(
        step
        for step in result.accepted_steps
        if any(step.result.equilibrium.evaluation.contacts[0].signature.active_rows)
    )
    if not separated or not contacting:
        raise RuntimeError("benchmark path did not contain both separated and contacting states")
    return separated, contacting


def _tangent_rows(
    separated: tuple[object, ...],
    contacting: tuple[object, ...],
) -> list[dict[str, object]]:
    robust_contacting = []
    for step in contacting:
        contact = step.result.equilibrium.evaluation.contacts[0]
        active = np.asarray(contact.signature.active_rows, dtype=bool)
        active_pressure = contact.pressure[active]
        pressure_scale = step.result.scales.interfaces[0].pressure
        if len(active_pressure) and float(np.min(active_pressure)) > 1.0e-6 * pressure_scale:
            robust_contacting.append(step)
    if not robust_contacting:
        raise RuntimeError("no active state lies safely away from the pressure-projection kink")

    selected = (
        ("separated", separated[-1], 2711),
        ("first_contact", robust_contacting[0], 2712),
        ("established", robust_contacting[-1], 2713),
    )
    rows = []
    for label, step, seed in selected:
        row = directional_tangent_error(step, seed=seed)
        row["state"] = label
        rows.append(row)
    return rows


def _attempt_rows(result: object) -> list[dict[str, object]]:
    return [
        {
            "attempt": item.attempt,
            "start_parameter": item.start_load_factor,
            "target_parameter": item.target_load_factor,
            "step_size": item.step_size,
            "action": item.action,
            "inner_termination_reason": item.inner_termination_reason,
            "augmentations": item.augmentations,
            "newton_iterations": item.newton_iterations,
            "contact_event_restarts": item.contact_event_restarts,
            "equilibrium_residual": item.equilibrium_residual,
            "maximum_penetration": item.maximum_penetration,
            "effective_load_norm": item.effective_load_norm,
            "reaction_norm": item.reaction_norm,
            "penalties_before": item.penalties_before,
            "penalties_after": item.penalties_after,
            "prescribed_values": item.prescribed_values,
            "penalty_update_reasons": item.penalty_update_reasons,
        }
        for item in result.attempts
    ]


def collect_histories(result: object) -> OnsetHistories:
    """Collect deterministic solver, contact-row, attempt, and tangent histories."""

    step_rows, interface_rows = _step_histories(result)
    separated, contacting = _classified_steps(result)
    return OnsetHistories(
        step_rows,
        interface_rows,
        _tangent_rows(separated, contacting),
        _attempt_rows(result),
        separated,
        contacting,
    )
