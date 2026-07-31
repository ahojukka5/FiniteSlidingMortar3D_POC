#!/usr/bin/env python3
"""Drive separated blocks through contact onset with a mixed boundary/load path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    AugmentedContactOptions,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    LinearBoundaryPath,
    LinearPathValue,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    solve_adaptive_contact_path,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.coupled_oracle import FrozenMatchingMortarInterface
from svg_plots import write_line_chart


def block_nodes(z_origin: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, z_origin],
            [1.0, 0.0, z_origin],
            [1.0, 1.0, z_origin],
            [0.0, 1.0, z_origin],
            [0.0, 0.0, z_origin + 1.0],
            [1.0, 0.0, z_origin + 1.0],
            [1.0, 1.0, z_origin + 1.0],
            [0.0, 1.0, z_origin + 1.0],
            [0.5, 0.5, z_origin + 0.5],
        ]
    )


def block_elements(offset: int) -> np.ndarray:
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    return np.asarray(
        [(offset + 8, offset + a, offset + b, offset + c) for a, b, c in triangles],
        dtype=np.int64,
    )


def model() -> tuple[CoupledEquilibriumProblem, LinearBoundaryPath]:
    separation = 0.05
    nodes = np.vstack([block_nodes(0.0), block_nodes(1.0 + separation)])
    elements = np.vstack([block_elements(0), block_elements(9)])
    mesh = Tet4Mesh(nodes, elements)
    interface = FrozenMatchingMortarInterface(
        np.array([9, 12, 11, 10], dtype=np.int64),
        np.array([4, 7, 6, 5], dtype=np.int64),
        np.array([0.0, 0.0, -1.0]),
        6400.0,
        initial_normal_gap=-separation,
    )

    constrained_dofs: list[int] = []
    final_values: list[float] = []
    for node in (0, 1, 2, 3):
        for component in range(3):
            constrained_dofs.append(3 * node + component)
            final_values.append(0.0)
    for node in (13, 14, 15, 16):
        for component, value in enumerate((0.02, 0.0, -0.12)):
            constrained_dofs.append(3 * node + component)
            final_values.append(value)

    constraints = DirichletConstraints(
        np.asarray(constrained_dofs, dtype=np.int64),
        np.asarray(final_values),
    )
    force = np.zeros(3 * mesh.node_count)
    force[3 * 17] = 2.0
    load = DeadLoad(force)
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    problem = CoupledEquilibriumProblem(mesh, material, constraints, load, (interface,))
    path = LinearBoundaryPath.proportional_mixed(
        problem,
        values=(
            LinearPathValue("tool_x", 0.0, 0.02),
            LinearPathValue("tool_z", 0.0, -0.12),
            LinearPathValue("dead_load_x", 0.0, 2.0),
        ),
    )
    return problem, path


def _accepted_step_rows(result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, step in enumerate(result.accepted_steps, start=1):
        contact = step.result.equilibrium.evaluation.contacts[0]
        reaction = step.reaction.reshape((-1, 3)).sum(axis=0)
        rows.append(
            {
                "accepted_step": index,
                "parameter": step.parameter,
                "tool_x": step.path_state.value("tool_x"),
                "tool_z": step.path_state.value("tool_z"),
                "dead_load_x": step.path_state.value("dead_load_x"),
                "effective_load_norm": step.path_state.effective_load_norm,
                "reaction_x": float(reaction[0]),
                "reaction_y": float(reaction[1]),
                "reaction_z": float(reaction[2]),
                "reaction_norm": step.reaction_norm,
                "maximum_penetration": contact.diagnostics.maximum_penetration,
                "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
                "supported_rows": int(
                    np.count_nonzero(contact.signature.supported_rows)
                ),
                "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
                "newton_iterations": sum(
                    equilibrium.iteration_count
                    for equilibrium in step.result.equilibria
                ),
                "augmentations": len(step.result.history),
                "contact_event_restarts": sum(
                    equilibrium.contact_event_restarts
                    for equilibrium in step.result.equilibria
                ),
            }
        )
    return rows


def _attempt_rows(result) -> list[dict[str, object]]:
    return [
        {
            "attempt": attempt.attempt,
            "start_parameter": attempt.start_load_factor,
            "target_parameter": attempt.target_load_factor,
            "step_size": attempt.step_size,
            "action": attempt.action,
            "inner_termination_reason": attempt.inner_termination_reason,
            "augmentations": attempt.augmentations,
            "newton_iterations": attempt.newton_iterations,
            "contact_event_restarts": attempt.contact_event_restarts,
            "equilibrium_residual": attempt.equilibrium_residual,
            "maximum_penetration": attempt.maximum_penetration,
            "effective_load_norm": attempt.effective_load_norm,
            "reaction_norm": attempt.reaction_norm,
            "penalties_before": attempt.penalties_before,
            "penalties_after": attempt.penalties_after,
            "prescribed_values": attempt.prescribed_values,
            "penalty_update_reasons": attempt.penalty_update_reasons,
        }
        for attempt in result.attempts
    ]


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    problem, path = model()
    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.25,
            minimum_step=1.0 / 1024.0,
            maximum_step=0.25,
            cutback_factor=0.5,
            growth_factor=1.5,
            easy_newton_iterations=8,
            maximum_attempts=100,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=4.0,
            maximum_penalty=1.0e7,
            maximum_updates_per_step=3,
        ),
        augmented=AugmentedContactOptions(
            maximum_augmentations=16,
            gap_tolerance=1.0e-8,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-5,
            multiplier_tolerance=1.0e-8,
            event_policy="restart",
            newton=NewtonOptions(
                maximum_iterations=40,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
            ),
        ),
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "mixed-contact-onset",
        seed=0,
        solver_settings={"path": path, "adaptive": options},
        repo_root=Path(__file__).resolve().parents[1],
    )
    result = solve_adaptive_contact_path(
        problem,
        1.0,
        path=path,
        options=options,
    )
    if not result.converged:
        raise RuntimeError(
            f"mixed contact-onset path failed: {result.termination_reason}"
        )

    step_rows = _accepted_step_rows(result)
    attempt_rows = _attempt_rows(result)
    onset = next(
        (float(row["parameter"]) for row in step_rows if int(row["active_rows"]) > 0),
        None,
    )
    metrics = {
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "accepted_steps": result.accepted_step_count,
        "cutbacks": result.cutback_count,
        "penalty_updates": result.penalty_update_count,
        "contact_onset_parameter": onset,
        "final_reaction_x": float(step_rows[-1]["reaction_x"]),
        "final_reaction_z": float(step_rows[-1]["reaction_z"]),
        "final_maximum_pressure": float(step_rows[-1]["maximum_pressure"]),
        "final_maximum_penetration": float(step_rows[-1]["maximum_penetration"]),
    }
    summary = {
        "schema_version": "contact3d-mixed-contact-onset/v1",
        "path": {
            "type": "linear mixed prescribed-displacement/dead-load",
            "initial_separation": 0.05,
            "final_tool_x": 0.02,
            "final_tool_z": -0.12,
            "final_dead_load_x": 2.0,
        },
        "metrics": metrics,
        "accepted_steps": step_rows,
        "attempts": attempt_rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-mixed-contact-onset/v1",
    )
    artifacts.write_csv(
        "accepted-steps.csv",
        step_rows,
        schema="contact3d-contact-path-steps/v1",
    )
    artifacts.write_csv(
        "attempt-history.csv",
        attempt_rows,
        schema="contact3d-continuation-attempts/v1",
    )

    final_step = result.accepted_steps[-1]
    final_problem = final_step.path_state.problem
    final_result = final_step.result
    final_evaluation = final_result.equilibrium.evaluation
    contact = final_evaluation.contacts[0]
    interface = final_problem.interfaces[0]
    total_dofs = 3 * final_problem.mesh.node_count
    contact_force = np.zeros(total_dofs)
    np.add.at(contact_force, interface.dofs, contact.residual)
    body_id = np.concatenate(
        [
            np.zeros(len(block_elements(0)), dtype=np.int64),
            np.ones(len(block_elements(9)), dtype=np.int64),
        ]
    )
    artifacts.write_tet4_vtu(
        "deformed.vtu",
        final_problem.mesh.reference_nodes,
        final_problem.mesh.elements,
        final_result.displacement,
        point_data={
            "reaction": final_step.reaction.reshape((-1, 3)),
            "contact_force": contact_force.reshape((-1, 3)),
            "effective_load": final_step.path_state.effective_force.reshape((-1, 3)),
        },
        cell_data={
            "body_id": body_id,
            "jacobian": np.asarray(
                [item.jacobian for item in final_evaluation.bulk.element_evaluations]
            ),
            "energy_density": np.asarray(
                [
                    item.energy_density
                    for item in final_evaluation.bulk.element_evaluations
                ]
            ),
        },
    )

    local_force = contact.residual.reshape((-1, 3))
    displacement = final_result.displacement.reshape((-1, 3))
    slave_ids = np.asarray(interface.slave_nodes, dtype=np.int64)
    master_ids = np.asarray(interface.master_nodes, dtype=np.int64)
    facet = (np.arange(4, dtype=np.int64),)
    normal = np.repeat(interface.normal[None, :], 4, axis=0)
    artifacts.write_surface_vtp(
        "slave-contact.vtp",
        final_problem.mesh.reference_nodes[slave_ids],
        facet,
        displacement[slave_ids],
        point_data={
            "normal": normal,
            "normal_gap": contact.normal_gaps,
            "pressure": contact.pressure,
            "multiplier": final_result.states[0].multipliers,
            "supported": np.asarray(contact.signature.supported_rows, dtype=np.int64),
            "active": np.asarray(contact.signature.active_rows, dtype=np.int64),
            "contact_force": local_force[:4],
        },
        cell_data={"interface_area": np.asarray([interface.area])},
    )
    artifacts.write_surface_vtp(
        "master-contact.vtp",
        final_problem.mesh.reference_nodes[master_ids],
        facet,
        displacement[master_ids],
        point_data={
            "normal": normal,
            "contact_force": local_force[4:],
        },
        cell_data={"interface_area": np.asarray([interface.area])},
    )

    parameter = np.asarray([float(row["parameter"]) for row in step_rows])
    write_line_chart(
        output / "reaction-path.svg",
        title="Mixed-path constrained reactions",
        x_label="continuation parameter",
        y_label="summed constrained reaction",
        x_values=parameter,
        series=(
            (np.asarray([float(row["reaction_x"]) for row in step_rows]), "reaction x"),
            (np.asarray([float(row["reaction_z"]) for row in step_rows]), "reaction z"),
        ),
    )
    write_line_chart(
        output / "contact-onset.svg",
        title="Contact onset and pressure growth",
        x_label="continuation parameter",
        y_label="contact measure",
        x_values=parameter,
        series=(
            (
                np.asarray([float(row["maximum_pressure"]) for row in step_rows]),
                "maximum pressure",
            ),
            (
                np.asarray([float(row["active_rows"]) for row in step_rows]),
                "active rows",
            ),
        ),
    )
    for path_name in output.glob("*.svg"):
        ElementTree.parse(path_name)
        artifacts.register(path_name.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "accepted-steps.csv",
            "attempt-history.csv",
            "deformed.vtu",
            "slave-contact.vtp",
            "master-contact.vtp",
            "reaction-path.svg",
            "contact-onset.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mixed-contact-onset"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
