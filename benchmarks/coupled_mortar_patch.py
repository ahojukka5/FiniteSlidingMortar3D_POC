#!/usr/bin/env python3
"""Generate coupled bulk/contact Newton and augmented-Lagrange artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import (
    AugmentedContactOptions,
    AugmentedContactResult,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    solve_augmented_contact,
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
    surface_triangles = [
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
        [(offset + 8, offset + a, offset + b, offset + c) for a, b, c in surface_triangles],
        dtype=np.int64,
    )


def problem(*, penalty: float = 6400.0) -> CoupledEquilibriumProblem:
    nodes = np.vstack([block_nodes(0.0), block_nodes(1.0)])
    elements = np.vstack([block_elements(0), block_elements(9)])
    mesh = Tet4Mesh(nodes, elements)
    interface = FrozenMatchingMortarInterface(
        np.array([9, 12, 11, 10], dtype=np.int64),
        np.array([4, 7, 6, 5], dtype=np.int64),
        np.array([0.0, 0.0, -1.0]),
        penalty,
    )

    constrained_dofs: list[int] = []
    prescribed_values: list[float] = []
    for node in (0, 1, 2, 3):
        for component in range(3):
            constrained_dofs.append(3 * node + component)
            prescribed_values.append(0.0)
    for node in (13, 14, 15, 16):
        for component in range(3):
            constrained_dofs.append(3 * node + component)
            prescribed_values.append(-0.12 if component == 2 else 0.0)

    constraints = DirichletConstraints(
        np.asarray(constrained_dofs, dtype=np.int64),
        np.asarray(prescribed_values),
    )
    load = DeadLoad(np.zeros(3 * mesh.node_count))
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    return CoupledEquilibriumProblem(mesh, material, constraints, load, (interface,))


def write_interface_plot(
    path: Path,
    gaps: np.ndarray,
    pressures: np.ndarray,
) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(np.max(pressures, initial=0.0)), 1.0)
    bar_width = plot_width / (2.0 * len(pressures))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Final interface pressure</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for node, (gap, pressure) in enumerate(zip(gaps, pressures, strict=True)):
        center = left + (node + 0.5) * plot_width / len(pressures)
        bar_height = pressure / maximum * plot_height
        x = center - bar_width / 2.0
        y = height - bottom - bar_height
        lines.extend(
            [
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{bar_height:.2f}" fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height-bottom+22}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12">A{node}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{max(top+14, y-6):.2f}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10">'
                    f"p={pressure:.4g}, g={gap:.3g}</text>"
                ),
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def solve_directional_residual(
    coupled_problem: CoupledEquilibriumProblem,
    result: AugmentedContactResult,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    from contact3d import evaluate_coupled_equilibrium

    return evaluate_coupled_equilibrium(
        coupled_problem,
        result.displacement + step * direction,
        result.states,
        assemble_tangent=False,
    ).residual


def _augmentation_rows(result: AugmentedContactResult) -> list[dict[str, object]]:
    return [
        {
            "augmentation": row.augmentation,
            "newton_iterations": row.newton_iterations,
            "contact_event_restarts": row.contact_event_restarts,
            "equilibrium_residual": row.equilibrium_residual,
            "maximum_penetration": row.maximum_penetration,
            "maximum_complementarity": row.maximum_complementarity,
            "maximum_projection_residual": row.maximum_projection_residual,
            "maximum_multiplier_increment": row.maximum_multiplier_increment,
            "active_rows": row.active_rows,
            "maximum_pressure": row.maximum_pressure,
        }
        for row in result.history
    ]


def _newton_rows(result: AugmentedContactResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    global_iteration = 0
    for augmentation, equilibrium in enumerate(result.equilibria, start=1):
        for row in equilibrium.history:
            global_iteration += 1
            linear = row.linear_solve
            rows.append(
                {
                    "augmentation": augmentation,
                    "global_iteration": global_iteration,
                    "iteration": row.iteration,
                    "residual_norm": row.residual_norm,
                    "relative_residual": row.relative_residual,
                    "bulk_potential": row.bulk_potential,
                    "minimum_jacobian": row.minimum_jacobian,
                    "maximum_penetration": row.maximum_penetration,
                    "step_norm": row.step_norm,
                    "accepted_step": row.accepted_step,
                    "line_search_iterations": row.line_search_iterations,
                    "contact_branch_changed": row.contact_branch_changed,
                    "linear_requested_backend": linear.requested_backend,
                    "linear_backend": linear.backend,
                    "linear_preconditioner": linear.preconditioner,
                    "linear_converged": linear.converged,
                    "linear_iterations": linear.iterations,
                    "linear_residual_norm": linear.residual_norm,
                    "linear_relative_residual": linear.relative_residual,
                    "linear_residual_history": linear.residual_history,
                    "linear_setup_seconds": linear.setup_seconds,
                    "linear_solve_seconds": linear.solve_seconds,
                    "linear_matrix_rows": linear.matrix_shape[0],
                    "linear_matrix_columns": linear.matrix_shape[1],
                    "linear_matrix_nnz": linear.matrix_nnz,
                    "linear_materialized_dense": linear.materialized_dense,
                    "linear_failure_reason": linear.failure_reason,
                }
            )
    return rows


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    coupled_problem = problem()
    options = AugmentedContactOptions(
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
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "coupled-mortar-patch",
        seed=8841,
        solver_settings=options,
        repo_root=Path(__file__).resolve().parents[1],
    )
    result = solve_augmented_contact(coupled_problem, options=options)
    if not result.converged:
        raise RuntimeError(
            f"coupled mortar patch failed: {result.termination_reason}"
        )

    augmentation_rows = _augmentation_rows(result)
    newton_rows = _newton_rows(result)
    contact = result.equilibrium.evaluation.contacts[0]
    final_evaluation = result.equilibrium.evaluation
    interface = coupled_problem.interfaces[0]

    rng = np.random.default_rng(8841)
    direction = np.zeros_like(result.displacement)
    direction[final_evaluation.free_dofs] = rng.normal(
        size=len(final_evaluation.free_dofs)
    )
    direction /= np.linalg.norm(direction)
    assert final_evaluation.tangent is not None
    exact_directional_tangent = final_evaluation.tangent.to_dense() @ direction
    tangent_step = 2.0e-7
    plus = solve_directional_residual(
        coupled_problem,
        result,
        direction,
        tangent_step,
    )
    minus = solve_directional_residual(
        coupled_problem,
        result,
        direction,
        -tangent_step,
    )
    numerical_directional_tangent = (plus - minus) / (2.0 * tangent_step)
    free = final_evaluation.free_dofs
    tangent_error = float(
        np.linalg.norm(
            exact_directional_tangent[free] - numerical_directional_tangent[free]
        )
        / np.linalg.norm(numerical_directional_tangent[free])
    )

    local_contact_force = contact.residual.reshape((-1, 3))
    interface_rows = [
        {
            "node": node,
            "global_node": int(interface.slave_nodes[node]),
            "normal_gap": float(contact.normal_gaps[node]),
            "pressure": float(contact.pressure[node]),
            "multiplier": float(result.states[0].multipliers[node]),
            "supported": bool(contact.signature.supported_rows[node]),
            "active": bool(contact.signature.active_rows[node]),
            "contact_force_x": float(local_contact_force[node, 0]),
            "contact_force_y": float(local_contact_force[node, 1]),
            "contact_force_z": float(local_contact_force[node, 2]),
        }
        for node in range(len(contact.normal_gaps))
    ]
    metrics = {
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "augmentations": len(result.history),
        "total_newton_iterations": sum(row.newton_iterations for row in result.history),
        "final_residual": final_evaluation.free_residual_norm,
        "maximum_penetration": final_evaluation.maximum_penetration,
        "contact_event_restarts": sum(
            row.contact_event_restarts for row in result.history
        ),
        "minimum_jacobian": final_evaluation.bulk.minimum_jacobian,
        "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
        "interface_mean_gap": float(np.mean(contact.normal_gaps)),
        "directional_tangent_error": tangent_error,
        "global_force_balance_norm": float(
            np.linalg.norm(final_evaluation.residual.reshape((-1, 3)).sum(axis=0))
        ),
        "contact_force_balance_norm": float(
            np.linalg.norm(contact.residual.reshape((-1, 3)).sum(axis=0))
        ),
    }
    summary = {
        "schema_version": "contact3d-coupled-mortar-patch/v1",
        "mesh": {
            "node_count": coupled_problem.mesh.node_count,
            "element_count": coupled_problem.mesh.element_count,
            "total_dofs": 3 * coupled_problem.mesh.node_count,
            "free_dofs": len(final_evaluation.free_dofs),
            "sparse_nnz": coupled_problem.sparsity.nnz,
        },
        "contact": {
            "penalty": interface.penalty,
            "area": interface.area,
            "normal": interface.normal.tolist(),
            "prescribed_compression": 0.12,
            "operator": "frozen matching Q1 standard-mortar D=M",
        },
        "metrics": metrics,
        "augmentation_history": augmentation_rows,
        "interface_state": interface_rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-coupled-mortar-patch/v1",
    )
    artifacts.write_csv(
        "augmentation-history.csv",
        augmentation_rows,
        schema="contact3d-augmentation-iterations/v1",
    )
    artifacts.write_csv(
        "newton-history.csv",
        newton_rows,
        schema="contact3d-coupled-newton-iterations/v1",
    )
    artifacts.write_csv(
        "interface-state.csv",
        interface_rows,
        schema="contact3d-contact-nodes/v1",
    )

    total_dofs = 3 * coupled_problem.mesh.node_count
    reaction = np.zeros(total_dofs)
    constrained = np.ones(total_dofs, dtype=bool)
    constrained[final_evaluation.free_dofs] = False
    reaction[constrained] = final_evaluation.residual[constrained]
    global_contact_force = np.zeros(total_dofs)
    np.add.at(global_contact_force, interface.dofs, contact.residual)
    body_id = np.concatenate(
        [
            np.zeros(len(block_elements(0)), dtype=np.int64),
            np.ones(len(block_elements(9)), dtype=np.int64),
        ]
    )
    artifacts.write_tet4_vtu(
        "deformed.vtu",
        coupled_problem.mesh.reference_nodes,
        coupled_problem.mesh.elements,
        result.displacement,
        point_data={
            "reaction": reaction.reshape((-1, 3)),
            "contact_force": global_contact_force.reshape((-1, 3)),
            "external_load": coupled_problem.load.force.reshape((-1, 3)),
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

    displacement_nodes = result.displacement.reshape((-1, 3))
    slave_ids = np.asarray(interface.slave_nodes, dtype=np.int64)
    master_ids = np.asarray(interface.master_nodes, dtype=np.int64)
    facet = (np.arange(4, dtype=np.int64),)
    repeated_normal = np.repeat(interface.normal[None, :], 4, axis=0)
    artifacts.write_surface_vtp(
        "slave-contact.vtp",
        coupled_problem.mesh.reference_nodes[slave_ids],
        facet,
        displacement_nodes[slave_ids],
        point_data={
            "normal": repeated_normal,
            "normal_gap": contact.normal_gaps,
            "pressure": contact.pressure,
            "multiplier": result.states[0].multipliers,
            "supported": np.asarray(contact.signature.supported_rows, dtype=np.int64),
            "active": np.asarray(contact.signature.active_rows, dtype=np.int64),
            "contact_force": local_contact_force[:4],
        },
        cell_data={"interface_area": np.asarray([interface.area])},
    )
    artifacts.write_surface_vtp(
        "master-contact.vtp",
        coupled_problem.mesh.reference_nodes[master_ids],
        facet,
        displacement_nodes[master_ids],
        point_data={
            "normal": repeated_normal,
            "contact_force": local_contact_force[4:],
        },
        cell_data={"interface_area": np.asarray([interface.area])},
    )

    augmentation_index = np.arange(len(augmentation_rows), dtype=float)
    write_line_chart(
        output / "kkt-convergence.svg",
        title="Augmented-Lagrange KKT convergence",
        x_label="augmentation",
        y_label="residual magnitude",
        x_values=augmentation_index,
        series=(
            (
                np.asarray(
                    [row["maximum_penetration"] for row in augmentation_rows]
                ),
                "penetration",
            ),
            (
                np.asarray(
                    [
                        row["maximum_projection_residual"]
                        for row in augmentation_rows
                    ]
                ),
                "projection",
            ),
        ),
        logarithmic_y=True,
    )
    write_line_chart(
        output / "newton-residual.svg",
        title="Coupled Newton convergence across augmentations",
        x_label="accepted Newton iteration",
        y_label="free residual norm",
        x_values=np.arange(1, len(newton_rows) + 1, dtype=float),
        series=(
            (
                np.asarray([row["residual_norm"] for row in newton_rows]),
                "residual",
            ),
        ),
        logarithmic_y=True,
    )
    write_interface_plot(
        output / "interface-pressure.svg",
        contact.normal_gaps,
        contact.pressure,
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "augmentation-history.csv",
            "newton-history.csv",
            "interface-state.csv",
            "deformed.vtu",
            "slave-contact.vtp",
            "master-contact.vtp",
            "kkt-convergence.svg",
            "newton-residual.svg",
            "interface-pressure.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/coupled-mortar-patch"),
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
