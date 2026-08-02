#!/usr/bin/env python3
"""Generate deterministic sparse-Newton equilibrium benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from svg_plots import write_line_chart, write_sparsity

from contact3d import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumProblem,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    evaluate_tet4_mesh,
    solve_equilibrium,
    solve_load_steps,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter


def cube_star_mesh() -> Tet4Mesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
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
    elements = np.array([(8, *triangle) for triangle in triangles], dtype=np.int64)
    return Tet4Mesh(nodes, elements)


def manufactured_problem() -> tuple[EquilibriumProblem, np.ndarray, np.ndarray]:
    mesh = cube_star_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    deformation = np.array(
        [[1.0, 0.0, 0.35], [0.0, 1.0, 0.06], [0.0, 0.0, 0.78]]
    )
    target = mesh.reference_nodes @ deformation.T - mesh.reference_nodes
    fixed_nodes = np.flatnonzero(mesh.reference_nodes[:, 2] == 0.0)
    constraints = DirichletConstraints.fixed_nodes(fixed_nodes)
    target_evaluation = evaluate_tet4_mesh(mesh, target, material)
    force = target_evaluation.residual.ravel().copy()
    force[constraints.dofs] = 0.0
    problem = EquilibriumProblem(mesh, material, constraints, DeadLoad(force))
    return problem, target, deformation


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    problem, target, deformation = manufactured_problem()
    options = NewtonOptions(
        maximum_iterations=30,
        absolute_tolerance=1.0e-11,
        relative_tolerance=1.0e-11,
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "nonlinear-equilibrium",
        seed=0,
        solver_settings=options,
        repo_root=Path(__file__).resolve().parents[1],
    )
    free_dofs = problem.constraints.free_dofs(3 * problem.mesh.node_count)
    initial_norm = float(np.linalg.norm(problem.load.force[free_dofs]))
    direct = solve_equilibrium(problem, options=options)
    if not direct.converged:
        raise RuntimeError(f"direct Newton solve failed: {direct.termination_reason}")
    factors = np.linspace(0.0, 1.0, 11)
    step_results = solve_load_steps(problem, factors, options=options)
    if len(step_results) != len(factors) or not all(item.converged for item in step_results):
        raise RuntimeError("load-step benchmark did not converge")

    history_rows = [
        {
            "iteration": item.iteration,
            "initial_residual_norm": initial_norm,
            "residual_norm": item.residual_norm,
            "relative_residual": item.relative_residual,
            "potential": item.potential,
            "minimum_jacobian": item.minimum_jacobian,
            "step_norm": item.step_norm,
            "accepted_step": item.accepted_step,
            "line_search_iterations": item.line_search_iterations,
            "linear_backend": item.linear_solve.backend,
            "linear_preconditioner": item.linear_solve.preconditioner,
            "linear_iterations": item.linear_solve.iterations,
            "linear_residual_norm": item.linear_solve.residual_norm,
            "linear_relative_residual": item.linear_solve.relative_residual,
            "linear_setup_seconds": item.linear_solve.setup_seconds,
            "linear_solve_seconds": item.linear_solve.solve_seconds,
            "linear_matrix_nnz": item.linear_solve.matrix_nnz,
            "linear_materialized_dense": item.linear_solve.materialized_dense,
        }
        for item in direct.history
    ]
    top_nodes = np.flatnonzero(problem.mesh.reference_nodes[:, 2] == 1.0)
    step_rows = []
    for result in step_results:
        displacement = result.displacement.reshape((-1, 3))
        step_rows.append(
            {
                "load_factor": result.load_factor,
                "iterations": result.iteration_count,
                "residual_norm": result.evaluation.free_residual_norm,
                "minimum_jacobian": result.evaluation.bulk.minimum_jacobian,
                "minimum_accepted_step": min(
                    (item.accepted_step for item in result.history),
                    default=1.0,
                ),
                "top_mean_x": float(np.mean(displacement[top_nodes, 0])),
                "top_mean_z": float(np.mean(displacement[top_nodes, 2])),
            }
        )

    free = direct.evaluation.free_dofs
    reaction = np.sum(direct.evaluation.reaction.reshape((-1, 3)), axis=0)
    external = np.sum(problem.load.force.reshape((-1, 3)), axis=0)
    summary = {
        "schema_version": "contact3d-nonlinear-equilibrium/v1",
        "mesh": {
            "node_count": problem.mesh.node_count,
            "element_count": problem.mesh.element_count,
            "total_dofs": 3 * problem.mesh.node_count,
            "free_dofs": len(free),
            "sparse_nnz": problem.sparsity.nnz,
            "dense_entries": (3 * problem.mesh.node_count) ** 2,
            "sparse_density": problem.sparsity.nnz / (3 * problem.mesh.node_count) ** 2,
        },
        "material": {
            "shear_modulus": problem.material.shear_modulus,
            "bulk_modulus": problem.material.bulk_modulus,
            "lame_lambda": problem.material.lame_lambda,
        },
        "target_deformation_gradient": deformation.tolist(),
        "direct_newton": {
            "converged": direct.converged,
            "termination_reason": direct.termination_reason,
            "iterations": direct.iteration_count,
            "initial_residual_norm": initial_norm,
            "final_residual_norm": direct.evaluation.free_residual_norm,
            "minimum_jacobian": direct.evaluation.bulk.minimum_jacobian,
            "minimum_accepted_step": min(item.accepted_step for item in direct.history),
            "displacement_error": float(
                np.linalg.norm(direct.displacement[free] - target.ravel()[free])
            ),
            "reaction_balance_norm": float(np.linalg.norm(reaction + external)),
        },
        "load_steps": step_rows,
        "newton_history": history_rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-nonlinear-equilibrium/v1",
    )
    artifacts.write_csv(
        "newton-history.csv",
        history_rows,
        schema="contact3d-newton-iterations/v1",
    )
    artifacts.write_csv(
        "load-steps.csv",
        step_rows,
        schema="contact3d-load-steps/v1",
    )
    artifacts.write_tet4_vtu(
        "deformed.vtu",
        problem.mesh.reference_nodes,
        problem.mesh.elements,
        direct.displacement,
        point_data={
            "reaction": direct.evaluation.reaction.reshape((-1, 3)),
            "external_load": problem.load.force.reshape((-1, 3)),
        },
        cell_data={
            "jacobian": np.array(
                [item.jacobian for item in direct.evaluation.bulk.element_evaluations]
            ),
            "energy_density": np.array(
                [item.energy_density for item in direct.evaluation.bulk.element_evaluations]
            ),
        },
    )
    residual_x = np.array([0.0, *[float(row["iteration"]) for row in history_rows]])
    residual_y = np.array(
        [initial_norm, *[float(row["residual_norm"]) for row in history_rows]]
    )
    write_line_chart(
        output / "newton-residual.svg",
        title="Newton residual convergence",
        x_label="accepted Newton iteration",
        y_label="free residual norm",
        x_values=residual_x,
        series=((residual_y, "residual"),),
        logarithmic_y=True,
    )
    step_x = np.array([float(row["load_factor"]) for row in step_rows])
    write_line_chart(
        output / "load-displacement.svg",
        title="Load-step displacement path",
        x_label="load factor",
        y_label="top-face mean displacement",
        x_values=step_x,
        series=(
            (np.array([float(row["top_mean_x"]) for row in step_rows]), "solid: x"),
            (np.array([float(row["top_mean_z"]) for row in step_rows]), "dashed: z"),
        ),
    )
    write_sparsity(problem, output / "sparsity-pattern.svg")
    for svg in output.glob("*.svg"):
        ElementTree.parse(svg)
        artifacts.register(svg.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "newton-history.csv",
            "load-steps.csv",
            "deformed.vtu",
            "newton-residual.svg",
            "load-displacement.svg",
            "sparsity-pattern.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nonlinear-equilibrium"),
    )
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps(result["direct_newton"], indent=2))


if __name__ == "__main__":
    main()
