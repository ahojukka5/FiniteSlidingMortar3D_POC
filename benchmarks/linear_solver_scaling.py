#!/usr/bin/env python3
"""Benchmark dense, sparse-direct, and Krylov Newton linear backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from svg_plots import write_line_chart

from contact3d import LinearSolverOptions, NewtonOptions, solve_coupled_equilibrium
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.verification_models import stacked_matching_block_contact_model

_BACKENDS = ("dense", "sparse_lu", "gmres_ilu", "bicgstab_ilu")


def _linear_options(name: str) -> LinearSolverOptions:
    common = {
        "relative_tolerance": 1.0e-11,
        "absolute_tolerance": 1.0e-13,
        "maximum_iterations": 1000,
        "restart": 100,
    }
    if name == "dense":
        return LinearSolverOptions(backend="dense", **common)
    if name == "sparse_lu":
        return LinearSolverOptions(backend="sparse_lu", **common)
    if name == "gmres_ilu":
        return LinearSolverOptions(
            backend="gmres",
            preconditioner="ilu",
            ilu_drop_tolerance=1.0e-5,
            ilu_fill_factor=20.0,
            **common,
        )
    if name == "bicgstab_ilu":
        return LinearSolverOptions(
            backend="bicgstab",
            preconditioner="ilu",
            ilu_drop_tolerance=1.0e-5,
            ilu_fill_factor=20.0,
            **common,
        )
    raise ValueError(f"unsupported benchmark backend: {name}")


def _newton_options(name: str) -> NewtonOptions:
    return NewtonOptions(
        maximum_iterations=40,
        absolute_tolerance=1.0e-9,
        relative_tolerance=1.0e-9,
        linear_solver=_linear_options(name),
    )


def _run_row(
    *,
    resolution: int,
    backend_name: str,
    model,
    result,
) -> dict[str, object]:
    diagnostics = [row.linear_solve for row in result.history]
    failure = result.linear_solve_failure
    selected_backends = sorted({item.backend for item in diagnostics})
    selected = ",".join(selected_backends)
    if not selected and failure is not None:
        selected = failure.backend
    final_pressure = max(
        (
            float(np.max(contact.pressure, initial=0.0))
            for contact in result.evaluation.contacts
        ),
        default=0.0,
    )
    active_rows = sum(
        int(np.count_nonzero(contact.signature.active_rows))
        for contact in result.evaluation.contacts
    )
    matrix_nnz = diagnostics[0].matrix_nnz if diagnostics else 0
    return {
        "resolution": resolution,
        "layers": model.layers,
        "backend": backend_name,
        "selected_backend": selected,
        "preconditioner": diagnostics[0].preconditioner if diagnostics else "none",
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "linear_failure_reason": failure.failure_reason if failure is not None else "",
        "node_count": model.problem.mesh.node_count,
        "element_count": model.problem.mesh.element_count,
        "interface_count": model.interface_count,
        "total_dofs": model.total_dofs,
        "free_dofs": len(model.free_dofs),
        "global_matrix_nnz": model.problem.sparsity.nnz,
        "reduced_matrix_nnz": matrix_nnz,
        "reduced_dense_entries": len(model.free_dofs) ** 2,
        "reduced_density": matrix_nnz / max(1, len(model.free_dofs) ** 2),
        "newton_iterations": result.iteration_count,
        "linear_iterations": sum(item.iterations for item in diagnostics),
        "linear_setup_seconds": sum(item.setup_seconds for item in diagnostics),
        "linear_solve_seconds": sum(item.solve_seconds for item in diagnostics),
        "maximum_linear_residual": max(
            (item.residual_norm for item in diagnostics),
            default=0.0,
        ),
        "maximum_relative_linear_residual": max(
            (item.relative_residual for item in diagnostics),
            default=0.0,
        ),
        "dense_materializations": sum(
            int(item.materialized_dense) for item in diagnostics
        ),
        "contact_event_restarts": result.contact_event_restarts,
        "final_equilibrium_residual": result.evaluation.free_residual_norm,
        "minimum_jacobian": result.evaluation.bulk.minimum_jacobian,
        "maximum_penetration": result.evaluation.maximum_penetration,
        "maximum_pressure": final_pressure,
        "active_rows": active_rows,
    }


def _iteration_rows(
    *,
    resolution: int,
    backend_name: str,
    result,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in result.history:
        linear = item.linear_solve
        rows.append(
            {
                "resolution": resolution,
                "backend": backend_name,
                "newton_iteration": item.iteration,
                "accepted": True,
                "selected_backend": linear.backend,
                "preconditioner": linear.preconditioner,
                "linear_converged": linear.converged,
                "linear_iterations": linear.iterations,
                "linear_residual_norm": linear.residual_norm,
                "linear_relative_residual": linear.relative_residual,
                "linear_setup_seconds": linear.setup_seconds,
                "linear_solve_seconds": linear.solve_seconds,
                "matrix_rows": linear.matrix_shape[0],
                "matrix_nnz": linear.matrix_nnz,
                "materialized_dense": linear.materialized_dense,
                "failure_reason": linear.failure_reason or "",
                "nonlinear_residual_norm": item.residual_norm,
                "relative_nonlinear_residual": item.relative_residual,
                "accepted_step": item.accepted_step,
                "line_search_iterations": item.line_search_iterations,
                "contact_branch_changed": item.contact_branch_changed,
            }
        )
    if result.linear_solve_failure is not None:
        linear = result.linear_solve_failure
        rows.append(
            {
                "resolution": resolution,
                "backend": backend_name,
                "newton_iteration": len(result.history) + 1,
                "accepted": False,
                "selected_backend": linear.backend,
                "preconditioner": linear.preconditioner,
                "linear_converged": linear.converged,
                "linear_iterations": linear.iterations,
                "linear_residual_norm": linear.residual_norm,
                "linear_relative_residual": linear.relative_residual,
                "linear_setup_seconds": linear.setup_seconds,
                "linear_solve_seconds": linear.solve_seconds,
                "matrix_rows": linear.matrix_shape[0],
                "matrix_nnz": linear.matrix_nnz,
                "materialized_dense": linear.materialized_dense,
                "failure_reason": linear.failure_reason or "",
                "nonlinear_residual_norm": result.evaluation.free_residual_norm,
                "relative_nonlinear_residual": "",
                "accepted_step": "",
                "line_search_iterations": "",
                "contact_branch_changed": "",
            }
        )
    return rows


def _plot_scaling(
    output: Path,
    rows: list[dict[str, object]],
    backend_names: tuple[str, ...],
) -> None:
    ordered = {
        backend: sorted(
            (row for row in rows if row["backend"] == backend),
            key=lambda row: int(row["free_dofs"]),
        )
        for backend in backend_names
    }
    x = np.asarray(
        [float(row["free_dofs"]) for row in ordered[backend_names[0]]],
        dtype=float,
    )
    for backend in backend_names[1:]:
        candidate = np.asarray(
            [float(row["free_dofs"]) for row in ordered[backend]],
            dtype=float,
        )
        if not np.array_equal(candidate, x):
            raise RuntimeError("backend scaling rows do not share the same mesh levels")

    charts = (
        (
            "linear-solve-time.svg",
            "Linear solve time by backend",
            "total linear solve time [s]",
            "linear_solve_seconds",
            False,
        ),
        (
            "linear-setup-time.svg",
            "Linear setup time by backend",
            "total linear setup time [s]",
            "linear_setup_seconds",
            False,
        ),
        (
            "linear-iterations.svg",
            "Accumulated linear iterations",
            "linear iterations",
            "linear_iterations",
            False,
        ),
        (
            "backend-agreement.svg",
            "Displacement agreement with dense Newton",
            "relative displacement difference",
            "relative_displacement_difference",
            True,
        ),
    )
    for filename, title, y_label, key, logarithmic in charts:
        series = tuple(
            (
                np.asarray(
                    [
                        max(float(row[key]), np.finfo(float).tiny)
                        if logarithmic
                        else float(row[key])
                        for row in ordered[backend]
                    ],
                    dtype=float,
                ),
                backend,
            )
            for backend in backend_names
        )
        write_line_chart(
            output / filename,
            title=title,
            x_label="free degrees of freedom",
            y_label=y_label,
            x_values=x,
            series=series,
            logarithmic_y=logarithmic,
        )

    reference_rows = (
        ordered["sparse_lu"] if "sparse_lu" in ordered else ordered[backend_names[0]]
    )
    write_line_chart(
        output / "reduced-matrix-density.svg",
        title="Reduced tangent storage density",
        x_label="free degrees of freedom",
        y_label="CSR density",
        x_values=np.asarray(
            [float(row["free_dofs"]) for row in reference_rows],
            dtype=float,
        ),
        series=(
            (
                np.asarray(
                    [float(row["reduced_density"]) for row in reference_rows],
                    dtype=float,
                ),
                "free-free CSR density",
            ),
        ),
    )


def _plot_largest_histories(
    output: Path,
    results: dict[tuple[int, str], object],
    largest_level: int,
    backend_names: tuple[str, ...],
) -> None:
    for backend in backend_names:
        result = results[(largest_level, backend)]
        if not result.history:
            continue
        x = np.asarray([row.iteration for row in result.history], dtype=float)
        nonlinear = np.asarray(
            [max(row.residual_norm, np.finfo(float).tiny) for row in result.history],
            dtype=float,
        )
        linear = np.asarray(
            [
                max(row.linear_solve.relative_residual, np.finfo(float).tiny)
                for row in result.history
            ],
            dtype=float,
        )
        write_line_chart(
            output / f"largest-newton-history-{backend}.svg",
            title=f"Largest model convergence: {backend}",
            x_label="accepted Newton iteration",
            y_label="residual",
            x_values=x,
            series=((nonlinear, "nonlinear norm"), (linear, "linear relative")),
            logarithmic_y=True,
        )


def run(
    output: Path,
    *,
    levels: tuple[int, ...] = (2, 4, 6),
    backend_names: tuple[str, ...] = ("dense", "sparse_lu", "gmres_ilu"),
    layers: int = 2,
    indentation: float = 0.04,
    penalty: float = 6400.0,
    minimum_free_dofs: int = 500,
) -> dict[str, object]:
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("levels must contain positive resolutions")
    if len(set(levels)) != len(levels):
        raise ValueError("levels must be unique")
    if not backend_names or any(name not in _BACKENDS for name in backend_names):
        raise ValueError(f"backends must be selected from {_BACKENDS}")
    if "dense" not in backend_names or "sparse_lu" not in backend_names:
        raise ValueError("dense and sparse_lu are required benchmark oracles")
    if minimum_free_dofs < 0:
        raise ValueError("minimum_free_dofs must be nonnegative")

    output.mkdir(parents=True, exist_ok=True)
    ordered_levels = tuple(sorted(levels))
    settings = {
        "levels": ordered_levels,
        "layers": layers,
        "indentation": indentation,
        "penalty": penalty,
        "backends": backend_names,
        "minimum_free_dofs": minimum_free_dofs,
        "newton_options": {
            backend: _newton_options(backend) for backend in backend_names
        },
    }
    artifacts = BenchmarkArtifactWriter(
        output,
        "linear-solver-scaling",
        seed=0,
        solver_settings=settings,
        repo_root=Path(__file__).resolve().parents[1],
    )
    summary_rows: list[dict[str, object]] = []
    iteration_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    results: dict[tuple[int, str], object] = {}

    for resolution in ordered_levels:
        model = stacked_matching_block_contact_model(
            resolution,
            layers=layers,
            indentation=indentation,
            penalty=penalty,
        )
        problem = model.problem
        model_rows.append(
            {
                "resolution": resolution,
                "layers": layers,
                "node_count": problem.mesh.node_count,
                "element_count": problem.mesh.element_count,
                "interface_count": model.interface_count,
                "total_dofs": model.total_dofs,
                "free_dofs": len(model.free_dofs),
                "global_matrix_nnz": problem.sparsity.nnz,
                "global_dense_entries": model.total_dofs**2,
                "global_density": problem.sparsity.nnz / model.total_dofs**2,
                "reference_volume": problem.mesh.reference_volume,
            }
        )
        for backend in backend_names:
            result = solve_coupled_equilibrium(
                problem,
                problem.initial_states(),
                event_policy="restart",
                options=_newton_options(backend),
            )
            results[(resolution, backend)] = result
            row = _run_row(
                resolution=resolution,
                backend_name=backend,
                model=model,
                result=result,
            )
            summary_rows.append(row)
            iteration_rows.extend(
                _iteration_rows(
                    resolution=resolution,
                    backend_name=backend,
                    result=result,
                )
            )

        dense = results[(resolution, "dense")]
        if not dense.converged:
            raise RuntimeError(
                f"dense oracle failed at resolution {resolution}: "
                f"{dense.termination_reason}"
            )
        reference_norm = max(
            float(np.linalg.norm(dense.displacement)),
            np.finfo(float).tiny,
        )
        for backend in backend_names:
            result = results[(resolution, backend)]
            difference = float(np.linalg.norm(result.displacement - dense.displacement))
            relative = difference / reference_norm
            row = next(
                item
                for item in summary_rows
                if item["resolution"] == resolution and item["backend"] == backend
            )
            row["displacement_difference"] = difference
            row["relative_displacement_difference"] = relative

    all_converged = all(bool(row["converged"]) for row in summary_rows)
    direct_rows = [
        row for row in summary_rows if row["backend"] in ("dense", "sparse_lu")
    ]
    maximum_direct_difference = max(
        float(row["relative_displacement_difference"]) for row in direct_rows
    )
    sparse_rows = [row for row in summary_rows if row["backend"] != "dense"]
    sparse_without_dense = all(
        int(row["dense_materializations"]) == 0 for row in sparse_rows
    )
    largest_free_dofs = max(int(row["free_dofs"]) for row in summary_rows)
    acceptance = {
        "all_requested_backends_converged": all_converged,
        "direct_backend_relative_difference": maximum_direct_difference,
        "direct_backend_agreement_tolerance": 1.0e-9,
        "direct_backends_agree": maximum_direct_difference <= 1.0e-9,
        "sparse_backends_materialized_dense": not sparse_without_dense,
        "sparse_backends_remained_sparse": sparse_without_dense,
        "largest_free_dofs": largest_free_dofs,
        "minimum_free_dofs": minimum_free_dofs,
        "minimum_problem_exercised": largest_free_dofs >= minimum_free_dofs,
    }
    acceptance["passed"] = all(
        (
            acceptance["all_requested_backends_converged"],
            acceptance["direct_backends_agree"],
            acceptance["sparse_backends_remained_sparse"],
            acceptance["minimum_problem_exercised"],
        )
    )

    artifacts.write_csv(
        "models.csv",
        model_rows,
        schema="contact3d-linear-solver-models/v1",
    )
    artifacts.write_csv(
        "backend-summary.csv",
        summary_rows,
        schema="contact3d-linear-solver-runs/v1",
    )
    artifacts.write_csv(
        "linear-iterations.csv",
        iteration_rows,
        schema="contact3d-linear-solver-iterations/v1",
    )
    _plot_scaling(output, summary_rows, backend_names)
    _plot_largest_histories(output, results, max(ordered_levels), backend_names)
    for svg in output.glob("*.svg"):
        ElementTree.parse(svg)
        artifacts.register(svg.name, "svg")

    summary = {
        "schema_version": "contact3d-linear-solver-scaling/v1",
        "benchmark": "medium-coupled-linear-solver-scaling",
        "settings": {
            "levels": list(ordered_levels),
            "layers": layers,
            "indentation": indentation,
            "penalty": penalty,
            "backends": list(backend_names),
            "minimum_free_dofs": minimum_free_dofs,
        },
        "models": model_rows,
        "runs": summary_rows,
        "acceptance": acceptance,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-linear-solver-scaling/v1",
    )
    required = [
        "summary.json",
        "models.csv",
        "backend-summary.csv",
        "linear-iterations.csv",
        "linear-solve-time.svg",
        "linear-setup-time.svg",
        "linear-iterations.svg",
        "backend-agreement.svg",
        "reduced-matrix-density.svg",
    ]
    required.extend(
        f"largest-newton-history-{backend}.svg" for backend in backend_names
    )
    artifacts.finalize(required=required)
    if not acceptance["passed"]:
        raise RuntimeError(f"linear-solver benchmark acceptance failed: {acceptance}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/linear-solver-scaling"),
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--indentation", type=float, default=0.04)
    parser.add_argument("--penalty", type=float, default=6400.0)
    parser.add_argument("--minimum-free-dofs", type=int, default=500)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=_BACKENDS,
        default=["dense", "sparse_lu", "gmres_ilu"],
    )
    arguments = parser.parse_args()
    summary = run(
        arguments.output,
        levels=tuple(arguments.levels),
        backend_names=tuple(arguments.backends),
        layers=arguments.layers,
        indentation=arguments.indentation,
        penalty=arguments.penalty,
        minimum_free_dofs=arguments.minimum_free_dofs,
    )
    print(json.dumps(summary["acceptance"], indent=2))


if __name__ == "__main__":
    main()
