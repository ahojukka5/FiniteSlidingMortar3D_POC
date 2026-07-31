"""Boundary conditions, dead loads, and Newton equilibrium solution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .bulk_material import BulkGeometryError, NeoHookeanMaterial
from .bulk_sparse import Tet4SparseEvaluation, Tet4Sparsity, assemble_tet4_sparse
from .linear_solver import (
    LinearSolveDiagnostics,
    LinearSolverOptions,
    solve_reduced_system,
)
from .model import FloatArray, IntArray
from .tet4 import Tet4Mesh


@dataclass(frozen=True, slots=True)
class DirichletConstraints:
    """Prescribed displacement values on unique global DOFs."""

    dofs: IntArray
    values: FloatArray

    def __post_init__(self) -> None:
        dofs = np.asarray(self.dofs, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        if dofs.ndim != 1 or values.shape != dofs.shape:
            raise ValueError("Dirichlet dofs and values must be one-dimensional and aligned")
        if np.any(dofs < 0):
            raise ValueError("Dirichlet dofs must be nonnegative")
        if len(np.unique(dofs)) != len(dofs):
            raise ValueError("Dirichlet dofs must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("Dirichlet values must be finite")
        order = np.argsort(dofs)
        object.__setattr__(self, "dofs", dofs[order].copy())
        object.__setattr__(self, "values", values[order].copy())

    @classmethod
    def fixed_nodes(cls, nodes: IntArray) -> DirichletConstraints:
        node_indices = np.asarray(nodes, dtype=np.int64)
        dofs = np.asarray(
            [3 * int(node) + component for node in node_indices for component in range(3)],
            dtype=np.int64,
        )
        return cls(dofs, np.zeros(len(dofs), dtype=float))

    def validate_for(self, total_dofs: int) -> None:
        if total_dofs < 0:
            raise ValueError("total_dofs must be nonnegative")
        if np.any(self.dofs >= total_dofs):
            raise ValueError("Dirichlet dof is out of range")

    def free_dofs(self, total_dofs: int) -> IntArray:
        self.validate_for(total_dofs)
        mask = np.ones(total_dofs, dtype=bool)
        mask[self.dofs] = False
        return np.flatnonzero(mask).astype(np.int64)

    def apply(self, displacement: FloatArray) -> FloatArray:
        values = np.asarray(displacement, dtype=float).copy().reshape(-1)
        self.validate_for(len(values))
        values[self.dofs] = self.values
        return values


@dataclass(frozen=True, slots=True)
class DeadLoad:
    """Configuration-independent nodal force vector."""

    force: FloatArray

    def __post_init__(self) -> None:
        force = np.asarray(self.force, dtype=float)
        if force.ndim != 1:
            raise ValueError("dead-load force must be a flat global vector")
        if not np.all(np.isfinite(force)):
            raise ValueError("dead-load force must be finite")
        object.__setattr__(self, "force", force.copy())

    @classmethod
    def from_nodal_forces(cls, node_forces: FloatArray) -> DeadLoad:
        values = np.asarray(node_forces, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("node_forces must have shape (node_count, 3)")
        return cls(values.ravel())


@dataclass(frozen=True, slots=True)
class EquilibriumProblem:
    """Finite-strain bulk problem with dead loads and essential constraints."""

    mesh: Tet4Mesh
    material: NeoHookeanMaterial
    constraints: DirichletConstraints
    load: DeadLoad
    sparsity: Tet4Sparsity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        total_dofs = 3 * self.mesh.node_count
        self.constraints.validate_for(total_dofs)
        if self.load.force.shape != (total_dofs,):
            raise ValueError("dead-load vector must match the mesh DOF count")
        object.__setattr__(self, "sparsity", Tet4Sparsity.from_mesh(self.mesh))


@dataclass(frozen=True, slots=True)
class EquilibriumEvaluation:
    """Bulk equilibrium residual and tangent at one feasible displacement."""

    displacement: FloatArray
    load_factor: float
    potential: float
    residual: FloatArray
    free_dofs: IntArray
    free_residual_norm: float
    bulk: Tet4SparseEvaluation

    @property
    def reaction(self) -> FloatArray:
        reaction = np.zeros_like(self.residual)
        constrained = np.ones(len(self.residual), dtype=bool)
        constrained[self.free_dofs] = False
        reaction[constrained] = self.residual[constrained]
        return reaction


@dataclass(frozen=True, slots=True)
class NewtonOptions:
    maximum_iterations: int = 30
    absolute_tolerance: float = 1.0e-10
    relative_tolerance: float = 1.0e-10
    armijo_coefficient: float = 1.0e-4
    line_search_reduction: float = 0.5
    minimum_step: float = 2.0**-20
    maximum_line_search_iterations: int = 24
    linear_solver: LinearSolverOptions = field(default_factory=LinearSolverOptions)

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.maximum_line_search_iterations <= 0:
            raise ValueError("maximum_line_search_iterations must be positive")
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
            ("armijo_coefficient", self.armijo_coefficient),
            ("minimum_step", self.minimum_step),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.armijo_coefficient < 1.0:
            raise ValueError("armijo_coefficient must lie between zero and one")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must lie between zero and one")


@dataclass(frozen=True, slots=True)
class NewtonIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    potential: float
    minimum_jacobian: float
    step_norm: float
    accepted_step: float
    line_search_iterations: int
    linear_solve: LinearSolveDiagnostics


TerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "line_search_failed",
    "singular_tangent",
    "linear_solve_failed",
]


@dataclass(frozen=True, slots=True)
class NewtonResult:
    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: TerminationReason
    evaluation: EquilibriumEvaluation
    history: tuple[NewtonIteration, ...]
    linear_solve_failure: LinearSolveDiagnostics | None = None

    @property
    def iteration_count(self) -> int:
        return len(self.history)


def evaluate_equilibrium(
    problem: EquilibriumProblem,
    displacement: FloatArray,
    *,
    load_factor: float = 1.0,
    tolerance: float = 1.0e-12,
) -> EquilibriumEvaluation:
    """Evaluate total potential and residual after enforcing essential constraints."""

    if not np.isfinite(load_factor) or load_factor < 0.0:
        raise ValueError("load_factor must be finite and nonnegative")
    total_dofs = 3 * problem.mesh.node_count
    values = np.asarray(displacement, dtype=float)
    if values.shape == (problem.mesh.node_count, 3):
        values = values.ravel()
    if values.shape != (total_dofs,):
        raise ValueError("displacement must match the mesh DOF count")
    if not np.all(np.isfinite(values)):
        raise ValueError("displacement must be finite")
    feasible = problem.constraints.apply(values)
    bulk = assemble_tet4_sparse(
        problem.mesh,
        feasible,
        problem.material,
        sparsity=problem.sparsity,
        tolerance=tolerance,
    )
    external = load_factor * problem.load.force
    residual = bulk.residual.ravel() - external
    free = problem.constraints.free_dofs(total_dofs)
    potential = bulk.energy - float(np.dot(external, feasible))
    return EquilibriumEvaluation(
        displacement=feasible,
        load_factor=float(load_factor),
        potential=float(potential),
        residual=residual,
        free_dofs=free,
        free_residual_norm=float(np.linalg.norm(residual[free])),
        bulk=bulk,
    )


def _relative_residual(norm: float, initial_norm: float) -> float:
    return norm / max(initial_norm, np.finfo(float).tiny)


def _linear_failure_reason(
    diagnostics: LinearSolveDiagnostics,
) -> TerminationReason:
    if diagnostics.failure_reason in {"singular_matrix", "factorization_failed"}:
        return "singular_tangent"
    return "linear_solve_failed"


def solve_equilibrium(
    problem: EquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    *,
    load_factor: float = 1.0,
    options: NewtonOptions | None = None,
    tolerance: float = 1.0e-12,
) -> NewtonResult:
    """Solve one load level with Newton iterations and an Armijo residual line search."""

    settings = NewtonOptions() if options is None else options
    total_dofs = 3 * problem.mesh.node_count
    displacement = (
        np.zeros(total_dofs, dtype=float)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).reshape(-1).copy()
    )
    if displacement.shape != (total_dofs,):
        raise ValueError("initial_displacement must match the mesh DOF count")
    displacement = problem.constraints.apply(displacement)
    evaluation = evaluate_equilibrium(
        problem,
        displacement,
        load_factor=load_factor,
        tolerance=tolerance,
    )
    initial_norm = evaluation.free_residual_norm
    threshold = max(
        settings.absolute_tolerance,
        settings.relative_tolerance * initial_norm,
    )
    history: list[NewtonIteration] = []
    if evaluation.free_residual_norm <= threshold:
        return NewtonResult(displacement, load_factor, True, "converged", evaluation, ())

    for iteration in range(settings.maximum_iterations):
        free = evaluation.free_dofs
        linear_result = solve_reduced_system(
            evaluation.bulk.tangent,
            free,
            -evaluation.residual[free],
            options=settings.linear_solver,
        )
        if linear_result.solution is None:
            return NewtonResult(
                displacement,
                load_factor,
                False,
                _linear_failure_reason(linear_result.diagnostics),
                evaluation,
                tuple(history),
                linear_result.diagnostics,
            )
        step_free = linear_result.solution
        step = np.zeros(total_dofs, dtype=float)
        step[free] = step_free
        step_norm = float(np.linalg.norm(step_free))

        merit = 0.5 * evaluation.free_residual_norm**2
        slope = -evaluation.free_residual_norm**2
        accepted: EquilibriumEvaluation | None = None
        alpha = 1.0
        line_iteration = 0
        for line_iteration in range(settings.maximum_line_search_iterations):
            trial_displacement = displacement + alpha * step
            try:
                trial = evaluate_equilibrium(
                    problem,
                    trial_displacement,
                    load_factor=load_factor,
                    tolerance=tolerance,
                )
            except BulkGeometryError:
                trial = None
            if trial is not None:
                trial_merit = 0.5 * trial.free_residual_norm**2
                armijo_bound = merit + settings.armijo_coefficient * alpha * slope
                if (
                    trial.free_residual_norm <= threshold
                    or trial_merit <= armijo_bound
                ):
                    accepted = trial
                    break
            alpha *= settings.line_search_reduction
            if alpha < settings.minimum_step:
                break
        if accepted is None:
            return NewtonResult(
                displacement,
                load_factor,
                False,
                "line_search_failed",
                evaluation,
                tuple(history),
            )

        displacement = accepted.displacement.copy()
        evaluation = accepted
        relative = _relative_residual(evaluation.free_residual_norm, initial_norm)
        history.append(
            NewtonIteration(
                iteration=iteration + 1,
                residual_norm=evaluation.free_residual_norm,
                relative_residual=relative,
                potential=evaluation.potential,
                minimum_jacobian=evaluation.bulk.minimum_jacobian,
                step_norm=step_norm,
                accepted_step=alpha,
                line_search_iterations=line_iteration,
                linear_solve=linear_result.diagnostics,
            )
        )
        if evaluation.free_residual_norm <= threshold:
            return NewtonResult(
                displacement,
                load_factor,
                True,
                "converged",
                evaluation,
                tuple(history),
            )

    return NewtonResult(
        displacement,
        load_factor,
        False,
        "maximum_iterations",
        evaluation,
        tuple(history),
    )


def solve_load_steps(
    problem: EquilibriumProblem,
    load_factors: FloatArray,
    initial_displacement: FloatArray | None = None,
    *,
    options: NewtonOptions | None = None,
    tolerance: float = 1.0e-12,
) -> tuple[NewtonResult, ...]:
    """Solve monotonically increasing load levels using the prior solution as predictor."""

    factors = np.asarray(load_factors, dtype=float)
    if factors.ndim != 1 or len(factors) == 0:
        raise ValueError("load_factors must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(factors)) or np.any(factors < 0.0):
        raise ValueError("load_factors must be finite and nonnegative")
    if np.any(factors[1:] <= factors[:-1]):
        raise ValueError("load_factors must be strictly increasing")

    displacement = initial_displacement
    results: list[NewtonResult] = []
    for factor in factors:
        result = solve_equilibrium(
            problem,
            displacement,
            load_factor=float(factor),
            options=options,
            tolerance=tolerance,
        )
        results.append(result)
        if not result.converged:
            break
        displacement = result.displacement
    return tuple(results)
