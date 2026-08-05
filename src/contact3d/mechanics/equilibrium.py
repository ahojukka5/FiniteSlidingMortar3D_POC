"""Standalone finite-strain bulk equilibrium assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bulk_material import NeoHookeanMaterial
from .model import FloatArray, IntArray
from .sparse_tet4 import Tet4SparseEvaluation, Tet4Sparsity, assemble_tet4_sparse
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
