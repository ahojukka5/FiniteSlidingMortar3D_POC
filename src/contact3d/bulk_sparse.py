"""Sparse global assembly for total-Lagrangian TET4 meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bulk_material import NeoHookeanMaterial
from .model import FloatArray, IntArray
from .sparse import CSRMatrix, SparseAccumulator
from .tet4 import Tet4Evaluation, Tet4Mesh, _validated_displacement, evaluate_tet4


@dataclass(frozen=True, slots=True)
class Tet4SparseEvaluation:
    """Sparse assembled energy, internal residual, and tangent."""

    current_nodes: FloatArray
    energy: float
    residual: FloatArray
    tangent: CSRMatrix
    element_evaluations: tuple[Tet4Evaluation, ...]

    @property
    def minimum_jacobian(self) -> float:
        return min(evaluation.jacobian for evaluation in self.element_evaluations)

    @property
    def force_balance(self) -> FloatArray:
        return np.sum(self.residual, axis=0)

    @property
    def moment_balance(self) -> FloatArray:
        return np.sum(np.cross(self.current_nodes, self.residual), axis=0)


def element_dofs(element: IntArray) -> IntArray:
    """Return interleaved translational DOFs for one four-node element."""

    return np.asarray(
        [3 * int(node) + component for node in element for component in range(3)],
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True)
class Tet4Sparsity:
    """Reusable symbolic CSR pattern and local-to-global value positions."""

    shape: tuple[int, int]
    indptr: IntArray
    indices: IntArray
    element_positions: IntArray

    @classmethod
    def from_mesh(cls, mesh: Tet4Mesh) -> Tet4Sparsity:
        total_dofs = 3 * mesh.node_count
        symbolic = SparseAccumulator((total_dofs, total_dofs))
        identity_block = np.ones((12, 12), dtype=float)
        dof_sets: list[IntArray] = []
        for element in mesh.elements:
            dofs = element_dofs(element)
            dof_sets.append(dofs)
            symbolic.add_block(dofs, dofs, identity_block)
        pattern = symbolic.to_csr()

        positions_by_row: list[dict[int, int]] = []
        for row in range(total_dofs):
            start = int(pattern.indptr[row])
            stop = int(pattern.indptr[row + 1])
            positions_by_row.append(
                {
                    int(column): start + offset
                    for offset, column in enumerate(pattern.indices[start:stop])
                }
            )
        element_positions = np.empty((mesh.element_count, 12, 12), dtype=np.int64)
        for element_index, dofs in enumerate(dof_sets):
            for local_row, row in enumerate(dofs):
                lookup = positions_by_row[int(row)]
                for local_column, column in enumerate(dofs):
                    element_positions[element_index, local_row, local_column] = lookup[
                        int(column)
                    ]
        return cls(
            pattern.shape,
            pattern.indptr,
            pattern.indices,
            element_positions,
        )

    @property
    def nnz(self) -> int:
        return len(self.indices)

    def matrix(self, data: FloatArray) -> CSRMatrix:
        values = np.asarray(data, dtype=float)
        if values.shape != (self.nnz,):
            raise ValueError("sparse tangent data must match the symbolic nonzero count")
        return CSRMatrix(self.shape, self.indptr, self.indices, values)


def assemble_tet4_sparse(
    mesh: Tet4Mesh,
    displacement: FloatArray,
    material: NeoHookeanMaterial,
    *,
    sparsity: Tet4Sparsity | None = None,
    tolerance: float = 1.0e-12,
) -> Tet4SparseEvaluation:
    """Assemble residual and tangent into a reusable deterministic CSR pattern."""

    values = _validated_displacement(displacement, mesh.node_count)
    pattern = Tet4Sparsity.from_mesh(mesh) if sparsity is None else sparsity
    expected_shape = (3 * mesh.node_count, 3 * mesh.node_count)
    if pattern.shape != expected_shape or pattern.element_positions.shape != (
        mesh.element_count,
        12,
        12,
    ):
        raise ValueError("TET4 sparsity pattern is incompatible with the mesh")

    residual = np.zeros((mesh.node_count, 3), dtype=float)
    tangent_data = np.zeros(pattern.nnz, dtype=float)
    evaluations: list[Tet4Evaluation] = []
    energy = 0.0

    for element_index, (element, reference) in enumerate(
        zip(mesh.elements, mesh.element_references, strict=True)
    ):
        evaluation = evaluate_tet4(
            reference,
            values[element],
            material,
            tolerance=tolerance,
        )
        evaluations.append(evaluation)
        energy += evaluation.energy
        residual[element] += evaluation.internal_force
        np.add.at(
            tangent_data,
            pattern.element_positions[element_index].ravel(),
            evaluation.tangent.ravel(),
        )

    return Tet4SparseEvaluation(
        current_nodes=mesh.reference_nodes + values,
        energy=float(energy),
        residual=residual,
        tangent=pattern.matrix(tangent_data),
        element_evaluations=tuple(evaluations),
    )
