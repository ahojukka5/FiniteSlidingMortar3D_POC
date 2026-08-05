from __future__ import annotations

import numpy as np

import contact3d.bulk_sparse as flat_sparse_tet4
import contact3d.sparse as flat_sparse
from contact3d.mechanics import (
    CSRMatrix,
    NeoHookeanMaterial,
    SparseAccumulator,
    Tet4Mesh,
    Tet4SparseEvaluation,
    Tet4Sparsity,
    assemble_tet4_sparse,
    evaluate_tet4_mesh,
)
from contact3d.mechanics.sparse_tet4 import element_dofs


def _single_tet4_mesh() -> Tet4Mesh:
    return Tet4Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )


def test_flat_sparse_modules_are_temporary_identity_exports() -> None:
    assert flat_sparse.CSRMatrix is CSRMatrix
    assert flat_sparse.SparseAccumulator is SparseAccumulator
    assert flat_sparse_tet4.Tet4SparseEvaluation is Tet4SparseEvaluation
    assert flat_sparse_tet4.Tet4Sparsity is Tet4Sparsity
    assert flat_sparse_tet4.assemble_tet4_sparse is assemble_tet4_sparse
    assert flat_sparse_tet4.element_dofs is element_dofs


def test_sparse_accumulator_keeps_deterministic_csr_ordering() -> None:
    accumulator = SparseAccumulator((3, 3))
    accumulator.add_block(
        np.array([2, 0], dtype=np.int64),
        np.array([2, 0], dtype=np.int64),
        np.array([[4.0, 3.0], [2.0, 1.0]]),
    )
    matrix = accumulator.to_csr()

    assert isinstance(matrix, CSRMatrix)
    np.testing.assert_array_equal(matrix.indptr, np.array([0, 2, 2, 4]))
    np.testing.assert_array_equal(matrix.indices, np.array([0, 2, 0, 2]))
    np.testing.assert_allclose(
        matrix.to_dense(),
        np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]),
    )


def test_sparse_tet4_assembly_matches_dense_mechanics() -> None:
    mesh = _single_tet4_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    displacement = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.002, 0.0],
            [0.0, 0.02, 0.001],
            [0.001, 0.0, 0.03],
        ]
    )

    dense = evaluate_tet4_mesh(mesh, displacement, material)
    pattern = Tet4Sparsity.from_mesh(
        mesh,
        (np.array([0, 1, 2], dtype=np.int64),),
    )
    sparse = assemble_tet4_sparse(
        mesh,
        displacement,
        material,
        sparsity=pattern,
    )

    assert pattern.additional_positions[0].shape == (3, 3)
    assert sparse.energy == dense.energy
    np.testing.assert_allclose(sparse.current_nodes, dense.current_nodes)
    np.testing.assert_allclose(sparse.residual, dense.residual)
    np.testing.assert_allclose(sparse.tangent.to_dense(), dense.tangent)
