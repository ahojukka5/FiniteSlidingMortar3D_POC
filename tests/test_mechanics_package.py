from __future__ import annotations

import contact3d.bulk as bulk
import contact3d.bulk_material as flat_material
import contact3d.bulk_oracle as flat_oracle
import contact3d.mechanics as mechanics
import contact3d.tet4 as flat_tet4
from contact3d.mechanics import (
    BulkGeometryError,
    NeoHookeanMaterial,
    Tet4Mesh,
    Tet4Reference,
    evaluate_neo_hookean,
    evaluate_tet4,
    evaluate_tet4_mesh,
    numerical_neo_hookean_tangent,
    numerical_tet4_mesh_tangent,
    numerical_tet4_tangent,
    tet4_deformation_gradient,
)


def test_mechanics_public_api_owns_production_kernels_and_oracles() -> None:
    assert mechanics.BulkGeometryError is BulkGeometryError
    assert mechanics.NeoHookeanMaterial is NeoHookeanMaterial
    assert mechanics.Tet4Mesh is Tet4Mesh
    assert mechanics.Tet4Reference is Tet4Reference
    assert mechanics.evaluate_neo_hookean is evaluate_neo_hookean
    assert mechanics.evaluate_tet4 is evaluate_tet4
    assert mechanics.evaluate_tet4_mesh is evaluate_tet4_mesh
    assert mechanics.numerical_neo_hookean_tangent is numerical_neo_hookean_tangent
    assert mechanics.numerical_tet4_tangent is numerical_tet4_tangent
    assert mechanics.numerical_tet4_mesh_tangent is numerical_tet4_mesh_tangent
    assert mechanics.tet4_deformation_gradient is tet4_deformation_gradient


def test_bulk_facade_reuses_mechanics_contract_identities() -> None:
    mechanics_exports = (
        "BulkGeometryError",
        "CSRMatrix",
        "NeoHookeanMaterial",
        "NeoHookeanResponse",
        "SparseAccumulator",
        "Tet4Evaluation",
        "Tet4Mesh",
        "Tet4MeshEvaluation",
        "Tet4Reference",
        "Tet4SparseEvaluation",
        "Tet4Sparsity",
        "assemble_tet4_sparse",
        "evaluate_neo_hookean",
        "evaluate_tet4",
        "evaluate_tet4_mesh",
        "numerical_neo_hookean_tangent",
        "numerical_tet4_mesh_tangent",
        "numerical_tet4_tangent",
        "tet4_deformation_gradient",
    )

    for name in mechanics_exports:
        assert getattr(bulk, name) is getattr(mechanics, name)


def test_flat_mechanics_modules_are_direct_temporary_reexports() -> None:
    assert flat_material.BulkGeometryError is BulkGeometryError
    assert flat_material.NeoHookeanMaterial is NeoHookeanMaterial
    assert flat_material.evaluate_neo_hookean is evaluate_neo_hookean
    assert flat_tet4.Tet4Mesh is Tet4Mesh
    assert flat_tet4.Tet4Reference is Tet4Reference
    assert flat_tet4.evaluate_tet4 is evaluate_tet4
    assert flat_tet4.evaluate_tet4_mesh is evaluate_tet4_mesh
    assert flat_tet4.tet4_deformation_gradient is tet4_deformation_gradient
    assert flat_oracle.numerical_neo_hookean_tangent is numerical_neo_hookean_tangent
    assert flat_oracle.numerical_tet4_tangent is numerical_tet4_tangent
    assert flat_oracle.numerical_tet4_mesh_tangent is numerical_tet4_mesh_tangent
