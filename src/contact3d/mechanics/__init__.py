"""Finite-strain mechanics public API."""

from .bulk_material import (
    BulkGeometryError,
    NeoHookeanMaterial,
    NeoHookeanResponse,
    evaluate_neo_hookean,
)
from .equilibrium import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumEvaluation,
    EquilibriumProblem,
    evaluate_equilibrium,
)
from .model import FloatArray, IntArray
from .oracle import (
    numerical_neo_hookean_tangent,
    numerical_tet4_mesh_tangent,
    numerical_tet4_tangent,
)
from .sparse import CSRMatrix, SparseAccumulator
from .sparse_tet4 import (
    Tet4SparseEvaluation,
    Tet4Sparsity,
    assemble_tet4_sparse,
)
from .tet4 import (
    Tet4Evaluation,
    Tet4Mesh,
    Tet4MeshEvaluation,
    Tet4Reference,
    evaluate_tet4,
    evaluate_tet4_mesh,
    tet4_deformation_gradient,
)

__all__ = [
    "BulkGeometryError",
    "CSRMatrix",
    "DeadLoad",
    "DirichletConstraints",
    "EquilibriumEvaluation",
    "EquilibriumProblem",
    "FloatArray",
    "IntArray",
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
    "evaluate_equilibrium",
    "evaluate_neo_hookean",
    "evaluate_tet4",
    "evaluate_tet4_mesh",
    "numerical_neo_hookean_tangent",
    "numerical_tet4_mesh_tangent",
    "numerical_tet4_tangent",
    "tet4_deformation_gradient",
]
