"""Public finite-strain bulk-mechanics API."""

from .bulk_material import (
    BulkGeometryError,
    NeoHookeanMaterial,
    NeoHookeanResponse,
    evaluate_neo_hookean,
)
from .bulk_oracle import (
    numerical_neo_hookean_tangent,
    numerical_tet4_mesh_tangent,
    numerical_tet4_tangent,
)
from .bulk_sparse import (
    Tet4SparseEvaluation,
    Tet4Sparsity,
    assemble_tet4_sparse,
)
from .equilibrium import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumEvaluation,
    EquilibriumProblem,
    NewtonIteration,
    NewtonOptions,
    NewtonResult,
    evaluate_equilibrium,
    solve_equilibrium,
    solve_load_steps,
)
from .sparse import CSRMatrix, SparseAccumulator
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
    "NeoHookeanMaterial",
    "NeoHookeanResponse",
    "NewtonIteration",
    "NewtonOptions",
    "NewtonResult",
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
    "solve_equilibrium",
    "solve_load_steps",
    "tet4_deformation_gradient",
]
