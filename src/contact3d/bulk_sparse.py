"""Temporary flat-path exports for sparse TET4 assembly.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mechanics import (
    Tet4SparseEvaluation,
    Tet4Sparsity,
    assemble_tet4_sparse,
    element_dofs,
)

__all__ = [
    "Tet4SparseEvaluation",
    "Tet4Sparsity",
    "assemble_tet4_sparse",
    "element_dofs",
]
