"""Temporary flat-path exports for dense total-Lagrangian TET4 mechanics.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mechanics import (
    Tet4Evaluation,
    Tet4Mesh,
    Tet4MeshEvaluation,
    Tet4Reference,
    evaluate_tet4,
    evaluate_tet4_mesh,
    tet4_deformation_gradient,
)

__all__ = [
    "Tet4Evaluation",
    "Tet4Mesh",
    "Tet4MeshEvaluation",
    "Tet4Reference",
    "evaluate_tet4",
    "evaluate_tet4_mesh",
    "tet4_deformation_gradient",
]
