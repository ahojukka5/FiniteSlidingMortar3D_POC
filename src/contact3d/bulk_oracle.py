"""Temporary flat-path exports for mechanics verification oracles.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mechanics import (
    numerical_neo_hookean_tangent,
    numerical_tet4_mesh_tangent,
    numerical_tet4_tangent,
)

__all__ = [
    "numerical_neo_hookean_tangent",
    "numerical_tet4_mesh_tangent",
    "numerical_tet4_tangent",
]
