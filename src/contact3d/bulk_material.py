"""Temporary flat-path exports for finite-strain material models.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mechanics import (
    BulkGeometryError,
    NeoHookeanMaterial,
    NeoHookeanResponse,
    evaluate_neo_hookean,
)

__all__ = [
    "BulkGeometryError",
    "NeoHookeanMaterial",
    "NeoHookeanResponse",
    "evaluate_neo_hookean",
]
