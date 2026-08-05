"""Temporary flat-path exports for :mod:`contact3d.mortar`.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mortar import (
    ContactEvaluation,
    ContactPair,
    GlobalMortarWeights,
    assemble_mortar_weights,
    evaluate_contact,
    fixed_mortar_contact_tangent,
    numerical_contact_tangent,
)

__all__ = [
    "ContactEvaluation",
    "ContactPair",
    "GlobalMortarWeights",
    "assemble_mortar_weights",
    "evaluate_contact",
    "fixed_mortar_contact_tangent",
    "numerical_contact_tangent",
]
