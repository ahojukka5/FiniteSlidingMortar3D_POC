"""Temporary migration exports for moving-overlap mortar tangents.

New code should import these objects from :mod:`contact3d.mortar`.
This module is deleted by issue #136.
"""

from .mortar.moving import (
    MortarWeightJacobian,
    analytical_mortar_weight_jacobian,
    moving_mortar_contact_tangent,
    numerical_mortar_weight_jacobian,
)

__all__ = [
    "MortarWeightJacobian",
    "analytical_mortar_weight_jacobian",
    "moving_mortar_contact_tangent",
    "numerical_mortar_weight_jacobian",
]
