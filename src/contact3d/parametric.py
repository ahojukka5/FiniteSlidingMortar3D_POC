"""Temporary migration exports for parametric geometry linearization.

Remove this module in issue #136 after repository imports use
``contact3d.geometry`` directly.
"""

from .geometry.parametric import (
    FacetQuadratureLinearization,
    InverseMapLinearization,
    InverseMapTopologyError,
    MortarQuadraturePointLinearization,
    inverse_map_2d_linearized,
    linearize_facet_quadrature,
)

__all__ = [
    "FacetQuadratureLinearization",
    "InverseMapLinearization",
    "InverseMapTopologyError",
    "MortarQuadraturePointLinearization",
    "inverse_map_2d_linearized",
    "linearize_facet_quadrature",
]
