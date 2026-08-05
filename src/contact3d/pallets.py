"""Temporary migration exports for geometry pallet linearization.

Remove this module in issue #136 after repository imports use
``contact3d.geometry`` directly.
"""

from .geometry.pallets import (
    FacetPalletLinearization,
    MortarPalletLinearization,
    PalletFanLinearization,
    PalletTopologyError,
    SignedAreaLinearization,
    linearize_centroid_fan,
    linearize_facet_pallets,
    polygon_signed_area_linearized,
)

__all__ = [
    "FacetPalletLinearization",
    "MortarPalletLinearization",
    "PalletFanLinearization",
    "PalletTopologyError",
    "SignedAreaLinearization",
    "linearize_centroid_fan",
    "linearize_facet_pallets",
    "polygon_signed_area_linearized",
]
