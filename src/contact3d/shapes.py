"""Compatibility exports for surface interpolation helpers.

New code should import these functions from :mod:`contact3d.geometry`.
"""

from .geometry.shapes import (
    center_parent,
    infer_facet_kind,
    inverse_map_2d,
    map_to_physical,
    shape_gradients,
    shape_values,
)

__all__ = [
    "center_parent",
    "infer_facet_kind",
    "inverse_map_2d",
    "map_to_physical",
    "shape_gradients",
    "shape_values",
]
