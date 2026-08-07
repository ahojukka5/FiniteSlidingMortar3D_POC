"""Compatibility facade for repository plotting infrastructure.

New repository code should import this API from :mod:`verification.plots`.
This module remains only until the compatibility cleanup in #136.
"""

from verification.plots import (
    write_bar_chart,
    write_category_timeline,
    write_line_chart,
    write_mesh_projection_overlay,
    write_polygon_overlay,
    write_sparsity,
)

__all__ = [
    "write_bar_chart",
    "write_category_timeline",
    "write_line_chart",
    "write_mesh_projection_overlay",
    "write_polygon_overlay",
    "write_sparsity",
]
