"""Compatibility export for triangle quadrature.

New code should import :func:`triangle_rule` from :mod:`contact3d.geometry`.
"""

from .geometry.quadrature import triangle_rule

__all__ = ["triangle_rule"]
