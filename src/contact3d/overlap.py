"""Temporary flat-path exports for :mod:`contact3d.mortar.overlap`.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mortar.overlap import build_facet_overlap, integrate_facet_pair

__all__ = ["build_facet_overlap", "integrate_facet_pair"]
