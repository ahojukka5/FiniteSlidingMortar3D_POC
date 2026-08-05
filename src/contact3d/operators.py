"""Temporary flat-path exports for :mod:`contact3d.mortar.operators`.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mortar.operators import (
    LocalMortarWeightLinearization,
    integrate_facet_pair_linearized,
)

__all__ = ["LocalMortarWeightLinearization", "integrate_facet_pair_linearized"]
