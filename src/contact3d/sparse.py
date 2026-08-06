"""Temporary flat-path exports for deterministic sparse matrix utilities.

This module is migration scaffolding only and is removed by issue #136.
"""

from .mechanics import CSRMatrix, SparseAccumulator

__all__ = ["CSRMatrix", "SparseAccumulator"]
