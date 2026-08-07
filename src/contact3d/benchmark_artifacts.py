"""Compatibility facade for repository artifact infrastructure.

New repository code should import this API from :mod:`verification.artifacts`.
This module remains only until the compatibility cleanup in #136.
"""

from verification.artifacts import (
    BENCHMARK_SCHEMA_VERSION,
    ArtifactKind,
    ArtifactRecord,
    BenchmarkArtifactError,
    BenchmarkArtifactWriter,
    NumericTolerance,
    benchmark_provenance,
    compare_numeric_metrics,
    validate_benchmark_manifest,
    write_surface_vtp,
    write_tet4_vtu,
)

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkArtifactError",
    "BenchmarkArtifactWriter",
    "NumericTolerance",
    "benchmark_provenance",
    "compare_numeric_metrics",
    "validate_benchmark_manifest",
    "write_surface_vtp",
    "write_tet4_vtu",
]
