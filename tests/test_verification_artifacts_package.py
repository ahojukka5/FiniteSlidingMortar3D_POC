from __future__ import annotations

import ast
from pathlib import Path

import contact3d.benchmark_artifacts as legacy
import verification.artifacts as owner

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src/contact3d/benchmark_artifacts.py"
PUBLIC_API = (
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
)


def test_artifact_compatibility_module_is_definition_free() -> None:
    tree = ast.parse(FACADE.read_text(encoding="utf-8"), filename=str(FACADE))
    definitions = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert not definitions


def test_artifact_compatibility_imports_preserve_identity() -> None:
    for name in PUBLIC_API:
        assert getattr(legacy, name) is getattr(owner, name)


def test_artifact_owner_is_outside_installable_contact3d_package() -> None:
    assert Path(owner.__file__).resolve().is_relative_to(ROOT / "verification")
    assert not Path(owner.__file__).resolve().is_relative_to(ROOT / "src/contact3d")
