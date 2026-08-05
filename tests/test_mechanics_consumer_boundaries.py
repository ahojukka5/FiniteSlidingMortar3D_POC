"""Architecture checks for mechanics consumers."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "contact3d"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
    }


def test_equilibrium_consumes_mechanics_package_boundary() -> None:
    imports = imported_modules(SOURCE_ROOT / "equilibrium.py")

    assert "mechanics" in imports
    assert "mechanics.model" in imports
    assert not imports.intersection(
        {"bulk_material", "bulk_sparse", "model", "sparse", "tet4"}
    )
