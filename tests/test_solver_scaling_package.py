from __future__ import annotations

import ast
from pathlib import Path

import contact3d.scaled_solver as legacy_scaling
import contact3d.solvers as solvers

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "contact3d"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            modules.add(prefix + (node.module or ""))
    return modules


def test_legacy_scaled_solver_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "scaled_solver.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_legacy_scale_aware_exports_preserve_object_identity() -> None:
    names = (
        "ScaleAwareAugmentationIteration",
        "ScaleAwareAugmentedContactResult",
        "ScaleAwareNewtonIteration",
        "solve_scale_aware_augmented_contact",
    )

    for name in names:
        assert getattr(legacy_scaling, name) is getattr(solvers, name)


def test_solver_scaling_does_not_depend_on_legacy_solver_facades() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "scaling.py")

    assert "..scaled_solver" not in imports
    assert "..coupled" not in imports
    assert "..bulk" not in imports
