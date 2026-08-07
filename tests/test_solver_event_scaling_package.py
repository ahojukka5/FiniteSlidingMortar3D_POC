from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_scaled as legacy_scaling
import contact3d.solvers.events as events

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


def test_legacy_event_scaled_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "event_scaled.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_legacy_event_scaling_exports_preserve_object_identity() -> None:
    names = (
        "EventAwareScaleAwareAugmentedContactResult",
        "solve_event_aware_scale_aware_augmented_contact",
    )

    for name in names:
        assert getattr(legacy_scaling, name) is getattr(events, name)


def test_event_scaling_uses_solver_owned_normalization_helpers() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "events" / "scaling.py")

    assert "...scaled_solver" not in imports
    assert "..scaling" in imports


def test_production_event_composition_uses_the_owning_package() -> None:
    adaptive_imports = imported_modules(SOURCE_ROOT / "event_adaptive.py")
    aggregate_imports = imported_modules(SOURCE_ROOT / "event_solver.py")

    assert ".event_scaled" not in adaptive_imports
    assert ".solvers.events" in adaptive_imports
    assert ".event_scaled" not in aggregate_imports
    assert ".solvers.events" in aggregate_imports
