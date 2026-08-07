from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_geometry as geometry
import contact3d.solvers.events.localization as localization

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


def defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_event_geometry_contains_only_shared_signature_construction() -> None:
    names = defined_names(SOURCE_ROOT / "event_geometry.py")

    assert names == {"contact_topology_signatures"}
    assert not hasattr(geometry, "_observation")
    assert not hasattr(geometry, "_event_signatures")
    assert not hasattr(geometry, "_RECOVERABLE_ERRORS")


def test_event_localization_owner_exposes_solver_observations() -> None:
    assert localization.RECOVERABLE_CONTACT_EVENT_ERRORS
    assert callable(localization.event_signatures)
    assert callable(localization.observe_event_trial)
    assert callable(localization.recoverable_event_kind)


def test_event_newton_uses_localization_owner() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "events" / "newton.py")

    assert ".localization" in imports
    assert "...event_geometry" not in imports


def test_localization_uses_shared_signature_geometry() -> None:
    imports = imported_modules(
        SOURCE_ROOT / "solvers" / "events" / "localization.py"
    )

    assert "...event_geometry" in imports
    assert "...event_newton" not in imports
    assert "...event_solver" not in imports


def test_topology_scan_remains_solver_independent() -> None:
    imports = imported_modules(SOURCE_ROOT / "topology_scan.py")

    assert ".event_geometry" in imports
    assert ".solvers.events.localization" not in imports
