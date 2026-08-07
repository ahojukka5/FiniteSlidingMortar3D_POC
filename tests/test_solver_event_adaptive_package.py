from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_adaptive as legacy
import contact3d.event_solver as aggregate
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


def test_legacy_event_adaptive_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "event_adaptive.py").read_text())
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_event_adaptive_exports_preserve_object_identity() -> None:
    names = (
        "AdaptiveTopologyEventBatch",
        "EventAwareAdaptiveContactResult",
        "solve_event_aware_adaptive_contact_path",
    )
    for name in names:
        assert getattr(legacy, name) is getattr(events, name)
        assert getattr(aggregate, name) is getattr(events, name)
        assert getattr(events, name).__module__ == "contact3d.solvers.events.adaptive"


def test_event_adaptive_owner_avoids_flat_solver_facades() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "events" / "adaptive.py")
    forbidden = {
        "...adaptive_model",
        "...coupled",
        "...enforcement_state",
        "...event_adaptive",
        "...event_augmented",
        "...event_solver",
        "...model",
        "...scaled_solver",
    }
    assert imports.isdisjoint(forbidden)


def test_core_consumers_use_event_adaptive_owner() -> None:
    aggregate_imports = imported_modules(SOURCE_ROOT / "event_solver.py")
    restart_imports = imported_modules(SOURCE_ROOT / "restart_diagnostics.py")
    assert ".event_adaptive" not in aggregate_imports
    assert ".solvers.events" in aggregate_imports
    assert ".event_adaptive" not in restart_imports
    assert ".solvers.events.adaptive" in restart_imports
