from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_model as legacy_model
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


def test_legacy_event_model_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "event_model.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_legacy_event_results_preserve_object_identity() -> None:
    names = (
        "EventAwareAugmentedContactResult",
        "EventAwareCoupledNewtonResult",
    )

    for name in names:
        assert getattr(legacy_model, name) is getattr(events, name)
        assert getattr(aggregate, name) is getattr(events, name)
        assert getattr(events, name).__module__ == "contact3d.solvers.events.results"


def test_event_result_owner_avoids_flat_solver_facades() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "events" / "results.py")
    forbidden = {
        "...event_model",
        "...event_solver",
        "...coupled",
        "...linear_solver",
        "...model",
        "...multiplier_transport",
    }

    assert imports.isdisjoint(forbidden)


def test_event_solver_aggregate_uses_the_owning_package() -> None:
    imports = imported_modules(SOURCE_ROOT / "event_solver.py")

    assert ".event_model" not in imports
    assert ".solvers.events" in imports
