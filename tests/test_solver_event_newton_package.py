from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_newton as legacy_newton
import contact3d.event_solver as aggregate
import contact3d.solvers.events as events
import contact3d.solvers.events.newton as owner

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


def test_legacy_event_newton_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "event_newton.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_event_newton_preserves_solver_identity() -> None:
    function = owner.solve_event_aware_coupled_equilibrium

    assert legacy_newton.solve_event_aware_coupled_equilibrium is function
    assert events.solve_event_aware_coupled_equilibrium is function
    assert aggregate.solve_event_aware_coupled_equilibrium is function
    assert function.__module__ == "contact3d.solvers.events.newton"


def test_event_newton_owner_avoids_flat_solver_facades() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "events" / "newton.py")
    forbidden = {
        "...coupled",
        "...enforcement_state",
        "...event_model",
        "...event_newton",
        "...linear_solver",
        "...model",
    }

    assert imports.isdisjoint(forbidden)


def test_event_solver_consumers_use_the_newton_owner() -> None:
    augmentation_imports = imported_modules(
        SOURCE_ROOT / "solvers" / "events" / "augmentation.py"
    )
    scaling_imports = imported_modules(
        SOURCE_ROOT / "solvers" / "events" / "scaling.py"
    )
    aggregate_imports = imported_modules(SOURCE_ROOT / "event_solver.py")

    assert ".newton" in augmentation_imports
    assert ".newton" in scaling_imports
    assert "...event_newton" not in augmentation_imports
    assert "...event_newton" not in scaling_imports
    assert ".event_newton" not in aggregate_imports
    assert ".solvers.events" in aggregate_imports
