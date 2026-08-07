from __future__ import annotations

import ast
from pathlib import Path

import contact3d.multiplier_transport as legacy
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


def test_legacy_multiplier_transport_module_is_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "multiplier_transport.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_legacy_multiplier_transport_preserves_object_identity() -> None:
    assert legacy.MultiplierTransportRecord is events.MultiplierTransportRecord
    assert legacy.transport_multiplier_states is events.transport_multiplier_states


def test_event_solver_consumers_use_the_owning_transport_module() -> None:
    model_imports = imported_modules(SOURCE_ROOT / "event_model.py")
    newton_imports = imported_modules(SOURCE_ROOT / "event_newton.py")
    aggregate_imports = imported_modules(SOURCE_ROOT / "event_solver.py")
    scaling_imports = imported_modules(
        SOURCE_ROOT / "solvers" / "events" / "scaling.py"
    )

    assert ".solvers.events.multiplier_transport" in model_imports
    assert ".solvers.events.multiplier_transport" in newton_imports
    assert ".solvers.events" in aggregate_imports
    assert ".multiplier_transport" in scaling_imports
    assert ".multiplier_transport" not in model_imports
    assert ".multiplier_transport" not in newton_imports
    assert "...multiplier_transport" not in scaling_imports


def test_transport_owner_does_not_import_legacy_solver_facades() -> None:
    imports = imported_modules(
        SOURCE_ROOT / "solvers" / "events" / "multiplier_transport.py"
    )
    forbidden = {
        "...event_solver",
        "...event_newton",
        "...event_model",
        "...multiplier_transport",
        "...scaled_solver",
    }

    assert imports.isdisjoint(forbidden)
