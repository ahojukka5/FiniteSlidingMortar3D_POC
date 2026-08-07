from __future__ import annotations

import ast
from pathlib import Path

import contact3d.adaptive as adaptive
import contact3d.adaptive_model as legacy_model
import contact3d.adaptive_options as legacy_options
import contact3d.adaptive_solver as legacy_solver
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


def test_legacy_continuation_modules_are_reexport_only_facades() -> None:
    for name in ("adaptive_model.py", "adaptive_options.py", "adaptive_solver.py"):
        tree = ast.parse((SOURCE_ROOT / name).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )


def test_legacy_continuation_contracts_preserve_object_identity() -> None:
    model_names = (
        "AdaptiveAcceptedStep",
        "AdaptiveAttemptAction",
        "AdaptiveContactAttempt",
        "AdaptiveContactResult",
        "AdaptiveTerminationReason",
    )
    option_names = (
        "AdaptiveContactOptions",
        "AdaptiveLoadOptions",
        "AdaptivePenaltyOptions",
    )
    driver_names = (
        "AdaptiveSolver",
        "contact_penalties",
        "solve_adaptive_contact_path",
        "with_contact_penalties",
    )

    for name in model_names:
        assert getattr(legacy_model, name) is getattr(solvers, name)
    for name in option_names:
        assert getattr(legacy_options, name) is getattr(solvers, name)
    for name in driver_names:
        assert getattr(legacy_solver, name) is getattr(solvers, name)


def test_aggregate_adaptive_api_uses_solver_owned_objects() -> None:
    names = (
        "AdaptiveAcceptedStep",
        "AdaptiveContactAttempt",
        "AdaptiveContactOptions",
        "AdaptiveContactResult",
        "AdaptiveLoadOptions",
        "AdaptivePenaltyOptions",
        "contact_penalties",
        "solve_adaptive_contact_path",
        "with_contact_penalties",
    )

    for name in names:
        assert getattr(adaptive, name) is getattr(solvers, name)


def test_continuation_implementation_avoids_legacy_solver_facades() -> None:
    imports = imported_modules(SOURCE_ROOT / "solvers" / "continuation.py")
    forbidden = {
        "..adaptive",
        "..adaptive_model",
        "..adaptive_options",
        "..adaptive_solver",
        "..coupled",
        "..scaled_solver",
    }

    assert imports.isdisjoint(forbidden)


def test_event_adaptive_consumes_the_solver_owned_driver() -> None:
    imports = imported_modules(SOURCE_ROOT / "event_adaptive.py")

    assert ".adaptive_solver" not in imports
    assert ".solvers.continuation" in imports
