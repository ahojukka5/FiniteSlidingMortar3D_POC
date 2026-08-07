from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_solver as event_solver
import contact3d.restart_diagnostics as legacy
import contact3d.solvers.events as events
import contact3d.solvers.events.restart as owner

ROOT = Path(__file__).resolve().parents[1] / "src" / "contact3d"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _defined_names(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_flat_restart_module_is_a_pure_compatibility_facade() -> None:
    path = ROOT / "restart_diagnostics.py"
    assert _defined_names(path) == set()


def test_restart_exports_preserve_identity() -> None:
    names = (
        "RestartAttemptDiagnostic",
        "RestartCount",
        "RestartDiagnosticOptions",
        "RestartDiagnostics",
        "RestartEventRecord",
        "RestartLoopDiagnostic",
        "analyze_restart_diagnostics",
    )
    for name in names:
        owned = getattr(owner, name)
        assert getattr(events, name) is owned
        assert getattr(event_solver, name) is owned
        assert getattr(legacy, name) is owned


def test_restart_models_report_the_owning_module() -> None:
    assert owner.RestartDiagnosticOptions.__module__ == (
        "contact3d.solvers.events.restart"
    )
    assert owner.RestartDiagnostics.__module__ == "contact3d.solvers.events.restart"


def test_restart_owner_avoids_flat_solver_facades() -> None:
    modules = _imported_modules(ROOT / "solvers" / "events" / "restart.py")
    forbidden = {
        "adaptive",
        "adaptive_model",
        "event_adaptive",
        "event_solver",
        "restart_diagnostics",
    }
    assert modules.isdisjoint(forbidden)


def test_event_solver_aggregates_restart_api_from_owner_package() -> None:
    source = (ROOT / "event_solver.py").read_text(encoding="utf-8")
    assert "from .solvers.events import" in source
    assert "from .restart_diagnostics import" not in source
