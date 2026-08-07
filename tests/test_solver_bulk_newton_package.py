from __future__ import annotations

import ast
from pathlib import Path

import contact3d
import contact3d.equilibrium as legacy
import contact3d.solvers as solvers
import contact3d.solvers.newton as owner
import contact3d.solvers.results as results

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


def test_flat_equilibrium_module_is_a_pure_compatibility_facade() -> None:
    assert _defined_names(ROOT / "equilibrium.py") == set()


def test_bulk_newton_exports_preserve_identity() -> None:
    assert legacy.NewtonOptions is results.NewtonOptions
    assert legacy.NewtonIteration is results.NewtonIteration
    assert legacy.NewtonResult is results.NewtonResult
    assert legacy.solve_equilibrium is owner.solve_equilibrium
    assert legacy.solve_load_steps is owner.solve_load_steps
    assert solvers.NewtonOptions is results.NewtonOptions
    assert solvers.solve_equilibrium is owner.solve_equilibrium
    assert solvers.solve_load_steps is owner.solve_load_steps
    assert contact3d.NewtonOptions is results.NewtonOptions
    assert contact3d.solve_equilibrium is owner.solve_equilibrium
    assert contact3d.solve_load_steps is owner.solve_load_steps


def test_bulk_newton_models_report_solver_ownership() -> None:
    assert results.NewtonOptions.__module__ == "contact3d.solvers.results"
    assert results.NewtonIteration.__module__ == "contact3d.solvers.results"
    assert results.NewtonResult.__module__ == "contact3d.solvers.results"


def test_solver_newton_and_results_do_not_import_flat_equilibrium_facade() -> None:
    for relative in ("solvers/newton.py", "solvers/results.py"):
        modules = _imported_modules(ROOT / relative)
        assert "equilibrium" not in modules
        assert "contact3d.equilibrium" not in modules


def test_equilibrium_facade_only_composes_mechanics_and_solver_owners() -> None:
    source = (ROOT / "equilibrium.py").read_text(encoding="utf-8")
    assert "from .mechanics import" in source
    assert "from .solvers.newton import" in source
    assert "from .solvers.results import" in source
