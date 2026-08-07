from __future__ import annotations

import ast
from pathlib import Path

import contact3d
import contact3d.linear_solver as legacy
import contact3d.solvers as solvers
import contact3d.solvers.linear as owner

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


def test_flat_linear_solver_module_is_a_pure_compatibility_facade() -> None:
    assert _defined_names(ROOT / "linear_solver.py") == set()


def test_linear_solver_exports_preserve_identity() -> None:
    names = (
        "LinearPreconditioner",
        "LinearSolveDiagnostics",
        "LinearSolveResult",
        "LinearSolverOptions",
        "block_jacobi_preconditioner_factory",
        "extract_csr_submatrix",
        "field_split_preconditioner_factory",
        "solve_linear_system",
        "solve_reduced_system",
    )
    for name in names:
        owned = getattr(owner, name)
        assert getattr(legacy, name) is owned
        assert getattr(solvers, name) is owned
        assert getattr(contact3d, name) is owned


def test_linear_solver_models_report_solver_ownership() -> None:
    assert owner.LinearSolverOptions.__module__ == "contact3d.solvers.linear"
    assert owner.LinearSolveDiagnostics.__module__ == "contact3d.solvers.linear"
    assert owner.LinearSolveResult.__module__ == "contact3d.solvers.linear"


def test_linear_owner_does_not_import_flat_linear_solver_facade() -> None:
    modules = _imported_modules(ROOT / "solvers" / "linear.py")
    assert "linear_solver" not in modules
    assert "contact3d.linear_solver" not in modules


def test_linear_facade_forwards_only_to_solver_owner() -> None:
    source = (ROOT / "linear_solver.py").read_text(encoding="utf-8")
    assert "from .solvers.linear import" in source
