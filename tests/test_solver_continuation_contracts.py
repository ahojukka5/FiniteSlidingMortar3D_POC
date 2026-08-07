from __future__ import annotations

import ast
from pathlib import Path

import contact3d.adaptive as adaptive
import contact3d.adaptive_model as legacy_model
import contact3d.adaptive_options as legacy_options
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


def test_legacy_continuation_contract_modules_are_reexport_only_facades() -> None:
    for name in ("adaptive_model.py", "adaptive_options.py"):
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

    for name in model_names:
        assert getattr(legacy_model, name) is getattr(solvers, name)
    for name in option_names:
        assert getattr(legacy_options, name) is getattr(solvers, name)


def test_aggregate_adaptive_api_uses_solver_owned_contracts() -> None:
    names = (
        "AdaptiveAcceptedStep",
        "AdaptiveContactAttempt",
        "AdaptiveContactOptions",
        "AdaptiveContactResult",
        "AdaptiveLoadOptions",
        "AdaptivePenaltyOptions",
    )

    for name in names:
        assert getattr(adaptive, name) is getattr(solvers, name)


def test_continuation_contracts_do_not_import_legacy_solver_facades() -> None:
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
