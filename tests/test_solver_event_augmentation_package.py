from __future__ import annotations

import ast
from pathlib import Path

import contact3d.event_augmented as legacy_augmentation
import contact3d.event_solver as aggregate
import contact3d.solvers.events as events
import contact3d.solvers.events.augmentation as owner

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


def test_legacy_event_augmentation_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "event_augmented.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_event_augmentation_preserves_solver_identity() -> None:
    function = owner.solve_event_aware_augmented_contact

    assert legacy_augmentation.solve_event_aware_augmented_contact is function
    assert events.solve_event_aware_augmented_contact is function
    assert aggregate.solve_event_aware_augmented_contact is function
    assert function.__module__ == "contact3d.solvers.events.augmentation"


def test_event_augmentation_owner_avoids_flat_solver_facades() -> None:
    path = SOURCE_ROOT / "solvers" / "events" / "augmentation.py"
    imports = imported_modules(path)
    forbidden = {
        "...coupled",
        "...enforcement_state",
        "...event_augmented",
        "...event_model",
        "...event_solver",
        "...model",
    }

    assert imports.isdisjoint(forbidden)
    assert "...coupling" in imports
    assert "...mechanics" in imports
    assert "...mortar.enforcement" in imports
    assert ".results" in imports


def test_event_augmentation_uses_public_state_validation() -> None:
    source = (
        SOURCE_ROOT / "solvers" / "events" / "augmentation.py"
    ).read_text()

    assert "problem.validate_states(initial_states)" in source
    assert "_validated_states" not in source


def test_event_solver_aggregate_uses_the_owning_package() -> None:
    imports = imported_modules(SOURCE_ROOT / "event_solver.py")

    assert ".event_augmented" not in imports
    assert ".solvers.events" in imports
