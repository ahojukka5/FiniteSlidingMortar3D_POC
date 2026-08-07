from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = (
    ROOT / "examples/contact_patch/model.py",
    ROOT / "examples/contact_patch/run.py",
    ROOT / "examples/sandwiched_beam/run.py",
    ROOT / "examples/rotating_blocks/model.py",
    ROOT / "examples/rotating_blocks/run.py",
)

FORBIDDEN_FLAT_MODULES = {
    "contact3d.adaptive_model",
    "contact3d.adaptive_options",
    "contact3d.adaptive_solver",
    "contact3d.enforcement_state",
    "contact3d.equilibrium",
    "contact3d.event_solver",
    "contact3d.linear_solver",
    "contact3d.scaled_solver",
}
FORBIDDEN_ROOT_SOLVER_NAMES = {
    "AdaptiveAcceptedStep",
    "AdaptiveContactOptions",
    "AdaptiveLoadOptions",
    "AdaptivePenaltyOptions",
    "AugmentedContactOptions",
    "LinearSolverOptions",
    "NewtonOptions",
    "ScaleAwareConvergenceOptions",
    "solve_adaptive_contact_path",
    "solve_equilibrium",
    "solve_event_aware_adaptive_contact_path",
    "solve_scale_aware_augmented_contact",
}
REQUIRED_OWNER_IMPORTS = {
    "examples/contact_patch/model.py": {
        "contact3d.scaling",
        "contact3d.solvers",
    },
    "examples/contact_patch/run.py": {"contact3d.solvers"},
    "examples/sandwiched_beam/run.py": {
        "contact3d.mortar.enforcement",
        "contact3d.scaling",
        "contact3d.solvers",
    },
    "examples/rotating_blocks/model.py": {
        "contact3d.scaling",
        "contact3d.solvers",
    },
    "examples/rotating_blocks/run.py": {"contact3d.solvers.events"},
}


def _imports(path: Path) -> tuple[ast.ImportFrom, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))


def test_examples_do_not_use_solver_compatibility_facades() -> None:
    for path in EXAMPLE_FILES:
        imports = _imports(path)
        flat_modules = {node.module for node in imports} & FORBIDDEN_FLAT_MODULES
        assert not flat_modules, f"{path}: flat solver imports {sorted(flat_modules)}"

        root_names = {
            alias.name
            for node in imports
            if node.module == "contact3d"
            for alias in node.names
        }
        leaked = root_names & FORBIDDEN_ROOT_SOLVER_NAMES
        assert not leaked, f"{path}: package-root solver imports {sorted(leaked)}"


def test_examples_import_solver_configuration_from_owners() -> None:
    for relative, expected in REQUIRED_OWNER_IMPORTS.items():
        modules = {node.module for node in _imports(ROOT / relative)}
        missing = expected - modules
        assert not missing, f"{relative}: missing owner imports {sorted(missing)}"
