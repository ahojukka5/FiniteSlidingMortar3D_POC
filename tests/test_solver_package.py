from __future__ import annotations

import ast
from pathlib import Path

import contact3d
import contact3d.coupled as legacy_coupled
import contact3d.coupling as coupling
import contact3d.equilibrium as equilibrium
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


def test_legacy_coupled_module_is_a_reexport_only_facade() -> None:
    tree = ast.parse((SOURCE_ROOT / "coupled.py").read_text())

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def test_legacy_coupling_exports_preserve_object_identity() -> None:
    coupling_names = (
        "ContactBranchSignature",
        "ContactInterfaceEvaluation",
        "ContactInterfaceUpdate",
        "CoupledContactInterface",
        "CoupledEquilibriumEvaluation",
        "CoupledEquilibriumProblem",
        "MortarContactInterface",
        "evaluate_coupled_equilibrium",
    )
    solver_names = (
        "AugmentationIteration",
        "AugmentedContactOptions",
        "AugmentedContactResult",
        "AugmentedTerminationReason",
        "ContactEventPolicy",
        "CoupledNewtonIteration",
        "CoupledNewtonResult",
        "CoupledTerminationReason",
        "solve_augmented_contact",
        "solve_coupled_equilibrium",
    )

    for name in coupling_names:
        assert getattr(legacy_coupled, name) is getattr(coupling, name)
    for name in solver_names:
        assert getattr(legacy_coupled, name) is getattr(solvers, name)


def test_newton_options_have_one_owner_during_solver_migration() -> None:
    assert solvers.NewtonOptions is equilibrium.NewtonOptions
    assert contact3d.NewtonOptions is solvers.NewtonOptions


def test_solver_package_has_no_application_or_verification_dependencies() -> None:
    forbidden = (
        "examples",
        "benchmarks",
        "benchmark_",
        "verification",
        "publication",
        "plot",
        "vtk",
        "golden",
    )
    violations: dict[str, set[str]] = {}
    solver_root = SOURCE_ROOT / "solvers"

    for path in solver_root.rglob("*.py"):
        imports = {
            module
            for module in imported_modules(path)
            if any(token in module for token in forbidden)
        }
        if imports:
            violations[str(path.relative_to(SOURCE_ROOT))] = imports

    assert violations == {}


def test_lower_level_subsystems_do_not_import_solvers() -> None:
    violations: dict[str, set[str]] = {}
    for subsystem in ("geometry", "mechanics", "mortar", "coupling"):
        for path in (SOURCE_ROOT / subsystem).rglob("*.py"):
            imports = {
                module
                for module in imported_modules(path)
                if module.startswith("solvers.")
                or module.startswith("contact3d.solvers")
                or module.startswith("..solvers")
            }
            if imports:
                violations[str(path.relative_to(SOURCE_ROOT))] = imports

    assert violations == {}
