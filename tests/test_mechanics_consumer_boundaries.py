"""Architecture checks for mechanics consumers."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "contact3d"


def parsed_imports(path: Path) -> tuple[ast.Import | ast.ImportFrom, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in parsed_imports(path):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
        else:
            modules.update(alias.name for alias in node.names)
    return modules


def imported_names(path: Path, module: str) -> set[str]:
    return {
        alias.name
        for node in parsed_imports(path)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_external_consumers_use_mechanics_public_api() -> None:
    violations: dict[str, set[str]] = {}
    mechanics_root = SOURCE_ROOT / "mechanics"
    for path in SOURCE_ROOT.rglob("*.py"):
        if mechanics_root in path.parents:
            continue
        private_imports = {
            module
            for module in imported_modules(path)
            if module.startswith("mechanics.")
            or module.startswith("contact3d.mechanics.")
        }
        if private_imports:
            violations[str(path.relative_to(SOURCE_ROOT))] = private_imports

    assert violations == {}


def test_equilibrium_consumes_mechanics_package_boundary() -> None:
    path = SOURCE_ROOT / "equilibrium.py"
    imports = imported_modules(path)

    assert "mechanics" in imports
    assert not imports.intersection(
        {"bulk_material", "bulk_sparse", "mechanics.model", "model", "sparse", "tet4"}
    )
    assert {
        "BulkGeometryError",
        "DeadLoad",
        "DirichletConstraints",
        "EquilibriumEvaluation",
        "EquilibriumProblem",
        "FloatArray",
        "evaluate_equilibrium",
    }.issubset(imported_names(path, "mechanics"))


def test_linear_solver_consumes_mechanics_storage_boundary() -> None:
    path = SOURCE_ROOT / "linear_solver.py"
    imports = imported_modules(path)

    assert "mechanics" in imports
    assert not imports.intersection({"mechanics.model", "model", "sparse"})
    assert {"CSRMatrix", "FloatArray", "IntArray"}.issubset(
        imported_names(path, "mechanics")
    )


def test_coupled_equilibrium_separates_mechanics_and_solver_ownership() -> None:
    path = SOURCE_ROOT / "coupled.py"
    imports = imported_modules(path)

    assert "mechanics" in imports
    assert not imports.intersection(
        {"bulk_material", "bulk_sparse", "mechanics.model", "model", "sparse", "tet4"}
    )
    assert {"DeadLoad", "DirichletConstraints", "FloatArray", "IntArray"}.issubset(
        imported_names(path, "mechanics")
    )
    assert imported_names(path, "equilibrium") == {"NewtonOptions"}


def test_bulk_facade_gathers_data_from_mechanics() -> None:
    path = SOURCE_ROOT / "bulk.py"
    imports = imported_modules(path)

    assert "mechanics" in imports
    assert not imports.intersection(
        {"bulk_material", "bulk_oracle", "bulk_sparse", "sparse", "tet4"}
    )
    mechanics_names = imported_names(path, "mechanics")
    assert {
        "DeadLoad",
        "DirichletConstraints",
        "EquilibriumEvaluation",
        "EquilibriumProblem",
        "evaluate_equilibrium",
    }.issubset(mechanics_names)
    assert not mechanics_names.isdisjoint({"NeoHookeanMaterial", "Tet4Mesh"})
    assert imported_names(path, "equilibrium") == {
        "NewtonIteration",
        "NewtonOptions",
        "NewtonResult",
        "solve_equilibrium",
        "solve_load_steps",
    }


def test_boundary_paths_consume_mechanics_data_contracts() -> None:
    for filename in ("load_path.py", "rigid_path.py"):
        path = SOURCE_ROOT / filename
        assert {"DeadLoad", "DirichletConstraints"}.issubset(
            imported_names(path, "mechanics")
        )
        assert "equilibrium" not in imported_modules(path)


def test_verification_models_use_mechanics_package_boundary() -> None:
    path = SOURCE_ROOT / "verification_models.py"

    assert {
        "DeadLoad",
        "DirichletConstraints",
        "NeoHookeanMaterial",
        "Tet4Mesh",
    }.issubset(imported_names(path, "mechanics"))
    assert not imported_modules(path).intersection(
        {"bulk_material", "equilibrium", "tet4"}
    )
