"""Ownership and dependency tests for mechanics equilibrium assembly."""

from __future__ import annotations

import ast
from pathlib import Path

import contact3d.equilibrium as flat_equilibrium
import contact3d.mechanics as mechanics
from contact3d.mechanics import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumEvaluation,
    EquilibriumProblem,
    evaluate_equilibrium,
)

SOURCE = (
    Path(__file__).parents[1]
    / "src"
    / "contact3d"
    / "mechanics"
    / "equilibrium.py"
)


def test_mechanics_public_api_owns_equilibrium_assembly() -> None:
    assert mechanics.DeadLoad is DeadLoad
    assert mechanics.DirichletConstraints is DirichletConstraints
    assert mechanics.EquilibriumEvaluation is EquilibriumEvaluation
    assert mechanics.EquilibriumProblem is EquilibriumProblem
    assert mechanics.evaluate_equilibrium is evaluate_equilibrium


def test_flat_equilibrium_module_reuses_moved_objects() -> None:
    assert flat_equilibrium.DeadLoad is DeadLoad
    assert flat_equilibrium.DirichletConstraints is DirichletConstraints
    assert flat_equilibrium.EquilibriumEvaluation is EquilibriumEvaluation
    assert flat_equilibrium.EquilibriumProblem is EquilibriumProblem
    assert flat_equilibrium.evaluate_equilibrium is evaluate_equilibrium


def test_mechanics_equilibrium_has_only_sibling_mechanics_dependencies() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
    }

    assert imports == {"bulk_material", "model", "sparse_tet4", "tet4"}
