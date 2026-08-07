"""Architecture checks for the coupling subsystem."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import contact3d.coupled as legacy_coupled
from contact3d import coupling
from contact3d.coupling import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
    CoupledContactInterface,
    CoupledEquilibriumProblem,
    evaluate_coupled_equilibrium,
)
from contact3d.mechanics import (
    DeadLoad,
    DirichletConstraints,
    NeoHookeanMaterial,
    Tet4Mesh,
)
from contact3d.mortar.enforcement import AugmentedLagrangeState, KKTDiagnostics

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "contact3d"
COUPLING_ROOT = SOURCE_ROOT / "coupling"
COUPLING_EXPORTS = {
    "ContactBranchSignature",
    "ContactInterfaceEvaluation",
    "ContactInterfaceUpdate",
    "CoupledContactInterface",
    "CoupledEquilibriumEvaluation",
    "CoupledEquilibriumProblem",
    "MortarContactInterface",
    "evaluate_coupled_equilibrium",
}
FORBIDDEN_DEPENDENCIES = {
    "adaptive",
    "adaptive_model",
    "adaptive_options",
    "adaptive_solver",
    "benchmark_artifacts",
    "benchmark_goldens",
    "benchmark_plots",
    "coupled",
    "equilibrium",
    "event_adaptive",
    "event_augmented",
    "event_geometry",
    "event_model",
    "event_newton",
    "event_scaled",
    "linear_solver",
    "load_path",
    "rigid_path",
    "scaled_solver",
    "staged_rigid_path",
    "topology_scan",
    "verification_models",
}


def parsed_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(parsed_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def imported_names(path: Path, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(parsed_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_legacy_coupled_module_preserves_coupling_identities() -> None:
    for name in COUPLING_EXPORTS:
        assert getattr(legacy_coupled, name) is getattr(coupling, name)


def test_coupling_package_owns_models_and_assembly() -> None:
    definitions: dict[str, str] = {}
    for path in COUPLING_ROOT.glob("*.py"):
        for node in parsed_tree(path).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions[node.name] = path.name

    assert definitions["CoupledContactInterface"] == "protocols.py"
    assert definitions["MortarContactInterface"] == "interface.py"
    assert definitions["CoupledEquilibriumProblem"] == "problem.py"
    assert definitions["CoupledEquilibriumEvaluation"] == "results.py"
    assert definitions["evaluate_coupled_equilibrium"] == "assembly.py"


def test_coupling_has_no_solver_or_orchestration_dependencies() -> None:
    violations: dict[str, set[str]] = {}
    for path in COUPLING_ROOT.glob("*.py"):
        forbidden = imported_modules(path).intersection(FORBIDDEN_DEPENDENCIES)
        if forbidden:
            violations[path.name] = forbidden

    assert violations == {}


def test_coupling_uses_public_subsystem_boundaries() -> None:
    violations: dict[str, set[str]] = {}
    for path in COUPLING_ROOT.glob("*.py"):
        private = {
            module
            for module in imported_modules(path)
            if module.startswith("mechanics.")
            or module.startswith("mortar.model")
            or module.startswith("mortar.enforcement.")
        }
        if private:
            violations[path.name] = private

    assert violations == {}


def test_legacy_module_imports_coupling_public_api() -> None:
    path = SOURCE_ROOT / "coupled.py"
    tree = parsed_tree(path)
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    assert COUPLING_EXPORTS.issubset(imported_names(path, "coupling"))
    assert class_names.isdisjoint(
        {
            "ContactBranchSignature",
            "ContactInterfaceEvaluation",
            "ContactInterfaceUpdate",
            "CoupledContactInterface",
            "CoupledEquilibriumEvaluation",
            "CoupledEquilibriumProblem",
            "MortarContactInterface",
        }
    )


class ZeroContactInterface:
    """Minimal mapped interface used to test coupling without continuation."""

    def __init__(self) -> None:
        self._dofs = np.array([0, 1, 2], dtype=np.int64)

    @property
    def dofs(self) -> np.ndarray:
        return self._dofs

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)

    def evaluate(
        self,
        displacement: np.ndarray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation:
        del displacement, state, tolerance
        zeros = np.zeros(1)
        diagnostics = KKTDiagnostics(zeros, zeros, zeros, zeros, zeros)
        return ContactInterfaceEvaluation(
            residual=np.zeros(3),
            diagnostics=diagnostics,
            signature=ContactBranchSignature((), (False,), (True,)),
            normal_gaps=zeros,
            pressure=zeros,
            raw=None,
        )

    def tangent(
        self,
        displacement: np.ndarray,
        state: AugmentedLagrangeState,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> np.ndarray:
        del displacement, state, evaluation, tolerance
        return np.zeros((3, 3))

    def augment(
        self,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> ContactInterfaceUpdate:
        del evaluation, tolerance
        state = self.initial_state()
        zeros = np.zeros(1)
        diagnostics = KKTDiagnostics(zeros, zeros, zeros, zeros, zeros)
        return ContactInterfaceUpdate(state, zeros, diagnostics)


def test_coupling_contracts_assemble_without_solver_orchestration() -> None:
    mesh = Tet4Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    interface = ZeroContactInterface()
    constraints = DirichletConstraints(np.arange(12), np.zeros(12))
    problem = CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(210.0, 0.3),
        constraints,
        DeadLoad(np.zeros(12)),
        (interface,),
    )

    assert isinstance(interface, CoupledContactInterface)
    evaluation = evaluate_coupled_equilibrium(
        problem,
        np.zeros(12),
        problem.initial_states(),
    )

    assert evaluation.free_residual_norm == 0.0
    assert evaluation.contacts[0].signature.supported_rows == (True,)
    with pytest.raises(ValueError, match="one multiplier state"):
        problem.validate_states(())
