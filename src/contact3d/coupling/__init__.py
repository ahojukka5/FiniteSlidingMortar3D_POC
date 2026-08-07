"""Coupling contracts, adapters, problem models, and global assembly."""

from .assembly import evaluate_coupled_equilibrium
from .interface import MortarContactInterface
from .problem import CoupledEquilibriumProblem
from .protocols import CoupledContactInterface
from .results import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
    CoupledEquilibriumEvaluation,
)

__all__ = [
    "ContactBranchSignature",
    "ContactInterfaceEvaluation",
    "ContactInterfaceUpdate",
    "CoupledContactInterface",
    "CoupledEquilibriumEvaluation",
    "CoupledEquilibriumProblem",
    "MortarContactInterface",
    "evaluate_coupled_equilibrium",
]
