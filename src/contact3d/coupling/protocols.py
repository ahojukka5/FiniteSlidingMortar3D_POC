"""Public contracts for mapped contact interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..mechanics import FloatArray, IntArray
from ..mortar.enforcement import AugmentedLagrangeState
from .results import ContactInterfaceEvaluation, ContactInterfaceUpdate


@runtime_checkable
class CoupledContactInterface(Protocol):
    """Contact interface contract consumed by coupled assembly and solvers."""

    @property
    def dofs(self) -> IntArray: ...

    def initial_state(self) -> AugmentedLagrangeState: ...

    def evaluate(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation: ...

    def tangent(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> FloatArray: ...

    def augment(
        self,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> ContactInterfaceUpdate: ...
