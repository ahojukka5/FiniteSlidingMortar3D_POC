"""Compatibility façade for the scale-aware augmented contact solver.

New code should import these records and algorithms from :mod:`contact3d.solvers`.
This module retains the historical path until the compatibility cleanup in #136.
"""

from .coupling import CoupledEquilibriumProblem
from .solvers.scaling import (
    ScaleAwareAugmentationIteration,
    ScaleAwareAugmentedContactResult,
    ScaleAwareNewtonIteration,
    solve_scale_aware_augmented_contact,
)
from .solvers.scaling import (
    _all_kkt_converged as _all_kkt_converged,
)
from .solvers.scaling import (
    _augmentation_row as _augmentation_row,
)
from .solvers.scaling import (
    _scaled_newton_history as _scaled_newton_history,
)

_validated_states = CoupledEquilibriumProblem.validate_states

__all__ = [
    "ScaleAwareAugmentationIteration",
    "ScaleAwareAugmentedContactResult",
    "ScaleAwareNewtonIteration",
    "solve_scale_aware_augmented_contact",
]
