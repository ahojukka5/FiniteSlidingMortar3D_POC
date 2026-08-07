"""Compatibility façade for the scale-aware augmented contact solver.

New code should import these records and algorithms from :mod:`contact3d.solvers`.
This module retains the historical path until the compatibility cleanup in #136.
"""

from .solvers.scaling import (
    ScaleAwareAugmentationIteration,
    ScaleAwareAugmentedContactResult,
    ScaleAwareNewtonIteration,
    solve_scale_aware_augmented_contact,
)

__all__ = [
    "ScaleAwareAugmentationIteration",
    "ScaleAwareAugmentedContactResult",
    "ScaleAwareNewtonIteration",
    "solve_scale_aware_augmented_contact",
]
