from __future__ import annotations

from contact3d.enforcement import (
    AugmentedLagrangeEvaluation as FlatAugmentedLagrangeEvaluation,
)
from contact3d.enforcement import AugmentedLagrangeState as FlatAugmentedLagrangeState
from contact3d.enforcement import AugmentedLagrangeUpdate as FlatAugmentedLagrangeUpdate
from contact3d.enforcement import KKTDiagnostics as FlatKKTDiagnostics
from contact3d.enforcement import augment_multipliers as flat_augment_multipliers
from contact3d.enforcement import (
    augmented_lagrange_contact_tangent as flat_augmented_lagrange_contact_tangent,
)
from contact3d.enforcement import (
    numerical_augmented_lagrange_tangent as flat_numerical_augmented_lagrange_tangent,
)
from contact3d.mortar.enforcement import (
    AugmentedLagrangeEvaluation,
    AugmentedLagrangeState,
    AugmentedLagrangeUpdate,
    KKTDiagnostics,
    augment_multipliers,
    augmented_lagrange_contact_tangent,
    numerical_augmented_lagrange_tangent,
)


def test_flat_enforcement_imports_are_temporary_direct_reexports() -> None:
    assert FlatAugmentedLagrangeState is AugmentedLagrangeState
    assert FlatKKTDiagnostics is KKTDiagnostics
    assert FlatAugmentedLagrangeEvaluation is AugmentedLagrangeEvaluation
    assert FlatAugmentedLagrangeUpdate is AugmentedLagrangeUpdate
    assert flat_augment_multipliers is augment_multipliers
    assert flat_augmented_lagrange_contact_tangent is augmented_lagrange_contact_tangent
    assert (
        flat_numerical_augmented_lagrange_tangent
        is numerical_augmented_lagrange_tangent
    )
