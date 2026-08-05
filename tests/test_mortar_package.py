from __future__ import annotations

import numpy as np

from contact3d.contact import ContactPair as FlatContactPair
from contact3d.model import LocalMortarWeights as FlatLocalMortarWeights
from contact3d.mortar import (
    ContactPair,
    LocalMortarWeightLinearization,
    LocalMortarWeights,
    MortarWeightJacobian,
    analytical_mortar_weight_jacobian,
    integrate_facet_pair,
    integrate_facet_pair_linearized,
    moving_mortar_contact_tangent,
    numerical_mortar_weight_jacobian,
)
from contact3d.moving import MortarWeightJacobian as FlatMortarWeightJacobian
from contact3d.moving import (
    analytical_mortar_weight_jacobian as flat_analytical_mortar_weight_jacobian,
)
from contact3d.moving import (
    moving_mortar_contact_tangent as flat_moving_mortar_contact_tangent,
)
from contact3d.moving import (
    numerical_mortar_weight_jacobian as flat_numerical_mortar_weight_jacobian,
)
from contact3d.operators import (
    LocalMortarWeightLinearization as FlatLocalMortarWeightLinearization,
)
from contact3d.operators import (
    integrate_facet_pair_linearized as flat_integrate_facet_pair_linearized,
)
from contact3d.overlap import integrate_facet_pair as flat_integrate_facet_pair


def test_flat_mortar_imports_are_temporary_direct_reexports() -> None:
    assert FlatContactPair is ContactPair
    assert FlatLocalMortarWeights is LocalMortarWeights
    assert FlatLocalMortarWeightLinearization is LocalMortarWeightLinearization
    assert FlatMortarWeightJacobian is MortarWeightJacobian
    assert flat_integrate_facet_pair is integrate_facet_pair
    assert flat_integrate_facet_pair_linearized is integrate_facet_pair_linearized
    assert (
        flat_analytical_mortar_weight_jacobian
        is analytical_mortar_weight_jacobian
    )
    assert flat_moving_mortar_contact_tangent is moving_mortar_contact_tangent
    assert flat_numerical_mortar_weight_jacobian is numerical_mortar_weight_jacobian


def test_mortar_package_owns_local_operator_results() -> None:
    slave = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.05, 0.03],
            [1.1, 1.05, -0.02],
            [-0.05, 0.95, 0.04],
        ]
    )
    master = np.array(
        [
            [0.18, -0.12, -0.13],
            [1.35, -0.05, -0.08],
            [1.28, 0.86, -0.11],
            [0.12, 0.92, -0.16],
        ]
    )

    weights = integrate_facet_pair(slave, master)
    linearization = integrate_facet_pair_linearized(slave, master)

    assert isinstance(weights, LocalMortarWeights)
    assert isinstance(linearization, LocalMortarWeightLinearization)
    assert weights.overlap.area > 0.0
    np.testing.assert_allclose(linearization.d, weights.d, atol=2.0e-12)
    np.testing.assert_allclose(linearization.m, weights.m, atol=2.0e-12)
