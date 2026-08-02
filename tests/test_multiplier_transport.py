from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.event_model import (
    EventAwareAugmentedContactResult,
    EventAwareCoupledNewtonResult,
)
from contact3d.multiplier_transport import transport_multiplier_states
from contact3d.topology_model import ContactTopologySignature


def _signature(
    supported: tuple[bool, ...],
    active: tuple[bool, ...] | None = None,
) -> ContactTopologySignature:
    active = supported if active is None else active
    return ContactTopologySignature((), active, supported)


def test_released_rows_are_zeroed_and_persistent_rows_are_retained() -> None:
    state = AugmentedLagrangeState(np.array([2.0, 3.0, 5.0]), augmentation=4)

    states, records = transport_multiplier_states(
        (state,),
        (_signature((True, True, False)),),
        (_signature((True, False, True)),),
    )

    assert np.array_equal(states[0].multipliers, np.array([2.0, 0.0, 0.0]))
    assert states[0].augmentation == 4
    assert len(records) == 1
    record = records[0]
    assert record.released_rows == (1,)
    assert record.activated_rows == (2,)
    assert record.changed_rows == (1, 2)
    assert record.values_before == (3.0, 5.0)
    assert record.values_after == (0.0, 0.0)
    assert record.maximum_unsupported_before == 3.0
    assert record.maximum_unsupported_after == 0.0


def test_new_support_is_zero_initialized_during_pressure_transition() -> None:
    state = AugmentedLagrangeState(np.array([7.0, 11.0]))
    left = _signature((True, False), (True, False))
    right = _signature((False, True), (False, True))

    states, records = transport_multiplier_states((state,), (left,), (right,))

    assert np.array_equal(states[0].multipliers, np.zeros(2))
    assert records[0].released_rows == (0,)
    assert records[0].activated_rows == (1,)
    assert records[0].initialization_rule == "zero"


def test_unchanged_support_produces_no_transport_record() -> None:
    state = AugmentedLagrangeState(np.array([1.0, 2.0]))
    signature = _signature((True, True), (True, False))

    states, records = transport_multiplier_states(
        (state,),
        (signature,),
        (_signature((True, True), (False, True)),),
    )

    assert np.array_equal(states[0].multipliers, state.multipliers)
    assert records == ()


def test_multiple_interfaces_are_transported_independently() -> None:
    states, records = transport_multiplier_states(
        (
            AugmentedLagrangeState(np.array([2.0, 4.0])),
            AugmentedLagrangeState(np.array([3.0])),
        ),
        (
            _signature((True, True)),
            _signature((False,)),
        ),
        (
            _signature((False, True)),
            _signature((True,)),
        ),
    )

    assert np.array_equal(states[0].multipliers, np.array([0.0, 4.0]))
    assert np.array_equal(states[1].multipliers, np.array([0.0]))
    assert tuple(record.interface for record in records) == (0, 1)


def test_transport_rows_are_strict_json_compatible() -> None:
    _, records = transport_multiplier_states(
        (AugmentedLagrangeState(np.array([2.0])),),
        (_signature((True,)),),
        (_signature((False,)),),
    )

    payload = records[0].as_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["maximum_unsupported_after"] == 0.0


def test_transport_validation_rejects_misaligned_inputs() -> None:
    state = AugmentedLagrangeState(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="equal length"):
        transport_multiplier_states((state,), (), ())
    with pytest.raises(ValueError, match="match multiplier"):
        transport_multiplier_states(
            (state,),
            (_signature((True,)),),
            (_signature((True,)),),
        )


def test_event_results_expose_transport_history() -> None:
    states, records = transport_multiplier_states(
        (AugmentedLagrangeState(np.array([6.0])),),
        (_signature((True,)),),
        (_signature((False,)),),
    )
    equilibrium = EventAwareCoupledNewtonResult(
        np.zeros(3),
        1.0,
        True,
        "converged",
        SimpleNamespace(),
        (),
        (),
        None,
        states,
        records,
    )
    result = EventAwareAugmentedContactResult(
        equilibrium.displacement,
        states,
        True,
        "converged",
        equilibrium,
        (equilibrium,),
        (),
    )

    assert equilibrium.multiplier_transport_count == 1
    assert equilibrium.multiplier_transport_rows()[0]["released_rows"] == [0]
    assert result.multiplier_transports == records
