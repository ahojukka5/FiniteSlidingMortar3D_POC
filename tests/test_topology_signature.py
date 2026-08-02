from __future__ import annotations

import json

import pytest

from contact3d.topology_model import ContactTopologySignature
from contact3d.topology_signature import (
    TOPOLOGY_SEQUENCE_SCHEMA,
    TOPOLOGY_SIGNATURE_SCHEMA,
    canonical_topology_json,
    canonical_topology_sequence,
    canonical_topology_signature,
    topology_sequence_hash,
    topology_signature_hash,
    validate_topology_signature_record,
)


def _signature(*, vertices: int = 4, pallets: int = 4) -> ContactTopologySignature:
    return ContactTopologySignature(
        ((0, 3), (1, 4)),
        (True, False, True),
        (True, True, True),
        ((0, 3, vertices, pallets, 1), (1, 4, 5, 5, 1)),
    )


def test_canonical_signature_is_versioned_and_complete() -> None:
    record = canonical_topology_signature(_signature())

    assert record == {
        "schema_version": TOPOLOGY_SIGNATURE_SCHEMA,
        "facet_pairs": [[0, 3], [1, 4]],
        "geometry_tokens": [[0, 3, 4, 4, 1], [1, 4, 5, 5, 1]],
        "supported_rows": [True, True, True],
        "active_rows": [True, False, True],
    }
    validate_topology_signature_record(record)


def test_signature_digest_has_a_fixed_cross_process_value() -> None:
    encoded = canonical_topology_json(_signature())

    assert encoded == (
        '{"active_rows":[true,false,true],"facet_pairs":[[0,3],[1,4]],'
        '"geometry_tokens":[[0,3,4,4,1],[1,4,5,5,1]],'
        '"schema_version":"contact3d-topology-signature/v1",'
        '"supported_rows":[true,true,true]}'
    )
    assert topology_signature_hash(_signature()) == (
        "06f3bb089ff959c84b0e41f27f60ce4be4bd646e7b573297ee80cd4b298eca39"
    )


def test_equal_pair_counts_do_not_hide_clipping_or_pallet_changes() -> None:
    baseline = _signature()
    clipping_change = _signature(vertices=5)
    pallet_change = _signature(pallets=6)

    assert len(baseline.facet_pairs) == len(clipping_change.facet_pairs)
    assert topology_signature_hash(baseline) != topology_signature_hash(clipping_change)
    assert topology_signature_hash(baseline) != topology_signature_hash(pallet_change)


def test_sequence_hash_retains_frame_and_interface_order() -> None:
    first = _signature()
    second = _signature(vertices=5)
    records = ((0.0, (first,)), (0.5, (second,)))
    payload = canonical_topology_sequence(records)

    assert payload["schema_version"] == TOPOLOGY_SEQUENCE_SCHEMA
    assert topology_sequence_hash(records) == topology_sequence_hash(records)
    assert topology_sequence_hash(records) != topology_sequence_hash(
        ((0.0, (second,)), (0.5, (first,)))
    )


def test_canonical_record_round_trips_through_strict_json() -> None:
    record = canonical_topology_signature(_signature())
    decoded = json.loads(json.dumps(record, allow_nan=False))

    validate_topology_signature_record(decoded)
    assert decoded == record


def test_validation_rejects_incomplete_or_misaligned_records() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_topology_signature_record({})

    record = canonical_topology_signature(_signature())
    record["active_rows"] = [True]
    with pytest.raises(ValueError, match="equal length"):
        validate_topology_signature_record(record)


def test_sequence_requires_strictly_increasing_parameters() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        canonical_topology_sequence(((0.5, (_signature(),)), (0.5, (_signature(),))))
