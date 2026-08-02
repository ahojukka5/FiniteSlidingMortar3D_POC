"""Versioned canonical serialization for contact-topology signatures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Protocol

TOPOLOGY_SIGNATURE_SCHEMA = "contact3d-topology-signature/v1"
TOPOLOGY_SEQUENCE_SCHEMA = "contact3d-topology-signature-sequence/v1"


class TopologySignatureLike(Protocol):
    """Discrete contact branch accepted by the canonical serializer."""

    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]
    geometry_tokens: tuple[tuple[int, int, int, int, int], ...]


def _ordered_pairs(values: object) -> list[list[int]]:
    return [[int(left), int(right)] for left, right in values]  # type: ignore[misc]


def _ordered_tokens(values: object) -> list[list[int]]:
    return [[int(component) for component in token] for token in values]  # type: ignore[misc]


def canonical_topology_signature(
    signature: TopologySignatureLike,
) -> dict[str, object]:
    """Return the complete ordered v1 representation of one interface branch."""

    pairs = _ordered_pairs(signature.facet_pairs)
    active = [bool(value) for value in signature.active_rows]
    supported = [bool(value) for value in signature.supported_rows]
    tokens = _ordered_tokens(getattr(signature, "geometry_tokens", ()))
    if len(active) != len(supported):
        raise ValueError("active and supported topology rows must have equal length")
    if any(len(token) != 5 for token in tokens):
        raise ValueError("geometry topology tokens must contain five integers")
    return {
        "schema_version": TOPOLOGY_SIGNATURE_SCHEMA,
        "facet_pairs": pairs,
        "geometry_tokens": tokens,
        "supported_rows": supported,
        "active_rows": active,
    }


def canonical_topology_json(signature: TopologySignatureLike) -> str:
    """Encode one topology signature as platform-independent canonical JSON."""

    return json.dumps(
        canonical_topology_signature(signature),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def topology_signature_hash(signature: TopologySignatureLike) -> str:
    """Return the stable SHA-256 digest of one complete topology signature."""

    return hashlib.sha256(canonical_topology_json(signature).encode("ascii")).hexdigest()


def canonical_topology_sequence(
    records: Iterable[tuple[float, Iterable[TopologySignatureLike]]],
) -> dict[str, object]:
    """Return a versioned ordered sequence used by path and repetition checks."""

    frames: list[dict[str, object]] = []
    previous: float | None = None
    for parameter, signatures in records:
        value = float(parameter)
        if previous is not None and value <= previous:
            raise ValueError("topology sequence parameters must be strictly increasing")
        previous = value
        frames.append(
            {
                "parameter": format(value, ".17g"),
                "interfaces": [
                    canonical_topology_signature(signature)
                    for signature in signatures
                ],
            }
        )
    if not frames:
        raise ValueError("topology sequence must contain at least one frame")
    return {
        "schema_version": TOPOLOGY_SEQUENCE_SCHEMA,
        "frames": frames,
    }


def topology_sequence_hash(
    records: Iterable[tuple[float, Iterable[TopologySignatureLike]]],
) -> str:
    """Return the stable SHA-256 digest of an ordered topology history."""

    payload = canonical_topology_sequence(records)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def validate_topology_signature_record(record: Mapping[str, object]) -> None:
    """Validate a decoded canonical v1 signature artifact."""

    if record.get("schema_version") != TOPOLOGY_SIGNATURE_SCHEMA:
        raise ValueError("unsupported topology signature schema")
    required = {"facet_pairs", "geometry_tokens", "supported_rows", "active_rows"}
    if not required.issubset(record):
        missing = ", ".join(sorted(required - set(record)))
        raise ValueError(f"topology signature record is missing fields: {missing}")
    active = record["active_rows"]
    supported = record["supported_rows"]
    if not isinstance(active, list) or not isinstance(supported, list):
        raise ValueError("topology row fields must be lists")
    if len(active) != len(supported):
        raise ValueError("active and supported topology rows must have equal length")
