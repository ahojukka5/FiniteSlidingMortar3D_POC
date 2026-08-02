# Contact-topology signature artifacts

Facet-pair counts are not sufficient to identify a contact branch. Two states can
contain the same number of integrated pairs while differing in pair identities,
clipping-polygon topology, centroid-fan pallets, mortar-row support, or unilateral
pressure activity.

The topology signature artifact therefore retains all discrete branch data:

- ordered `(slave facet, master facet)` pairs;
- ordered geometry tokens containing pair identity, intersection-polygon vertex
  count, pallet count, and signed orientation;
- one support flag for every slave mortar row;
- one pressure-activity flag for every slave mortar row.

## Canonical representation

`canonical_topology_signature` returns the versioned
`contact3d-topology-signature/v1` object. Integer tuples become JSON arrays,
boolean row vectors remain booleans, and field names are emitted through strict
JSON with sorted keys and compact separators.

The serializer does not sort facet pairs or geometry tokens. Their existing
production order is part of the signature. This makes an accidental ordering
change visible rather than silently normalizing it away.

`topology_signature_hash` computes SHA-256 over the ASCII canonical JSON. The hash
is stable across Python processes and platforms because it does not use Python's
runtime hash function, floating-point locale formatting, object identity, or
unordered containers.

## Path histories

`canonical_topology_sequence` stores strictly increasing continuation parameters
using 17-digit formatting and preserves both frame order and interface order. Its
schema is `contact3d-topology-signature-sequence/v1`.

`topology_sequence_hash` is suitable for:

- deterministic repetition checks;
- quick comparisons before a field-by-field diagnostic;
- identifying the left, event, and selected-right histories in event records;
- comparing coarse, medium, and fine rotating-blocks paths.

A hash mismatch is only a detection mechanism. Human-readable decomposed fields
must remain beside the digest so the first differing pair, geometry token, support
row, or active row can be reported.

## Versioning boundary

Changing field meaning, field coverage, ordering semantics, or parameter encoding
requires a new schema version. Adding continuous quantities such as overlap area,
pressure, reaction, or timing does not belong in this discrete topology hash;
those values use explicit numeric tolerances in benchmark comparison tables.
