# Rotating-blocks contact-retention monitor

The contact-retention monitor applies only after the compression phase, when the
prescribed path is rotating the upper block. Compression establishes contact; the
monitor then determines whether each accepted rotation state still represents the
same load-bearing contact problem.

## Recorded fields

Every accepted rotation state records:

- continuation and phase parameters and the prescribed rotation angle;
- projected overlap area;
- supported and active slave-row counts;
- the integrated normal reaction, computed as the mortar pressure dotted with the
  mortar row areas;
- the maximum reported normal gap and maximum separation;
- the canonical SHA-256 topology-signature hash; and
- the preceding and following accepted signature hashes.

The package uses the penetration-positive gap convention. `maximum_gap` is
therefore the largest positive penetration value. `maximum_separation` is the
largest value of `-normal_gap`. Both are diagnostic histories; contact retention
is decided from overlap, support, activity, and normal reaction rather than from a
single nodal separation value.

## Retained state

A normal accepted rotation state must satisfy both structural and load-bearing
conditions.

Structural contact requires:

```text
overlap_area >= 1e-12
supported_rows >= 1
```

Load-bearing contact requires:

```text
active_rows >= 1
normal_reaction >= 1e-12
```

These thresholds are versioned in `RetentionThresholds`. They are identical for
the quick and full physical models. The profile-dependent quantity is the largest
accepted interval that can represent an isolated localized event.

## Localized transition exception

An accepted state with zero active rows or negligible normal reaction may be
classified as `localized_transition` only when all of the following hold:

1. overlap and mortar support remain present at the state;
2. the immediately preceding and following accepted states are structurally
   supported and load bearing;
3. a localized topology event lies between those neighboring parameters; and
4. the complete neighboring interval is no larger than two requested path
   increments.

The exception can therefore cover one isolated accepted state at a localized
branch transition. Two consecutive non-load-bearing states are sustained contact
loss and fail. A state with no projected overlap or no supported mortar row cannot
use the exception.

## Failure evidence

A failed row records all applicable reasons, including missing overlap, missing
support, missing active rows, negligible reaction, lack of neighboring retained
states, missing event localization, or an excessive bracket interval. The first
failure in the summary includes the preceding and following topology-signature
hashes so the divergent branches can be inspected directly.

## Acceptance and artifacts

The complete result bundle writes:

- `tables/contact-retention.csv`;
- `contact-retention.json`;
- `plots/contact-retention-metrics.svg`; and
- `plots/contact-retention-status.svg`.

The monitor adds `contact_retention_monitor` to the common acceptance-gate table.
A retention failure is reported together with the observed state, failure reasons,
and neighboring signature hashes. The bundle command writes all artifacts before
raising its final nonzero error.

This monitor demonstrates frictionless normal-contact retention for the current
compression-then-rotation path. It does not establish frictional stick/slip
retention or assign a generalized derivative at exact edge-on-edge or on-vertex
special states; those are separate regression topics.
