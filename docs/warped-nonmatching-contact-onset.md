# Warped nonmatching contact-onset benchmark

## Purpose

This benchmark is the first boundary-value problem in the repository that combines the
finite-strain bulk solver with the complete production moving-overlap mortar interface.
It replaces the frozen matching operator used by the earlier mixed-path harness.

The test is intentionally small enough for dense verification linear algebra, but it is
not a contact-law surrogate. Every accepted state is assembled through the same
broad-phase, projection, clipping, pallet, inverse-map, moving-operator, augmented
Lagrange, Newton, and adaptive-continuation code used by the general interface.

## Geometry

Two cube-star `TET4` solids contain nine nodes and twelve tetrahedra each. Their opposing
surfaces are deliberately incompatible:

- the upper slave surface is one warped `QUAD4` with a downward nominal normal;
- the lower master surface is two warped `TRI3` facets;
- the slave footprint is translated and skewed relative to the master square, producing
  partial overlap without projected vertex or edge coincidences;
- both surfaces have nonconstant reference `z` coordinates;
- the minimum initial normal clearance is positive;
- every reference tetrahedron has positive orientation.

The upper body is driven by a proportional mixed path. Its top face receives a final
prescribed translation of `(0.04, 0, -0.09)`, while a small horizontal dead load is
applied to its interior node. The lower bottom face is fixed.

## Solver path

The continuation uses fixed increments of at most `0.1`, adaptive cutback, contact-event
restarts, projected multiplier updates, scale-aware convergence measures, and bounded
interface-local penalty adaptation. The broad-phase distance is large enough to retain
the candidate pair before contact, so the path records a separated regime, the first
active state, and established compression.

The benchmark fails rather than writing a successful result if any of these conditions
is absent:

- the continuation does not reach parameter one;
- no separated state is accepted;
- no active contact state is accepted;
- a directional derivative crosses a discrete contact branch;
- a requested result table would be empty.

## Directional tangent checks

Three accepted smooth states are checked independently:

1. the last separated state;
2. the first accepted active-contact state;
3. the final established-contact state.

For a deterministic free-DOF direction `p`, the centered-difference oracle is

```text
[r(u + h p) - r(u - h p)] / (2 h).
```

The analytical comparison uses the complete global CSR tangent. Both perturbed states
must retain the same facet-pair, support, and active-row signature as the base state.
This explicitly excludes the exact unilateral kink and clipping topology events from a
smooth derivative claim.

## Recorded diagnostics

`accepted-steps.csv` contains:

- prescribed tool motion and constrained reactions;
- dimensional and normalized equilibrium residuals;
- minimum element Jacobian;
- dimensional and normalized penetration;
- maximum pressure;
- facet-pair count and overlap area;
- mortar partition consistency;
- supported and active row counts;
- Newton iterations, augmentations, event restarts, and the current penalty.

`interface-rows.csv` records every nodal gap, pressure, support flag, and activity flag.
`attempt-history.csv` retains accepted, cut-back, and penalty-retry attempts with
normalized measures and update reasons. `tangent-checks.csv` contains the three
independent directional errors.

A successful reference run generates:

- reference and final deformed meshes in the `x-z` plane;
- final nodal pressure;
- final projected slave, master, and intersection polygons;
- constrained reaction history;
- normalized equilibrium and penetration histories.

## Running

```bash
uv run python benchmarks/warped_nonmatching_contact_onset.py \
  --output results/warped-nonmatching-contact-onset
```

The benchmark is deterministic. It uses fixed geometry, solver settings, continuation
increments, and random seeds for the directional checks.

## Formulation boundary

The benchmark validates smooth moving-overlap contact and explicit outer restarts when a
contact branch changes. It does not provide generalized derivatives for projected
vertex-on-edge, edge-on-edge, zero-area, or appearing/disappearing-pallet events. Those
states remain the scope of the topology-event state machine.

## Review status

The benchmark implementation, geometry preflight, deterministic output pipeline, and
focused derivative tests are included in this change. The complete production solve must
be run against the full repository before issue #17 is closed and its reference result
files are committed. This boundary prevents preflight or mocked data from being presented
as a mechanics result.
