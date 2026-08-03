# Rotating-blocks pressure redistribution

The rotating-blocks benchmark records the complete mortar pressure state at
every accepted continuation parameter. The evidence is limited to the current
frictionless normal-contact formulation and the event-localized continuation
policy used by the benchmark.

## Nodal state

Each accepted state records, for every slave node:

- current coordinates;
- mortar row area;
- normal gap;
- projected pressure;
- accepted augmented-Lagrange multiplier;
- geometric support and active-set flags; and
- the nodal pressure measure `pressure * row_area`.

Rows without geometric support must have zero pressure. Multiplier transport
also sets released and newly supported rows to zero before Newton restarts, so
accepted unsupported rows must have zero multiplier. Both conditions are
checked after normalization by the largest corresponding accepted-state value.

## Aggregate measures

Let `A_i` be the mortar row area and `p_i` the nodal pressure. The reported
normal-pressure resultant is

```text
P = sum_i p_i A_i.
```

The pressure centroid is the current-coordinate first moment divided by `P`.
It is undefined while `P` is numerically zero and is therefore omitted rather
than assigned an artificial coordinate. The supported-area mean, variance,
root-mean-square value, and area-weighted L2 norm are

```text
mean = sum_i p_i A_i / sum_i A_i
variance = sum_i A_i (p_i - mean)^2 / sum_i A_i
rms = sqrt(sum_i A_i p_i^2 / sum_i A_i)
L2 = sqrt(sum_i A_i p_i^2).
```

The scalar resultant is compared with the normal projection of the assembled
slave contact force. Tangential force and slave/master cancellation are also
reported to expose accidental friction or node-mapping errors.

## Continuity and topology events

Centroid motion is compared only between consecutive pressure-bearing states
whose parameter interval contains no localized topology event. Intervals that
cross pair, support, or activity events are marked and excluded from the smooth
continuity check. The remaining jump is normalized by the slave-surface
reference diagonal; the current diagnostic limit is `0.5`.

## Path refinement

The medium and fine continuation histories are interpolated to the same path
parameters. Aggregate resultants, moments, norms, variance, supported area, and
centroid coordinates are compared there. Pressure, multiplier, gap, and row
area are also compared node by node. Discrete support or activity mismatches
are permitted only within two fine-grid increments of a localized event.

Aggregate relative errors use the maximum magnitude of the fine history as the
scale. Centroid errors use the slave-surface characteristic length. The current
limits are five percent for aggregate fields and support area, and ten percent
for nodal pressure, multiplier, and gap histories.

## Artifacts

The result bundle contains:

- `tables/pressure-nodes.csv`;
- `tables/pressure-aggregates.csv`;
- `tables/refinement-pressure-nodes.csv`;
- `tables/refinement-pressure-aggregates.csv`;
- `pressure-summary.json`; and
- nodal, aggregate, centroid, and refinement SVG plots.

These artifacts complement the checkpoint VTK files. The CSV histories retain
every accepted state, while the VTK files remain a bounded set of physical
checkpoints for visual inspection.
