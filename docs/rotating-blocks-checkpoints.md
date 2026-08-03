# Rotating-blocks physical checkpoints

The rotating-blocks result bundle exports a bounded set of physical states so
that contact, overlap, pressure, reactions, and bulk deformation can be traced
through the staged motion. Checkpoint selection is based on accepted nonlinear
states, not on retry attempts or plotting samples.

## Requested regimes

The bundle requests the following checkpoints in this fixed order:

1. `pre-contact`: the undeformed reference state at continuation parameter
   zero, evaluated before the prescribed motion starts.
2. `first-contact`: the earliest accepted state with positive projected
   overlap, at least one supported row, and active contact pressure.
3. `compressed`: the accepted state nearest the compression-phase boundary.
4. `quarter-rotation`: the accepted rotation-phase state nearest 25 percent of
   the prescribed rotation interval.
5. `half-rotation`: the accepted rotation-phase state nearest 50 percent of the
   prescribed rotation interval.
6. `final`: the accepted state at the completed prescribed motion.

For compression end parameter `c` and final continuation parameter `e`, the
rotation targets are

```text
quarter = c + 0.25 * (e - c)
half    = c + 0.50 * (e - c)
```

The nearest-state rule orders candidates first by absolute target error, then
by continuation parameter, then by accepted-step number. This makes ties
stable under repeated artifact generation. The selected accepted-step number,
target parameter, selected parameter, selection error, and textual selection
rule are retained in `tables/checkpoints.csv` and `summary.json`.

## Missing regimes

A requested regime is never silently dropped. The checkpoint table always
contains all six requests. An unavailable regime has `present=false`, empty
selected-state fields, and a nonempty `missing_reason`.

Typical missing cases are:

- no accepted state contains active contact pressure;
- no accepted rotation-phase state exists;
- the accepted path does not reach the prescribed final parameter.

The complete result bundle fails its `checkpoint_regimes_complete` criterion
when any requested regime is missing.

## Artifact layout

Available checkpoints are exported in request order using deterministic
zero-based ordinals:

```text
checkpoints/00-pre-contact/
checkpoints/01-first-contact/
checkpoints/02-compressed/
checkpoints/03-quarter-rotation/
checkpoints/04-half-rotation/
checkpoints/05-final/
```

Each directory contains:

- `volume.vtu` with displacement, reaction, external load, contact force,
  element Jacobian, and energy-density fields;
- `slave-contact.vtp` with gap, pressure, multiplier, support, activity, and
  contact-force fields;
- `master-contact.vtp` with contact force and accumulated overlap area;
- `projected-overlap.vtp` with slave, master, and clipped projected regions.

The artifact manifest validates every exported file. Checkpoint names and
ordinals depend only on the requested physical regimes, so path refinement does
not rename the result directories.

## Evidence boundary

The checkpoint selection records accepted states produced by the current
frictionless normal-contact benchmark and its event-localized continuation
policy. It does not replace load-step refinement, contact-retention, force and
moment balance, pressure-redistribution, or mesh-quality evidence. Those
criteria remain separate parts of the benchmark acceptance gate.
