# Rotating-blocks result bundle

The rotating-blocks benchmark writes one versioned, manifest-validated result
directory. The command is:

```bash
uv run python benchmarks/rotating_blocks_bundle.py \
  --profile full \
  --output results/rotating-blocks
```

The full command executes the production adaptive contact solve and the configured
coarse/medium/fine path-refinement study before writing artifacts. The quick
profile uses the same physical model with its reduced execution settings.

The physical meaning and claim boundary of every metric are defined in the
[benchmark interpretation note](rotating-blocks-interpretation.md).

## Tables

The `tables/` directory contains accepted steps, adaptive attempts, complete
solver diagnostics, topology events, selected checkpoints, interface rows,
facet-pair overlap areas, projected overlap regions, and the refinement field
and event comparisons. Every CSV has a versioned schema entry in
`manifest.json`.

Interface rows contain normal gap, pressure, accepted multiplier, row area,
support, activity, and the three components of slave contact force. Facet-pair
and projected-region rows retain the checkpoint, path parameter, slave/master
facet indices, and overlap area.

## Checkpoints

Six deterministic physical regimes are requested:

- `pre-contact` is the reference state at path parameter zero;
- `first-contact` is the earliest accepted state with overlap, support, and
  active pressure;
- `compressed` is the accepted state nearest the compression-phase endpoint;
- `quarter-rotation` is the rotation-phase state nearest one quarter of the
  prescribed rotation;
- `half-rotation` is the rotation-phase state nearest one half of the prescribed
  rotation;
- `final` is the accepted state at the completed prescribed motion.

The checkpoint table records the selection rule, target parameter, selected
accepted-step index, selected parameter, selection error, maximum displacement,
maximum pressure, and overlap area. An unavailable regime is retained as an
explicit table row with a typed missing reason rather than being silently omitted.
Available checkpoint directories use stable ordinal-regime names, so requested
regimes retain deterministic paths when load-step resolution changes.

Each available checkpoint directory contains:

- `volume.vtu` with displacement, reactions, external load, contact force,
  element Jacobian, and energy density;
- `slave-contact.vtp` with gap, pressure, multiplier, support, activity, and
  contact force;
- `master-contact.vtp` with master contact force and integrated overlap area
  per master facet;
- `projected-overlap.vtp` with independently inspectable slave, master, and
  clipped intersection polygons.

The projected polygons are written in the local two-dimensional projection
plane used by the mortar overlap calculation. `region_kind` values are zero for
slave, one for master, and two for clipped intersection polygons.

The complete checkpoint selection contract is documented in the
[checkpoint note](rotating-blocks-checkpoints.md).

## Plots and validation

The bundle includes path plots for overlap area, maximum pressure, controlled
reactions, deformation, pressure redistribution, and topology-event locations,
plus a final projected-overlap view. SVG files are parsed before registration.

Finalization fails when a required table, plot, or available VTK checkpoint is
absent. The summary records solver and refinement status, checkpoint selection
completeness and errors, and row counts, while `manifest.json` records Git,
Python, package, platform, seed, and solver provenance. This bundle is
frictionless normal-contact evidence; it does not claim frictional behavior or
generalized derivatives at exact edge-on-edge and on-vertex states.
