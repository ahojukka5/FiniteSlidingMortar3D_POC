# Rotating blocks with changing overlap topology

This example demonstrates finite sliding while the mortar overlap topology
changes as a smaller upper block is compressed against, translated over, and
rotated on a larger lower block. The upper surface remains inside the larger
master surface, so its total overlap area stays approximately constant while
the intersecting master/slave facet pairs change.

The model intentionally stays small:

- coarse finite-strain `TET4` bulk meshes;
- nonmatching `QUAD4` contact surfaces;
- upper surface as the biased mortar/slave side;
- frictionless standard mortar with projected augmented-Lagrange enforcement;
- prescribed compression followed by translation and a 90-degree rotation;
- adaptive continuation with contact-event restarts.

## Run

From the repository root:

```bash
uv sync --extra dev
uv run python -m examples.rotating_blocks
```

Choose another output directory with:

```bash
uv run python -m examples.rotating_blocks --output /tmp/rotating-blocks
```

## Outputs

The default `results/` directory contains exactly six files:

- `summary.json`: compact accepted-state and topology-event histories;
- `compression.vtu`: the accepted state nearest the completed compression
  phase;
- `mid-rotation.vtu`: the accepted state nearest 45 degrees of rotation;
- `final.vtu`: the completed 90-degree rotating-block state;
- `deformed.svg`: reference and final x-y mesh projections;
- `reaction-path.svg`: controlled-block reaction norm over the path.

Each VTK state contains displacement, reaction, effective load, contact force,
contact pressure, normal gap, active/support flags, and body identifiers.

The command fails unless the path reaches its final state, contact is
established and supported, the overlap topology changes, topology events are
observed, equilibrium and penetration tolerances hold, global force balance is
satisfied, and every element Jacobian remains positive.

## Limitations

This is a verification-sized proof of concept rather than a production sliding
contact analysis. It does not establish frictional behavior, slave-side
independence, dual multipliers, higher-order geometry, mesh convergence, or
large-model performance. Angular-momentum error is not an acceptance gate
because this standard biased formulation does not conserve it exactly in
general.
