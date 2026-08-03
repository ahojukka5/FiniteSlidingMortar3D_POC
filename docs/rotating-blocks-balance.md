# Rotating-blocks force and moment balance

The rotating-blocks result bundle audits force and moment balance at every
accepted continuation state. The table schema is
`contact3d-rotating-blocks-balance/v1`.

## Resultants

The applied resultant is the dead-load vector used by the accepted equilibrium
state. The reaction resultant is the residual retained on constrained degrees
of freedom. Their sum is the global force-balance error. Internal bulk forces
and contact forces are not added to this global resultant because both are
internal action-reaction systems.

The contact residual is split using the interface-local ordering: all slave
nodes first, followed by all master nodes. Slave and master resultants are
reported separately. Their sum is the contact force-cancellation error. This
separation makes contact-force sign and global-node mapping errors visible even
when the free equilibrium residual is small.

## Moments

Nodal moments use current coordinates. Global applied and reaction moments, and
slave and master contact moments, are evaluated about two points:

- the fixed global origin;
- the current rigid-motion pivot.

The current pivot is the reference rotation pivot plus the staged rigid-path
translation. During compression this moves with the prescribed compression.
During rotation it also includes the interpolated tangential translation.

## Scale-aware errors

Force errors are divided by the largest total variation of the relevant nodal
force systems. Total variation is the sum of nodal force magnitudes, so equal
and opposite forces still define a nonzero physical scale.

Moment errors are divided by the largest of the corresponding nodal moment
total variations and force scale times the current geometric length scale. The
length scale is the maximum nodal distance from the origin or moving pivot.
This keeps force and moment checks dimensionally separate and avoids a fixed
problem-unit normalization.

The quick and full bundles currently use the same limits:

- normalized force errors: `1e-7`;
- normalized moment errors: `1e-7`.

The bundle fails its balance criterion when any accepted state exceeds a limit.
The summary records every maximum and identifies the worst force state and
worst moment state by metric, accepted-step index, continuation parameter, and
observed value. Ties are resolved deterministically by step index and metric
name.

## Artifacts

The result directory contains:

- `tables/force-moment-balance.csv` with resultants, moments, scales, and errors;
- `plots/force-balance.svg` with global and contact force errors;
- `plots/moment-balance.svg` with origin and pivot moment errors;
- `plots/balance-worst-states.svg` locating the worst accepted states.

This audit covers frictionless normal contact. It does not establish tangential
traction balance or frictional couple behavior.