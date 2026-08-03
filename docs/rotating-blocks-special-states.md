# Rotating-blocks special-state regressions

The production rotating-blocks path is offset from exact grid coincidences so that
its nonlinear acceptance run tests long finite sliding rather than beginning on a
nondifferentiable clipping state. Exact edge-on-edge and on-vertex configurations
remain important, but they have a different contract: the current formulation
supports deterministic one-sided smooth branches and a typed event diagnostic at
the exact coincidence. It does not claim a unique generalized derivative there.

Run the isolated regressions with:

```bash
uv run python benchmarks/rotating_blocks_special_states.py \
  --output results/rotating-blocks-special-states
```

## Construction

The local `QUAD4` dimensions are derived from the quick rotating-blocks contact
meshes:

- one lower master facet spans `1.0 x 1.0`;
- one upper slave facet spans `(1.3 / 3) x (0.7 / 2)`;
- both facets lie on the lower reference contact plane.

Two exact projected states are constructed:

1. **edge-on-edge:** the slave left edge coincides with the master right edge;
2. **on-vertex:** the slave lower-left vertex coincides with the master
   upper-right vertex.

The crossing direction moves the slave from a small positive-overlap state to a
separated state. The perturbation size is deliberately much larger than the
geometric event tolerance while remaining small relative to either facet.

## Verified behavior

For each state the benchmark checks:

- the exact configuration raises `ClippingTopologyError`,
  `PalletTopologyError`, or `InverseMapTopologyError` rather than producing NaNs
  or an untyped geometry failure;
- the topology state machine records a recoverable clipping event;
- explicit `left` selection returns a valid topology state immediately before
  the localized event;
- explicit `right` selection returns a valid, distinct topology state immediately
  after the localized event;
- the finite negative and positive perturbations have distinct smooth-branch
  signatures and represent the overlap and separation sides of the exact state;
- the analytical directional derivatives of the local standard-mortar `D` and
  `M` operators agree with centered differences taken entirely inside each
  finite smooth branch.

The localized samples are chosen by the event tolerance and bisection policy.
They are not required to equal the farther perturbation samples used for tangent
verification, because more than one discrete clipping token may be crossed inside
a very small geometric neighborhood.

`branches.csv` records overlap area, operator norm, directional tangent norm,
tangent error, support count, intersection vertex count, and pallet count for
both finite perturbation sides. `summary.json` records the typed exact-state
diagnostic, localized event kinds, selected fractions, and pass/fail criteria.

## Claim boundary

The exact configuration itself is not evaluated with a smooth tangent. A vertex
on a clipping edge changes the discrete polygon construction, and the derivative
can depend on the selected side. The supported behavior is therefore:

- typed recognition of the nonsmooth state;
- deterministic localization;
- explicit left or right branch selection;
- verified smooth-branch residual and tangent after selection.

A unique semismooth or generalized derivative at every exact geometric
coincidence remains outside the frictionless v0.1 proof-of-concept claim. The
physical rotating-blocks benchmark continues to exercise the production solver
away from these exact states, while this isolated regression prevents accidental
NaNs, silent branch selection, or loss of the typed event contract.
