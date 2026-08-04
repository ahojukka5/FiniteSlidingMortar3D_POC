# Nonmatching sandwiched-beam bending

This example demonstrates compressive load transfer through a nonmatching
three-dimensional mortar interface while two finite-strain TET4 beams bend
together.

The selected v0.1 model is intentionally modest:

- the existing coarse deterministic mesh family;
- upper beam as the biased mortar/slave side;
- frictionless standard mortar with projected augmented-Lagrange enforcement;
- ambient normal preload followed by a self-equilibrated end moment;
- one conforming monolithic TET4 beam as a response reference.

The two contact surfaces are coincident in the reference configuration. The
solver therefore starts with a small uniform multiplier predictor equal to the
first physical preload increment. This keeps the zero-gap interface on its
compressive branch; it is not a fallback contact operator or an artificial
constraint.

## Run

From the repository root:

```bash
uv sync --extra dev
uv run python -m examples.sandwiched_beam
```

Choose another output directory with:

```bash
uv run python -m examples.sandwiched_beam --output /tmp/sandwiched-beam
```

## Outputs

The default `results/` directory contains exactly four files:

- `summary.json`: compact contact and conforming-reference histories plus the
  final residual, penetration, pressure, rotation, and balance metrics;
- `final.vtu`: the final nonmatching contact solution for ParaView;
- `deformed.svg`: the reference and final x-z mesh projection;
- `moment-rotation.svg`: nonmatching and conforming end-rotation histories.

The run is accepted only when it reaches the final path parameter, maintains
supported compressive contact during bending, satisfies the documented
residual, penetration, and force-balance tolerances, keeps all element
Jacobians positive, and produces a bending response with the same sign as the
conforming reference.

The current-configuration angular-momentum residual remains in `summary.json`
as a diagnostic. The standard biased mortar virtual work used by this proof of
concept conserves linear momentum exactly but does not conserve angular
momentum exactly in general. Exact angular-momentum conservation would require
additional gap-dependent variation terms outside the v0.1 formulation.

## Limitations

This is a proof-of-concept bending problem, not a locking study. The coarse
linear TET4 discretization is intentionally retained because adding HEX8 or an
anti-locking element is outside v0.1. The reported difference from the
conforming reference is diagnostic rather than a mesh-convergence claim. The
example does not establish slave-side independence, frictional behavior, dual
multipliers, exact angular-momentum conservation, or production-scale
robustness.