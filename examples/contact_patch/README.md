# Nonmatching frictionless contact patch

This is the smallest complete contact problem in the repository. Two tiny
finite-strain TET4 bodies start separated and are brought into contact through
a monotone mixed path. The upper contact side is one warped `QUAD4`; the lower
side is two warped `TRI3` facets with different in-plane nodes.

The formulation is deliberately bounded:

- frictionless biased single-pass standard mortar;
- projected augmented-Lagrange normal enforcement;
- compressible neo-Hookean TET4 bulk mechanics;
- one verification-sized model on one machine.

## Run

From the repository root:

```bash
uv sync --extra dev
uv run python -m examples.contact_patch
```

Choose another output directory with:

```bash
uv run python -m examples.contact_patch --output /tmp/contact-patch
```

## Outputs

The default `results/` directory contains exactly three files:

- `summary.json`: compact geometry, solver, reaction, residual, gap, and
  force-balance metrics;
- `final.vtu`: the deformed TET4 bodies with reaction, load, contact-force,
  pressure, gap, active-row, and support fields;
- `pressure.svg`: final mortar pressure at the four slave rows.

A run passes only when it reaches path parameter 1.0, establishes active
contact, keeps the normalized equilibrium residual at or below `1e-8`, keeps
normalized penetration at or below `2e-7`, preserves the mortar partition to
`1e-10`, and retains positive element Jacobians.

## What this demonstrates

The example demonstrates that the current solver can assemble and solve one
small three-dimensional nonmatching frictionless mortar contact problem from a
clean checkout and produce results that can be inspected without benchmark
campaign tooling.

It does not claim mesh convergence, unbiased slave/master behavior, HEX8
accuracy, friction, dual multipliers, Hertz agreement, or production-scale
robustness.
