# Rotating-blocks execution profiles

The rotating-blocks benchmark has two named execution profiles. Both use the same
physical geometry, frictionless mortar formulation, compression-then-rotation
path, event-localized continuation policy, and artifact schemas. A profile changes
only mesh resolution, requested continuation resolution, diagnostic sampling, and
the amount of optional evidence generated.

## Quick profile

The quick profile uses the `4 x 4 x 2` lower support mesh and the shared `3 x 2 x
1` rigid upper mesh. It requests 16 path steps, permits adaptive cutbacks down to
`1/1024`, samples the kinematic topology oracle at 65 states, and runs the
repetition check twice.

Quick mode keeps repeated topology transitions and exercises the same model and
solver entry points as full mode. It disables the expensive complete refinement
campaign and checkpoint export. Its purpose is bounded integration validation,
not a synthetic substitute for the physical benchmark.

## Full profile

The full profile uses the `8 x 8 x 4` lower support mesh and the same rigid upper
mesh. It requests 64 path steps, permits cutbacks down to `1/4096`, samples 129
kinematic states, enables checkpoint export, and enables the complete 32/64/128
step refinement study.

Full mode is the publication-oriented evidence profile. Machine-dependent timings
remain provenance only and are not numerical acceptance criteria.

## Deterministic selection

`rotating_blocks_execution_profile("quick")` and
`rotating_blocks_execution_profile("full")` return immutable validated records.
The standardized benchmark runner can serialize `as_dict()` directly into its
manifest. Unknown names and inconsistent model/profile combinations fail before a
solver is started.

The profile records do not claim nonlinear completion by themselves. Production
completion and KKT evidence belong to the rotating-blocks solver-integration and
acceptance-gate work.
