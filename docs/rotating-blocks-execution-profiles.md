# Rotating-blocks execution profiles

The rotating-blocks benchmark has two named execution profiles. Both use the same
physical geometry, frictionless mortar formulation, compression-then-rotation
path, event-localized continuation policy, and result schemas. A profile changes
mesh resolution, requested continuation resolution, diagnostic sampling, and the
amount of optional evidence generated.

## Quick profile

The quick profile uses the `2 x 2 x 1` lower support mesh and the shared `3 x 2 x
1` rigid upper mesh. It requests 16 path steps and permits adaptive cutbacks down
to `1/1024`. This still gives a three-dimensional deformable support, nonmatching
QUAD4 mortar grids, contact onset, the complete prescribed rotation and
translation, and production topology-event handling.

The standardized quick runner executes one production nonlinear solve. It writes
accepted-state, attempt, solver, event, force/moment-balance, and contact-retention
evidence, together with deterministic plots and a complete final-state VTK
checkpoint. It does not run the optional repeated kinematic scan or the
coarse/medium/fine continuation study. Those are independent scientific evidence
campaigns rather than prerequisites for every integration check.

Quick mode is therefore a bounded physical integration test, not a synthetic
oracle and not a publication discretization. It must reach the same final motion,
meet the same scale-aware equilibrium and penetration limits, retain contact, and
exercise the same mechanics and nonlinear solver entry points as full mode.

## Full profile

The full profile uses the `8 x 8 x 4` lower support mesh and the same rigid upper
mesh. It requests 64 path steps, permits cutbacks down to `1/4096`, samples 129
kinematic states, enables all six physical checkpoint exports, and enables the
complete 32/64/128-step continuation-refinement study and deterministic repetition
checks.

Full mode is the publication-oriented evidence profile. Machine-dependent timings
remain provenance only and are not numerical acceptance criteria.

## Deterministic selection

`rotating_blocks_execution_profile("quick")` and
`rotating_blocks_execution_profile("full")` return immutable validated records.
The standardized benchmark runner selects `rotating_blocks_quick.py` in quick
mode and `rotating_blocks_bundle.py --profile full` in full mode. Unknown names
and inconsistent model/profile combinations fail before a solver is started.

The profile records do not claim nonlinear completion by themselves. Production
completion and KKT evidence belong to the rotating-blocks solver integration and
the profile-specific acceptance gates.
