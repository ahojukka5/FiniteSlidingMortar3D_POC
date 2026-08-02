# Rotating-blocks production solver integration

The rotating-blocks benchmark now has one entry point that sends the shared model
and staged compression/rotation path through the production event-aware,
scale-aware, adaptive augmented-Lagrange solver.

```bash
uv run python benchmarks/rotating_blocks_solver.py --profile quick
uv run python benchmarks/rotating_blocks_solver.py --profile full
```

The command fails when the requested final motion is not reached or when required
solver evidence is incomplete. `--summary PATH` writes the same strict JSON summary
that is printed to standard output.

## Production configuration

Both profiles enable scale-aware convergence, interface-local penalty adaptation,
event-localized Newton restarts, and multiplier transport across support changes.
The execution profile controls the requested continuation resolution and adaptive
step bounds:

- quick starts at `1/16`, permits cutbacks to `1/1024`, and allows 128 attempts;
- full starts at `1/64`, permits cutbacks to `1/4096`, and allows 1024 attempts.

The physical model, final rigid motion, contact formulation, and convergence limits
are unchanged between profiles. The full profile increases mesh and path resolution
rather than replacing the benchmark with a different problem.

## Retained evidence

`RotatingBlocksSolverRun` keeps the complete solver result and deterministic rows
for later refinement and artifact writers:

- accepted states with path parameter, phase coordinate, rotation, reaction norm,
  and inner-solver status;
- every accepted, cut-back, and penalty-retried attempt with normalized residuals;
- every localized atomic topology event in absolute continuation coordinates;
- progress-aware restart diagnostics and repeated pair-entry/exit counts.

The summary checks final-path completion, converged accepted states, normalized
equilibrium and penetration limits, healthy restart history, and repeated production
pair entries and exits. It reports every criterion separately before combining them
into the top-level `passed` value.

This integration intentionally does not write the complete JSON/CSV/VTK/SVG result
bundle. Result packaging, checkpoint export, force and moment balance, pressure
redistribution, and the standardized benchmark gate remain separate follow-up
issues so they can be reviewed and reverted independently.
