# Rotating-blocks production solver integration

The rotating-blocks benchmark sends the shared model and staged
compression/rotation path through the production event-aware, scale-aware,
adaptive augmented-Lagrange solver.

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

- quick starts at `1/16`, permits cutbacks to `1/1024`, allows 128 attempts, and
  allows 32 augmented iterations per attempted state;
- full starts at `1/64`, permits cutbacks to `1/4096`, allows 1024 attempts, and
  allows 48 augmented iterations per attempted state.

The continuation recovery threshold uses the same profile-specific work bound.
This permits a warm-started state that needs several inexpensive augmentation
updates to regrow the path step after contact-onset cutbacks. It does not bypass
any KKT criterion: every accepted state must still satisfy all normalized limits.

The physical geometry, final rigid motion, contact formulation, material, and
convergence definitions are unchanged between profiles. The quick profile reduces
the deformable support mesh for bounded integration validation; the full profile
increases mesh and path resolution for publication evidence.

The scale-aware limits are:

| Measure | Normalized limit |
|---|---:|
| free equilibrium residual | `1e-8` |
| maximum penetration | `1e-7` |
| complementarity | `1e-7` |
| multiplier admissibility | `1e-7` |
| projection residual | `1e-5` |

The projection residual is pressure-valued: after normalization it contains the
interface penalty-to-pressure-scale ratio multiplying the normalized gap. Giving
it the same numerical limit as penetration would therefore demand a gap tens or
hundreds of times smaller than the stated `1e-7` penetration requirement. The
`1e-5` projection limit preserves the strict gap and complementarity limits while
preventing an already equilibrated first-contact state from exhausting the outer
augmentation loop solely because of this scaling difference.

## Retained evidence

`RotatingBlocksSolverRun` keeps the complete solver result and deterministic rows
for later artifact and refinement writers:

- accepted states with path parameter, phase coordinate, rotation, reaction norm,
  and inner-solver status;
- every accepted, cut-back, and penalty-retried attempt with normalized residuals;
- every localized atomic topology event in absolute continuation coordinates;
- progress-aware restart diagnostics and event-kind, pair, and support counts;
- deterministic linear, Newton, augmentation, retry, and event work counters.

The summary checks final-path completion, converged accepted states, normalized
equilibrium and penetration limits, healthy restart history, at least two distinct
localized production topology transitions, and the configured sparse-backend
policy. A transition is deduplicated by event kind, interface, entity, and absolute
continuation coordinate so repeated augmentation records of one event cannot pass
the gate.

Pair-entry and pair-exit counts remain reported. They are not mandatory because a
coarse quick model can retain the same broad-phase candidate pairs while still
crossing clipping-vertex, pallet, support, and pressure branches in the exact
contact integration and active-set machinery.

The bounded standardized writer adds force/moment balance, contact retention,
plots, manifest validation, and one complete final-state VTK checkpoint to a
single quick production solve. The full result-bundle writer additionally runs
repetition and continuation-refinement campaigns, pressure and mesh-quality
comparisons, all six physical checkpoints, and the aggregate publication gate.
