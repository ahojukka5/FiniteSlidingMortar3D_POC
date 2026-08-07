# Production solver composition

The v0.1 examples use one solver ownership model.  Numerical problem data is
assembled by `contact3d.coupling`, while solution algorithms live under
`contact3d.solvers`.  Topology-event variants are an optional extension under
`contact3d.solvers.events` rather than a parallel flat solver stack.

The production path is:

```text
contact3d.coupling
    -> contact3d.solvers.linear
    -> contact3d.solvers.newton
    -> contact3d.solvers.augmented
    -> contact3d.solvers.scaling
    -> contact3d.solvers.continuation
    -> contact3d.solvers.events        # only when topology restarts are needed
```

`contact3d.solvers` is the public solver aggregate for linear solves, Newton
options and results, augmented-Lagrange solves, scale-aware augmentation, and
adaptive continuation.  The rotating-blocks example additionally imports the
event-aware adaptive driver from `contact3d.solvers.events`.

Solver configuration that belongs to another subsystem stays with its owner.
In particular, `ScaleAwareConvergenceOptions` comes from `contact3d.scaling`
and `AugmentedLagrangeState` comes from `contact3d.mortar.enforcement`.
Examples may still use the package-root API for model-construction objects;
solver algorithms and solver options must not be reached through package-root
aliases or flat compatibility modules.

The flat modules such as `adaptive_solver`, `event_solver`, `linear_solver`,
`scaled_solver`, and `equilibrium` are migration scaffolding only.  Repository
code must not add new dependencies on them, and #136 removes them after the
refactor stack has landed.

## Example mapping

- `examples/contact_patch` uses `contact3d.solvers.solve_adaptive_contact_path`.
- `examples/sandwiched_beam` composes bulk Newton, scale-aware augmentation,
  and adaptive continuation from `contact3d.solvers`.
- `examples/rotating_blocks` uses
  `contact3d.solvers.events.solve_event_aware_adaptive_contact_path`.

This keeps all three examples on the same composition while preserving the
existing numerical tolerances, diagnostics, and output contracts.
