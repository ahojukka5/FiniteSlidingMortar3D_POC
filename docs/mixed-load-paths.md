# Prescribed-displacement and mixed load paths

The coupled Newton and augmented-Lagrange kernels solve one immutable
`CoupledEquilibriumProblem`. Continuation must additionally define which boundary data
belong to a requested pseudo-time or load parameter. This layer is deliberately separate
from the mechanics kernels: changing prescribed values or a dead-load vector does not
change the bulk/contact topology or the symbolic CSR pattern.

## Path state

A `CoupledLoadPath` evaluates a template problem at a nonnegative parameter `s` and
returns a `CoupledPathState` containing:

- the problem carrying the candidate Dirichlet constraints and dead load;
- the scalar load factor passed to the inner coupled solver;
- the complete prescribed DOF/value snapshot;
- the effective external force vector;
- optional named benchmark values such as tool position or rotation.

The original API remains the default through `LoadFactorPath`. It leaves the problem
unchanged and passes `s` as the existing dead-load multiplier.

`LinearBoundaryPath` interpolates endpoint data,

```text
u_bar(s) = u_bar_0 + s (u_bar_1 - u_bar_0)
f_ext(s) = f_0     + s (f_1     - f_0),
```

and passes an inner load factor of one because the candidate problem already contains the
physical force vector. Convenience constructors provide:

- proportional prescribed displacement with a constant dead load;
- proportional dead load with constant prescribed values;
- proportional mixed prescribed-displacement/dead-load control.

Explicit endpoint objects remain available when some prescribed values must stay fixed
while others evolve.

## Fixed sparsity

`with_coupled_boundary_data` copies the immutable coupled problem and replaces only its
constraints and dead load. The exact existing `Tet4Sparsity` object is retained. This is
valid because the symbolic pattern depends on the bulk mesh and mapped contact DOFs, not
on prescribed values or force magnitudes.

Penalty changes may still create an equivalent problem through the existing penalty
replacement path. Issue #16 will formalize that interface and make interface-local penalty
updates reuse the same symbolic data as well.

## Transactional continuation

For a candidate parameter `s_trial`, the adaptive driver evaluates one complete path state
and calls the inner augmented solver with that state's problem and solver load factor.
A successful solve commits together:

```text
(path state, problem, displacement, multipliers, penalties, reaction).
```

A failed solve commits none of them. A cutback reevaluates the path from the last accepted
problem, so failed prescribed values, force vectors, multiplier iterates, and temporary
penalty changes cannot leak into the retry.

Penalty escalation at a fixed candidate parameter retains the same boundary snapshot.
The escalated penalty becomes part of the accepted path only after the retry reaches
coupled equilibrium and the requested KKT tolerances.

## Reactions and histories

Every accepted state is stored as an `AdaptiveAcceptedStep`. Its reaction vector is the
assembled coupled residual restricted to constrained DOFs after equilibrium. Attempt
records additionally contain:

- requested prescribed DOFs and values;
- named path values;
- effective load norm;
- constrained reaction norm;
- the existing Newton, augmentation, penetration, event, and penalty diagnostics.

This makes prescribed-displacement response curves reproducible without reconstructing
boundary data from benchmark-specific code.

## Verification boundary

`mixed_path_regression.py` is a deterministic controller regression. It demonstrates a
failed mixed-boundary candidate, rollback, cutback, fixed-sparsity reuse, accepted reactions,
and final recovery of the requested endpoint. It does not claim a new mechanics result.

`mixed_contact_onset.py` is the physical verification benchmark. It places two finite-strain
blocks at a known initial separation, drives the upper block by proportional prescribed
translation while simultaneously increasing a dead load, and records first contact,
reactions, KKT quantities, and pressure. Its matching frozen mortar operator isolates the
new path semantics. Issue #17 replaces that oracle with the complete warped nonmatching
moving-overlap production interface.
