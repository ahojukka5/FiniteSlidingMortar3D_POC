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

## Rigid-body boundary motion

`RigidBodyBoundaryPath` prescribes all three displacement components of selected mesh nodes
from an interpolated axis-angle transform. For reference position `X`, pivot `c`, pivot
translation `t(s)`, normalized axis `a`, and angle `theta(s)`, the prescribed current
position is

```text
x_bar(X, s) = c + t(s) + R(a, theta(s)) (X - c).
```

The path therefore imposes

```text
u_bar(X, s) = x_bar(X, s) - X
```

without replacing unrelated fixed constraints. Both translation and angle are interpolated
linearly between their start and end values. The dead-load endpoint is interpolated in the
same way and the inner solver load factor remains one.

The `from_problem` factory expects every controlled node to be fully constrained in the
template problem. It removes those controlled DOFs from the fixed subset and obtains their
reference coordinates directly from the mesh. A caller can retain a constant existing dead
load or request proportional loading from zero.

```python
path = RigidBodyBoundaryPath.from_problem(
    problem,
    controlled_nodes=np.array([13, 14, 15, 16]),
    pivot=np.array([0.5, 0.5, 2.0]),
    axis=np.array([0.0, 0.0, 1.0]),
    end_angle=np.pi / 2.0,
    end_translation=np.array([0.0, 0.0, -0.08]),
)
```

Every path state records `rotation_angle`, `translation_x`, `translation_y`, and
`translation_z`. Additional `LinearPathValue` records can describe benchmark phases or tool
coordinates without reconstructing the transform afterward. `controlled_displacements(s)`
provides the exact nodal values independently of problem construction for verification and
visualization.

This path is the kinematic foundation for the rotating-blocks and concentric-spheres
benchmarks. It keeps rigid boundary motion out of benchmark-specific interpolation code and
makes load-step refinement compare the same geometric path.

## Fixed sparsity

`with_coupled_boundary_data` copies the immutable coupled problem and replaces only its
constraints and dead load. The exact existing `Tet4Sparsity` object is retained. This is
valid because the symbolic pattern depends on the bulk mesh and mapped contact DOFs, not
on prescribed values or force magnitudes.

Penalty changes may still create an equivalent problem through the existing penalty
replacement path. Interface-local penalty updates retain the same symbolic data.

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
new path semantics. The warped nonmatching production-onset benchmark exercises the complete
moving-overlap interface.
