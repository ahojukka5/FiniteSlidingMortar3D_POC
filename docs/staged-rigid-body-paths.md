# Staged rigid-body boundary paths

`RigidBodyBoundaryPath` describes one linear interpolation of axis-angle rotation,
pivot translation, and dead load. Production contact benchmarks often require a
sequence instead: compress the bodies, hold the normal approach, then rotate and
translate tangentially. `StagedRigidBodyBoundaryPath` composes those motions while
retaining one absolute continuation parameter.

## Motion segments

A `RigidBodyMotionSegment` maps one `RigidBodyBoundaryPath` onto an interval
`[s_a, s_b]`. Its local parameter is

```text
q = (s - s_a) / (s_b - s_a).
```

The staged path evaluates the segment at `q` but returns a `CoupledPathState`
whose `parameter` remains the absolute value `s`. Adaptive cutbacks and event
records therefore refer to the same physical path coordinate instead of a
segment-local coordinate.

Segments must:

- start at absolute parameter zero;
- have unique names and contiguous non-overlapping intervals;
- control the same nodes and use the same reference coordinates;
- retain the same unrelated fixed constraints;
- have identical prescribed displacements at every shared boundary;
- have identical dead loads at every shared boundary.

Continuity is checked with the path's explicit `continuity_tolerance`. A gap,
overlap, load jump, or prescribed-motion jump is rejected during construction.

## Boundary ownership

An interior phase boundary belongs to the following segment. At the end of a
compression interval, for example, the state is reported as rotation phase local
parameter zero. This convention makes phase transitions deterministic while the
physical boundary state remains identical on both sides.

Every evaluated state records:

- `phase_index`;
- `phase_parameter`;
- `phase_start`;
- `phase_end`;
- the underlying rigid-path values `rotation_angle` and `translation_x/y/z`.

The string phase name is available through `phase_name(s)`.

## Compression then rotation

The convenience factory constructs the two stages required by the
rotating-blocks benchmark:

```python
path = StagedRigidBodyBoundaryPath.compression_then_rotation(
    problem,
    controlled_nodes=upper_block_nodes,
    compression=np.array([0.0, 0.0, -0.05]),
    compression_end=0.25,
    pivot=upper_block_center,
    axis=np.array([0.0, 0.0, 1.0]),
    end_angle=np.deg2rad(120.0),
    tangential_translation=np.array([0.8, 0.1, 0.0]),
)
```

The compression stage moves from zero displacement to the supplied compression
vector with zero rotation. The second stage starts from that exact compressed
state and interpolates rotation and optional tangential translation while
holding the compression component.

With `proportional_load=True`, the dead load grows from zero to its full value
during compression and remains constant during rotation. Otherwise the existing
dead load is constant throughout both phases.

## Transactional continuation

Each segment delegates boundary construction to `RigidBodyBoundaryPath`, which
replaces only constraints and the dead-load vector. The exact symbolic CSR
sparsity object and the current contact-interface penalty state are retained.
Reevaluating an absolute parameter after an adaptive cutback therefore rebuilds
the same boundary state from the last accepted problem without retaining failed
trial displacements or temporary boundary values.

## Verification boundary

The focused tests verify:

- absolute and local phase parameters;
- phase-boundary displacement and load continuity;
- exact fixed-sparsity reuse;
- proportional-load completion during compression;
- rejection of interval gaps, discontinuous states, discontinuous loads,
  reserved phase-value names, and out-of-range parameters.

This layer provides kinematics only. The rotating-blocks benchmark still must
establish nonlinear convergence, contact retention, deterministic topology
events, pressure redistribution, and load-step refinement independently.
