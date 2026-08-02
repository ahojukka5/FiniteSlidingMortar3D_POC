# Rotating-blocks benchmark model

The rotating-blocks campaign uses one deterministic physical model for the
kinematic topology oracle, nonlinear continuation, refinement study, and result
visualization. `benchmarks/rotating_blocks_model.py` owns that definition so the
later studies cannot drift to slightly different geometries or boundary paths.

## Physical geometry

The lower body is a deformable rectangular support:

```text
x in [-1.00, 1.00]
y in [-1.00, 1.00]
z in [ 0.00, 0.50]
```

The prescribed upper body is a smaller rectangular block:

```text
x in [-0.65, 0.65]
y in [-0.35, 0.35]
z in [ 0.52, 0.82]
```

The lower top surface is therefore at `z = 0.50` and the upper contact surface
starts at `z = 0.52`. The reference separation is `0.02`. The rotation pivot is
the center of the upper contact plane, `(0, 0, 0.52)`.

The upper footprint is rectangular rather than square. A 90-degree in-plane
rotation therefore changes the projected overlap polygons even though the motion
is centered. The final tangential translation of `0.10` in the global x direction
forces additional master-facet transitions while the footprint remains inside
the lower support.

## Bulk meshes

Both bodies use the same deterministic six-tetrahedron subdivision of every
structured hexahedral cell. The decomposition has positive reference orientation
for all cells. The model factory computes and stores the minimum signed TET4
reference determinant and rejects an inverted mesh.

The canonical profiles are:

| Profile | Lower cells | Upper cells | Lower TET4 | Upper TET4 |
|---|---:|---:|---:|---:|
| `quick` | `4 x 4 x 2` | `3 x 2 x 1` | 192 | 36 |
| `full` | `8 x 8 x 4` | `3 x 2 x 1` | 1536 | 36 |

The upper discretization is unchanged because every upper node follows an exact
rigid-body prescription. Refinement is concentrated in the deformable support and
its master contact surface. Both profiles retain identical dimensions, separation,
pivot, compression, final angle, translation, material parameters, contact law,
and path phase boundary.

A `RotatingBlocksProfile` object can also provide an intermediate structured mesh
for the later load-step and mesh-refinement studies while retaining the same
physical `RotatingBlocksGeometry`.

## Nonmatching mortar surfaces

The upper bottom surface is the non-mortar slave side. Its six QUAD4 facets use
12 nodes and a negative normal sign so the reference nominal normal points toward
the lower support.

The lower top surface is the master side. It contains 16 QUAD4 facets and 25 nodes
in the quick profile, or 64 facets and 81 nodes in the full profile. Consequently,
the two contact grids are nonmatching in both canonical profiles.

`MortarContactInterface.validate_for` verifies that local contact coordinates are
exactly equal to the mapped global bulk coordinates. The model factory additionally
checks that all contact facets are QUAD4 and that the search distance `0.12`
exceeds the initial separation.

The initial normal penalty is `3200`. The production solver may increase it through
the existing scale-aware interface-local penalty policy; the model factory only
defines the common starting problem.

## Boundary conditions

Every node on the lower bottom face is fixed in all three components. Every node
of the upper block is constrained in all three components and belongs to the
rigid-body path. Thus, the upper TET4 mesh is retained for geometry, reactions, and
visualization, but it has no deformable free mode.

The fixed lower and controlled upper node sets are disjoint. The model factory
requires the problem constraint vector to contain exactly those two sets and to
start from zero prescribed displacement.

## Staged motion

The path uses `StagedRigidBodyBoundaryPath.compression_then_rotation`:

1. `s in [0, 0.25]`: translate the upper block by `(0, 0, -0.04)`;
2. `s in [0.25, 1]`: hold that compression, rotate through `pi/2` about the global
   z axis, and add `(0.10, 0, 0)` tangential translation.

At the end of compression the initial `0.02` separation has been consumed and the
upper footprint has moved `0.02` into the undeformed lower reference plane. The
nonlinear solution determines the actual lower-body deformation and contact
pressure required to satisfy equilibrium.

The path records the absolute parameter, phase index, phase-local parameter,
rotation angle, and translation components. Interior phase boundaries belong to
the following phase, making retries and event localization deterministic.

## Validation boundary

The model-factory tests establish only geometry and kinematics:

- deterministic nodes, elements, contact mappings, and constrained sets;
- positive signed reference determinants;
- exact QUAD4 surface-to-bulk mappings;
- complete rigid control of every upper node;
- identical physical endpoints in quick and full profiles;
- rejection of unknown canonical profile names and inadequate search distance.

They do not claim that the nonlinear rotating contact path converges. That evidence
belongs to the production event-aware adaptive solver benchmark, its kinematic
topology oracle, and the refinement and acceptance-gate studies tracked under
issue #24.
