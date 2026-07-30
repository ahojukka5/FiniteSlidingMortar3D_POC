# Implementation plan

## Phase 1 — geometric overlap kernel — complete

- center tangent plane for `TRI3` and `QUAD4` non-mortar facets;
- projection of both facets into a shared two-dimensional frame;
- deterministic convex polygon clipping;
- centroid-fan pallet triangulation;
- inverse parent mappings for triangles and bilinear quadrilaterals;
- common quadrature and local `D`/`M` assembly;
- row-wise linear-momentum consistency tests.

## Phase 2 — frictionless contact residual — complete

- validated surface topology and current coordinates;
- AABB facet-pair broad phase;
- current-configuration nodal normals following Appendix A;
- global `D`/`M` assembly from independently integrated facet pairs;
- weighted gap vectors and row-area-normalized scalar normal gaps;
- frictionless penalty activation with an explicitly freezable active set;
- force and moment balance diagnostics;
- frozen-pair numerical tangent oracle.

The penalty residual is intentionally a verification kernel, not yet the paper's augmented-Lagrange solver. It retains the exact weighted gap and exposes the area normalization explicitly so later constraint laws can replace it without changing the overlap or force assembly.

## Phase 3 — consistent geometric linearization — in progress

Completed:

- analytical Appendix A nominal-normal Jacobian;
- analytical fixed-`D/M` force-law tangent;
- topology-frozen numerical Jacobians of the moving `D` and `M` operators;
- complete smooth tangent decomposition including row-area and force-distribution derivatives;
- analytical projection-plane origin, tangent, and normal Jacobians;
- analytical projected slave and master vertex Jacobians;
- topology-frozen clipping traces and analytical intersection-vertex Jacobians;
- analytical centroid-fan centers, pallet vertices, signed areas, and total area;
- analytical `TRI3` and `QUAD4` inverse-parent Jacobians;
- analytical pallet quadrature-point, shape-value, and integration-weight Jacobians;
- derivative-level momentum consistency and rigid-translation tests;
- comparison against independent centered-difference oracles.

Remaining analytical derivatives from Section 4 and Appendices A–B:

- explicit edge-on-edge and on-vertex generalized derivatives;
- analytical local and global `D` and `M` operator Jacobians replacing the numerical geometry oracle.

Topology, facet pairs, and unilateral activity are frozen during one generalized derivative evaluation. The moving-operator Jacobian defines the tensor contract that every analytical geometry layer must satisfy. Clipping states inside a configurable event band, degenerate pallets, and singular inverse maps are rejected as outer topology events instead of receiving an arbitrary smooth derivative.

## Phase 4 — constraint enforcement

- augmented-Lagrange update matching the 2004 examples;
- exact KKT diagnostics;
- optional primal-dual active-set/dual-mortar path based on Popp et al. (2010);
- local condensation once a biorthogonal multiplier basis is introduced.

## Phase 5 — three-dimensional solid mechanics and benchmarks

- `TET4` and `HEX8` finite-strain neo-Hookean bulk elements;
- adaptive Newton and line search;
- warped-interface patch test;
- rotating blocks with changing overlap area;
- cylindrical ironing;
- rotating concentric spheres;
- Hertz contact and mesh-convergence studies.

Every benchmark must write machine-readable convergence histories and visual deformation/contact-pressure plots.

## Phase 6 — extensions

- objective friction from Puso–Laursen (2004) and Gitterle et al. (2010);
- dual linear and quadratic multiplier spaces;
- crosspoint-safe point/line/surface contact;
- unbiased dual-pass contact and self-contact;
- scalable sparse solvers and multigrid.
