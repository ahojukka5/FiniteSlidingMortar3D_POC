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

Completed first slice:

- analytical Appendix A nominal-normal Jacobian;
- fixed-`D/M` weighted-gap derivative;
- pressure and traction derivatives with frozen unilateral activity;
- slave and master force-distribution derivatives;
- column-wise verification against a frozen-weight centered-difference oracle;
- exact common-translation nullspace regression.

Remaining derivatives from Section 4 and Appendices A–B:

- projection plane origin and normal;
- projected facet vertices;
- clipping intersection vertices, including edge-on-edge degeneracy;
- pallet centers and areas;
- both inverse maps and shape functions;
- moving `D` and `M` operators and the resulting complete contact tangent.

Topology, facet pairs, and unilateral activity are frozen during one generalized derivative evaluation. The numerical tangent introduced in Phase 2 remains the verification oracle for every analytical contribution.

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
