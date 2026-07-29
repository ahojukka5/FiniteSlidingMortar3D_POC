# Implementation plan

## Phase 1 — geometric overlap kernel

- center tangent plane for `TRI3` and `QUAD4` non-mortar facets;
- projection of both facets into a shared two-dimensional frame;
- deterministic convex polygon clipping;
- centroid-fan pallet triangulation;
- inverse parent mappings for triangles and bilinear quadrilaterals;
- common quadrature and local `D`/`M` assembly;
- row-wise linear-momentum consistency tests.

This phase is independent of contact activation, pressure enforcement, bulk elements, and Newton iteration. It provides a small numerical oracle for all later residual and tangent work.

## Phase 2 — frictionless contact residual

- surface mesh and facet-pair broad phase;
- current-configuration nodal normals following Appendix A;
- weighted gap vectors and scalar normal gaps;
- penalty residual with transactional active-set state;
- force and moment balance diagnostics;
- numerical directional derivative oracle.

## Phase 3 — consistent geometric linearization

Implement the derivatives described in Section 4 and Appendices A–B:

- projection plane origin and normal;
- projected facet vertices;
- clipping intersection vertices, including edge-on-edge degeneracy;
- pallet centers and areas;
- both inverse maps and shape functions;
- `D`, `M`, weighted gap, nodal normal, pressure, and contact force.

Topology is frozen during one generalized derivative evaluation. Candidate-facet discovery is rebuilt outside the smooth tangent.

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
