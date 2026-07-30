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

The penalty residual remains the zero-multiplier baseline for verification. The same geometry and force assembly now also serve the projected augmented-Lagrange law in Phase 4.

## Phase 3 — consistent geometric linearization — smooth branch complete

Completed:

- analytical Appendix A nominal-normal Jacobian;
- analytical fixed-`D/M` force-law tangent;
- complete smooth tangent decomposition including row-area and force-distribution derivatives;
- analytical projection-plane origin, tangent, and normal Jacobians;
- analytical projected slave and master vertex Jacobians;
- topology-frozen clipping traces and analytical intersection-vertex Jacobians;
- analytical centroid-fan centers, pallet vertices, signed areas, and total area;
- analytical `TRI3` and `QUAD4` inverse-parent Jacobians;
- analytical pallet quadrature-point, shape-value, and integration-weight Jacobians;
- analytical local and global `D` and `M` operator Jacobians;
- fully analytical smooth-branch contact tangent;
- retained numerical `D`/`M` and residual-tangent oracles;
- derivative-level momentum, overlap-area, and rigid-translation tests;
- comparison against independent centered-difference oracles.

Remaining generalized derivatives:

- explicit edge-on-edge and on-vertex clipping derivatives;
- transition policies for zero-area and newly appearing/disappearing pallets;
- semismooth treatment of those topology events in the nonlinear solver.

Topology, facet pairs, and unilateral activity are frozen during one smooth derivative evaluation. Clipping states inside a configurable event band, degenerate pallets, and singular inverse maps are rejected as outer topology events instead of receiving an arbitrary derivative.

## Phase 4 — constraint enforcement — verification path complete

Completed:

- immutable nonnegative nodal multiplier state;
- projected update `lambda <- max(0, lambda + kappa g)`;
- exact primal, dual, complementarity, support, and projection-form KKT diagnostics;
- explicit separation between inner equilibrium evaluation and accepted outer augmentation;
- analytical augmented-Lagrange tangent with multipliers and activity frozen;
- independent centered-difference tangent oracle;
- penalty equivalence at zero multipliers and pressure-release regressions;
- mapped mortar surfaces in global bulk-mesh DOFs;
- combined bulk/contact residual and tangent assembly;
- fixed-multiplier coupled Newton equilibrium;
- outer augmented-Lagrange iteration with per-interface KKT stopping criteria;
- explicit restart or rejection policies for facet-pair, support, and activity changes;
- complete inner-Newton and outer-augmentation histories.

Remaining:

- adaptive penalty updates based on penetration and conditioning;
- optional primal-dual active-set/dual-mortar path based on Popp et al. (2010);
- local condensation once a biorthogonal multiplier basis is introduced.

## Phase 5 — three-dimensional solid mechanics and benchmarks — in progress

Completed:

- compressible logarithmic neo-Hookean energy density;
- analytical first Piola stress and fourth-order material tangent;
- positively oriented total-Lagrangian `TET4` reference geometry;
- analytical `TET4` internal force and consistent tangent;
- dense multi-element `TET4` assembly for verification-sized systems;
- reusable symbolic CSR pattern and sparse numerical tangent assembly;
- contact-generated cross-body blocks in the symbolic CSR pattern;
- strong essential boundary conditions and configuration-independent nodal loads;
- free-DOF Newton solve with Armijo residual line search;
- monotone load stepping with warm-started predictors;
- machine-readable nonlinear histories and explicit failure reasons;
- independent constitutive, element, dense-assembly, and sparse-assembly oracles;
- energy-gradient, objectivity, force-balance, moment-balance, and reaction regressions;
- 12-element cube-star affine patch test with an interior equilibrium node;
- manufactured large-deformation equilibrium benchmark with line-search cutbacks;
- two-block matching-mortar coupled benchmark with Newton/KKT/pressure plots;
- JSON/CSV result tables and SVG convergence, response, sparsity, and pressure plots.

Remaining:

- adaptive load cutback across failed coupled increments;
- scalable sparse direct and Krylov linear-solver backends;
- `HEX8` finite-strain neo-Hookean element and locking studies;
- warped-interface contact patch test using the full moving-overlap adapter;
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
