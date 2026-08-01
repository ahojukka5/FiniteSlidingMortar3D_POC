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

The penalty residual remains the zero-multiplier baseline for verification. The same geometry and
force assembly also serve the projected augmented-Lagrange law in Phase 4.

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
- comparison against independent centered-difference oracles;
- warped nonmatching `QUAD4`/`TRI3` production-adapter regression.

Completed topology-event foundation:

- typed pair, clipping, pallet, support, pressure, and inverse-map event records;
- production overlap geometry tokens in the branch signature;
- deterministic segment bisection and explicit left/right branch selection;
- exact post-event Newton restart with retained smooth tangents;
- event-aware dimensional and scale-aware augmented-Lagrange solvers;
- adaptive load, prescribed-displacement, and mixed-path event propagation;
- absolute continuation targets separated from local Newton event fractions;
- event retention across accepted, cut-back, and penalty-retried attempts;
- subdivision-invariance tables and event timeline plots.

Remaining generalized derivatives and validation:

- exact special-state derivatives at edge-on-edge and on-vertex configurations where a unique
  generalized branch cannot be represented by one-sided smooth restart alone;
- event-local multiplier transport when supported row sets change dimension;
- rotating-overlap physical validation through repeated topology events.

## Phase 4 — constraint enforcement — in progress

Completed:

- immutable nonnegative nodal multiplier state;
- projected update `lambda <- max(0, lambda + kappa g)`;
- exact primal, dual, complementarity, support, and projection-form KKT diagnostics;
- explicit separation between inner equilibrium evaluation and accepted outer augmentation;
- analytical augmented-Lagrange tangent with multipliers and activity frozen;
- independent centered-difference tangent oracle;
- penalty equivalence at zero multipliers and pressure-release regressions;
- coupled bulk/contact equilibrium and outer augmentation driver;
- adaptive load cutback and growth;
- transactional normal-penalty escalation after under-resolved augmentations;
- explicit attempt histories proving rollback across failed candidates;
- immutable prescribed-displacement, dead-load, and mixed continuation paths;
- axis-angle rigid-body boundary paths with pivot translation and fixed-constraint retention;
- complete boundary-state rollback across cutbacks and penalty retries;
- fixed symbolic CSR reuse when only boundary values and loads change;
- constrained reaction vectors for every accepted path state;
- reference bulk/interface length, pressure, force, energy, and penalty scales;
- dimensional and normalized Newton/KKT convergence histories;
- explicit penalty-control protocol with production and verification adapters;
- interface-local penetration and penalty-conditioning indicators;
- bounded interface-local penalty updates with recorded reasons;
- unit-system invariance regression for normalized stopping and penalty decisions.

Remaining:

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
- strong essential boundary conditions and configuration-independent nodal loads;
- free-DOF Newton solve with Armijo residual line search;
- monotone load stepping with warm-started predictors;
- coupled bulk/contact sparse assembly and contact-event restart policies;
- outer augmented-Lagrange iteration with per-interface KKT stopping criteria;
- adaptive continuation controller with load cutback and penalty retries;
- proportional, mixed, and rigid-body boundary/load-path infrastructure;
- separated matching-interface contact-onset benchmark harness;
- scale-aware interface-local penalty control;
- scale-aware two-interface penalty policy and unit-conversion regression;
- production warped nonmatching moving-overlap contact-onset benchmark harness;
- deterministic separated, first-contact, and established-contact tangent checks;
- synthetic topology-event localization benchmark;
- adaptive mixed-path event-propagation benchmark;
- machine-readable nonlinear, coupled, continuation, and event histories;
- independent constitutive, element, dense-assembly, sparse-assembly, and coupled oracles;
- energy-gradient, objectivity, force-balance, moment-balance, and reaction regressions;
- 12-element cube-star affine patch test with an interior equilibrium node;
- manufactured large-deformation equilibrium benchmark with line-search cutbacks;
- matching-mortar two-block coupled benchmark;
- sparse direct and Krylov linear-solver backends with scaling evidence;
- JSON/CSV result tables and SVG convergence, response, pressure, and continuation plots.

Remaining:

- production rotating blocks with changing overlap area;
- `HEX8` finite-strain neo-Hookean element and locking studies;
- warped-interface contact patch convergence study;
- cylindrical ironing;
- rotating concentric spheres;
- Hertz contact and mesh-convergence studies.

Every benchmark must write machine-readable convergence histories and visual deformation/contact-
pressure plots.

## Phase 6 — extensions

- objective friction from Puso–Laursen (2004) and Gitterle et al. (2010);
- dual linear and quadratic multiplier spaces;
- crosspoint-safe point/line/surface contact;
- unbiased dual-pass contact and self-contact;
- scalable sparse solvers and multigrid.
