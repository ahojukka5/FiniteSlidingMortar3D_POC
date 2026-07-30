# FiniteSlidingMortar3D_POC

Research implementation of three-dimensional finite-sliding mortar contact methods.

The first target is the frictionless large-deformation segment-to-segment formulation of Puso and Laursen (2004). The implementation follows the paper's geometric construction: project each proximate facet pair to the non-mortar center plane, clip the polygons, triangulate the overlap, and integrate both sides at identical physical quadrature points.

## Current capability

- linear `TRI3` and bilinear `QUAD4` contact facets;
- center-plane projection for curved and warped current configurations;
- analytical center-plane and projected-vertex Jacobians;
- deterministic convex polygon clipping and triangular mortar pallets;
- topology-frozen clipping traces and analytical intersection-vertex Jacobians;
- analytical centroid-fan center, pallet-vertex, and signed-area Jacobians;
- analytical inverse-parent, quadrature-point, shape-value, and integration-weight Jacobians;
- analytical local and global standard-mortar `D`/`M` operator Jacobians;
- 1-, 3-, and 7-point triangle quadrature;
- local and global standard-mortar `D` and `M` matrices;
- row-wise partition-of-unity diagnostics for linear momentum conservation;
- current-configuration averaged nodal normals from Appendix A;
- analytical Appendix A nodal-normal Jacobian;
- AABB broad-phase discovery independent of projected vertex ownership;
- weighted mortar gaps and area-normalized frictionless penalty enforcement;
- projected augmented-Lagrange multiplier state and updates;
- primal, dual, complementarity, and projection-form KKT diagnostics;
- analytical penalty and augmented-Lagrange smooth-branch tangents;
- finite-strain compressible neo-Hookean energy, first Piola stress, and material tangent;
- positively oriented total-Lagrangian `TET4` elements with analytical residuals and tangents;
- dense and deterministic CSR `TET4` mesh assembly with independent derivative oracles;
- strong essential boundary conditions, dead nodal loads, and reaction extraction;
- Newton equilibrium solution with Armijo line search and load stepping;
- mapped bulk/contact residual and tangent assembly in one global CSR system;
- fixed-multiplier coupled Newton solves with explicit contact-event restarts;
- outer augmented-Lagrange solves with exact KKT stopping criteria;
- machine-readable affine-patch, bulk-equilibrium, and coupled-contact artifacts;
- force and moment balance diagnostics;
- retained numerical operator and residual-tangent oracles.

The package now solves verification-sized frictionless bulk/contact boundary-value problems. Adaptive load and penalty control, scalable sparse linear solvers, `HEX8` elements, friction, generalized clipping-event derivatives, dual multiplier spaces, and production finite-sliding benchmarks are tracked in [the implementation plan](docs/implementation-plan.md).

The residual equations and normalization boundary are documented in [the frictionless contact note](docs/frictionless-contact-residual.md). The tangent decomposition is documented in [the linearization note](docs/consistent-linearization.md), [the moving-overlap note](docs/moving-overlap-tangent.md), [the projection-plane note](docs/projection-plane-linearization.md), [the clipping note](docs/clipping-linearization.md), [the pallet note](docs/pallet-linearization.md), [the inverse-map note](docs/inverse-map-linearization.md), and [the operator note](docs/operator-linearization.md). The multiplier update and KKT residuals are documented in [the augmented-Lagrange note](docs/augmented-lagrange.md). The first bulk formulation is documented in [the neo-Hookean `TET4` note](docs/tet4-neo-hookean.md), its sparse nonlinear equilibrium layer in [the equilibrium note](docs/nonlinear-equilibrium.md), and the combined solver in [the coupled-equilibrium note](docs/coupled-equilibrium.md).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

Regenerate the verification benchmarks with:

```bash
uv run python benchmarks/tet4_patch.py --output results/tet4-patch
uv run python benchmarks/nonlinear_equilibrium.py --output results/nonlinear-equilibrium
uv run python benchmarks/coupled_mortar_patch.py --output results/coupled-mortar-patch
```

## Principal source

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for large deformation solid mechanics,” *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629. DOI: `10.1016/j.cma.2003.10.010`.

See [the literature review](docs/literature-review.md) for the formulation boundary and follow-up sources.
