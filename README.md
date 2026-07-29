# FiniteSlidingMortar3D_POC

Research implementation of three-dimensional finite-sliding mortar contact methods.

The first target is the frictionless large-deformation segment-to-segment formulation of Puso and Laursen (2004). The implementation follows the paper's geometric construction: project each proximate facet pair to the non-mortar center plane, clip the polygons, triangulate the overlap, and integrate both sides at identical physical quadrature points.

## Current capability

- linear `TRI3` and bilinear `QUAD4` contact facets;
- center-plane projection for curved and warped current configurations;
- deterministic convex polygon clipping and triangular mortar pallets;
- 1-, 3-, and 7-point triangle quadrature;
- local and global standard-mortar `D` and `M` matrices;
- row-wise partition-of-unity diagnostics for linear momentum conservation;
- current-configuration averaged nodal normals from Appendix A;
- analytical Appendix A nodal-normal Jacobian;
- AABB broad-phase discovery independent of projected vertex ownership;
- weighted mortar gaps and an area-normalized frictionless penalty residual;
- force and moment balance diagnostics;
- frozen-pair and frozen-active-set numerical tangent oracle;
- analytical fixed-`D/M` penalty-force tangent;
- topology-frozen moving-`D/M` Jacobian and complete smooth tangent oracle.

The package does **not** yet solve a complete contact boundary-value problem. The fully analytical moving-overlap derivatives, bulk finite elements, augmented-Lagrange enforcement, friction, and dual multipliers are tracked in [the implementation plan](docs/implementation-plan.md).

The residual equations and normalization boundary are documented in [the frictionless contact note](docs/frictionless-contact-residual.md). The tangent decomposition is documented in [the linearization note](docs/consistent-linearization.md) and [the moving-overlap note](docs/moving-overlap-tangent.md).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

## Principal source

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for large deformation solid mechanics,” *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629. DOI: `10.1016/j.cma.2003.10.010`.

See [the literature review](docs/literature-review.md) for the formulation boundary and follow-up sources.
