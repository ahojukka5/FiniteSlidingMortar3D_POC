# FiniteSlidingMortar3D_POC

Research implementation of three-dimensional finite-sliding mortar contact methods.

The first target is the frictionless large-deformation segment-to-segment formulation of Puso and Laursen (2004). The implementation begins with the geometric operation on which the method depends: project an overlapping facet pair to the non-mortar center plane, clip the polygons, triangulate the overlap, and integrate both sides at identical physical quadrature points.

## Current capability

- linear `TRI3` and bilinear `QUAD4` contact facets;
- center-plane projection for curved/warped current configurations;
- deterministic convex polygon clipping;
- triangular mortar pallets;
- 1-, 3-, and 7-point triangle quadrature;
- local standard-mortar `D` and `M` matrices;
- exact row-wise partition-of-unity check for linear momentum conservation.

The package does **not** yet solve a contact boundary-value problem. Contact activation, pressure enforcement, bulk finite elements, consistent linearization, friction, and dual multipliers are tracked in [the implementation plan](docs/implementation-plan.md).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

## Principal source

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for large deformation solid mechanics,” *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629. DOI: `10.1016/j.cma.2003.10.010`.

See [the literature review](docs/literature-review.md) for the formulation boundary and follow-up sources.
