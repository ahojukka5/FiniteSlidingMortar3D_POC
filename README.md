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
- coupled bulk/contact Newton equilibrium with explicit contact-event restarts;
- outer augmented-Lagrange solution with exact KKT stopping criteria;
- adaptive load cutback, growth, and transactional normal-penalty escalation;
- immutable prescribed-displacement, dead-load, and mixed continuation paths;
- accepted-step reaction histories with full boundary snapshots;
- warped nonmatching production-adapter tangent regression;
- machine-readable patch, nonlinear, coupled, and continuation artifacts;
- force and moment balance diagnostics;
- retained numerical operator and residual-tangent oracles.

The package solves verification-sized coupled contact boundary-value problems. Production moving-overlap continuation benchmarks, scale-aware local penalty control, scalable sparse linear solvers, `HEX8` elements, friction, generalized clipping-event derivatives, and dual multiplier spaces are tracked in [the implementation plan](docs/implementation-plan.md).

The residual equations and normalization boundary are documented in [the frictionless contact note](docs/frictionless-contact-residual.md). The tangent decomposition is documented in [the linearization note](docs/consistent-linearization.md), [the moving-overlap note](docs/moving-overlap-tangent.md), [the projection-plane note](docs/projection-plane-linearization.md), [the clipping note](docs/clipping-linearization.md), [the pallet note](docs/pallet-linearization.md), [the inverse-map note](docs/inverse-map-linearization.md), and [the operator note](docs/operator-linearization.md). The multiplier update and KKT residuals are documented in [the augmented-Lagrange note](docs/augmented-lagrange.md). The first bulk formulation is documented in [the neo-Hookean `TET4` note](docs/tet4-neo-hookean.md), its sparse nonlinear equilibrium layer in [the equilibrium note](docs/nonlinear-equilibrium.md), the coupled driver in [the coupled-equilibrium note](docs/coupled-equilibrium.md), the continuation policy in [the adaptive-contact note](docs/adaptive-contact-continuation.md), and mixed boundary paths in [the path note](docs/mixed-load-paths.md).

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
uv run python benchmarks/adaptive_policy_regression.py --output results/adaptive-contact-policy
uv run python benchmarks/mixed_path_regression.py --output results/mixed-load-path
uv run python benchmarks/mixed_contact_onset.py --output results/mixed-contact-onset
uv run python benchmarks/warped_nonmatching_adapter.py --output results/warped-nonmatching-adapter
```

## Principal source

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for large deformation solid mechanics,” *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629. DOI: `10.1016/j.cma.2003.10.010`.

See [the literature review](docs/literature-review.md) for the formulation boundary and follow-up sources.
