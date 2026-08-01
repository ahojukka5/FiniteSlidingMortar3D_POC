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
- deterministic refittable facet BVH with a quadratic discovery oracle and diagnostics;
- weighted mortar gaps and area-normalized frictionless penalty enforcement;
- projected augmented-Lagrange multiplier state and updates;
- primal, dual, complementarity, and projection-form KKT diagnostics;
- analytical penalty and augmented-Lagrange smooth-branch tangents;
- finite-strain compressible neo-Hookean energy, first Piola stress, and material tangent;
- positively oriented total-Lagrangian `TET4` elements with analytical residuals and tangents;
- dense and deterministic CSR `TET4` mesh assembly with independent derivative oracles;
- configurable dense, sparse-LU, GMRES, and BiCGSTAB linear-system backends;
- linear-solver residual, timing, storage, and failure diagnostics;
- medium coupled dense/sparse/Krylov scaling benchmark with CSR storage checks;
- strong essential boundary conditions, dead nodal loads, and reaction extraction;
- coupled bulk/contact Newton equilibrium with explicit contact-event restarts;
- typed pair, clipping, pallet, support, and pressure event localization;
- deterministic left/right branch selection and exact post-event Newton restart;
- outer augmented-Lagrange solution with exact KKT stopping criteria;
- adaptive load cutback, growth, and transactional normal-penalty escalation;
- immutable prescribed-displacement, dead-load, and mixed continuation paths;
- accepted-step reaction histories with full boundary snapshots;
- unit-consistent bulk/interface force, pressure, energy, and penalty scales;
- dimensional and normalized Newton/KKT convergence histories;
- bounded interface-local normal-penalty adaptation with explicit update reasons;
- warped nonmatching production-adapter tangent regression;
- fully coupled warped nonmatching contact-onset continuation benchmark;
- versioned benchmark manifests with Git, runtime, package, seed, and solver provenance;
- ParaView-readable TET4 volume, contact-surface, and projected-overlap VTK output;
- reusable validated SVG line, bar, event, polygon, mesh, and sparsity plots;
- checked profile-aware numeric golden specifications with explicit tolerances;
- standardized patch, bulk, coupled, adaptive, mixed-path, onset, scale-aware,
  production-interface, warped production-onset, topology-event, BVH, and linear-solver
  benchmark suite;
- machine-readable patch, nonlinear, coupled, continuation, scaling, event, search, solver,
  and golden-regression artifacts;
- force and moment balance diagnostics;
- retained numerical operator and residual-tangent oracles.

The package solves verification-sized coupled moving-overlap contact boundary-value problems. Sparse linear backends are wired into the bulk, coupled-contact, and event-localized Newton drivers and exercised by a reproducible medium coupled scaling study. `HEX8` elements, friction, adaptive event propagation, generalized clipping-event derivatives, and dual multiplier spaces are tracked in [the implementation plan](docs/implementation-plan.md).

The residual equations and normalization boundary are documented in [the frictionless contact note](docs/frictionless-contact-residual.md). The tangent decomposition is documented in [the linearization note](docs/consistent-linearization.md), [the moving-overlap note](docs/moving-overlap-tangent.md), [the projection-plane note](docs/projection-plane-linearization.md), [the clipping note](docs/clipping-linearization.md), [the pallet note](docs/pallet-linearization.md), [the inverse-map note](docs/inverse-map-linearization.md), and [the operator note](docs/operator-linearization.md). The multiplier update and KKT residuals are documented in [the augmented-Lagrange note](docs/augmented-lagrange.md). The first bulk formulation is documented in [the neo-Hookean `TET4` note](docs/tet4-neo-hookean.md), its sparse nonlinear equilibrium layer in [the equilibrium note](docs/nonlinear-equilibrium.md), the linear backend boundary in [the solver note](docs/linear-solver-backends.md), its medium coupled evidence in [the solver scaling note](docs/linear-solver-scaling.md), the coupled driver in [the coupled-equilibrium note](docs/coupled-equilibrium.md), the continuation policy in [the adaptive-contact note](docs/adaptive-contact-continuation.md), mixed boundary paths in [the path note](docs/mixed-load-paths.md), unit-consistent convergence in [the scaling note](docs/scale-aware-convergence.md), the first production boundary-value problem in [the warped-onset note](docs/warped-nonmatching-contact-onset.md), explicit topology localization in [the event note](docs/contact-topology-events.md), incremental candidate discovery in [the BVH note](docs/broad-phase-bvh.md), and versioned result directories, plots, VTK output, and golden checks in [the benchmark artifact note](docs/benchmark-artifacts.md).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

Install only the optional sparse runtime when development tools are not needed:

```bash
uv sync --extra sparse
```

Run and validate the complete standardized benchmark suite with published settings:

```bash
uv run python benchmarks/run_standardized.py \
  --output results/standardized-benchmarks
```

This validates every manifest and applicable checked golden metric and writes both
`suite-summary.json` and `golden-regressions.json`.

Use the bounded quick profile for integration checks while retaining every benchmark family:

```bash
uv run python benchmarks/run_standardized.py \
  --quick \
  --output results/standardized-benchmarks-quick
```

Regenerate the verification benchmarks individually with:

```bash
uv run python benchmarks/tet4_patch.py --output results/tet4-patch
uv run python benchmarks/nonlinear_equilibrium.py --output results/nonlinear-equilibrium
uv run python benchmarks/coupled_mortar_patch.py --output results/coupled-mortar-patch
uv run python benchmarks/adaptive_policy_regression.py --output results/adaptive-contact-policy
uv run python benchmarks/mixed_path_regression.py --output results/mixed-load-path
uv run python benchmarks/mixed_contact_onset.py --output results/mixed-contact-onset
uv run python benchmarks/scale_aware_penalty_regression.py --output results/scale-aware-penalty
uv run python benchmarks/warped_nonmatching_adapter.py --output results/warped-nonmatching-adapter
uv run python benchmarks/warped_nonmatching_contact_onset.py --output results/warped-nonmatching-contact-onset
uv run python benchmarks/topology_event_regression.py --output results/topology-events
uv run python benchmarks/broad_phase_scaling.py --output results/broad-phase-scaling
uv run python benchmarks/linear_solver_scaling.py --output results/linear-solver-scaling
```

## Principal source

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for large deformation solid mechanics,” *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629. DOI: `10.1016/j.cma.2003.10.010`.

See [the literature review](docs/literature-review.md) for the formulation boundary and follow-up sources.
