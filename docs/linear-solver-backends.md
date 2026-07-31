# Linear-solver backends

The global finite-element tangent is stored in the package's deterministic `CSRMatrix`
format. The linear-solver layer consumes that storage directly and keeps dense conversion
behind the explicit `dense` backend.

Bulk, coupled-contact, and event-localized Newton drivers all use this backend boundary.
The backend selection is part of `NewtonOptions`, so benchmark and application code can
change the linear algebra without changing mechanics code.

## Backends

`LinearSolverOptions.backend` accepts:

- `dense`: convert the reduced CSR block to dense storage and call NumPy;
- `sparse_lu`: factor the reduced CSR block with SciPy sparse LU;
- `gmres`: solve a general nonsymmetric system with restarted GMRES;
- `bicgstab`: solve a general nonsymmetric system with BiCGSTAB;
- `auto`: use dense storage through `dense_threshold`, otherwise choose sparse LU,
  or GMRES when a preconditioner is configured.

SciPy is optional for package users:

```bash
uv sync --extra sparse
```

The normal development environment includes SciPy so all backend tests run under
`uv run pytest`.

## Strong Dirichlet reduction

`extract_csr_submatrix` forms a deterministic reduced CSR matrix in the exact ordering of
its requested rows and columns. `solve_reduced_system` uses it to form and solve a free-free
block without materializing the complete tangent as a dense array.

```python
from contact3d import LinearSolverOptions, solve_reduced_system

result = solve_reduced_system(
    tangent,
    free_dofs,
    -residual[free_dofs],
    options=LinearSolverOptions(backend="sparse_lu"),
)
```

## Newton integration

Configure all Newton variants through the nested linear-solver options:

```python
from contact3d import LinearSolverOptions, NewtonOptions, solve_coupled_equilibrium

options = NewtonOptions(
    maximum_iterations=40,
    linear_solver=LinearSolverOptions(
        backend="gmres",
        preconditioner="ilu",
        relative_tolerance=1.0e-10,
    ),
)
result = solve_coupled_equilibrium(
    problem,
    problem.initial_states(),
    options=options,
)
```

Every accepted `NewtonIteration` or `CoupledNewtonIteration` stores its
`LinearSolveDiagnostics` as `linear_solve`. A failed backend is retained as
`linear_solve_failure` on the Newton result. Singular direct factorizations preserve the
legacy `singular_tangent` termination reason; other backend failures use
`linear_solve_failed` and retain the more specific backend failure reason.

The event-localized solver uses the same reduced solve before localizing and restarting at
a topology event. Its event-aware result preserves the same accepted and failed linear
records, and conversion to the legacy coupled result keeps those diagnostics.

## Preconditioners

GMRES and BiCGSTAB support built-in `none`, `jacobi`, and `ilu` choices. A
`preconditioner_factory` hook accepts any object implementing
`LinearPreconditioner.apply`.

Two helpers cover the planned mechanics partitions:

- `block_jacobi_preconditioner_factory` for independent body blocks;
- `field_split_preconditioner_factory` for bulk/contact-aware reduced-DOF fields.

The supplied blocks must partition every reduced row exactly once. The current helpers use
independent dense inverses and are verification implementations; scalable block solvers can
replace them behind the same protocol.

## Diagnostics

Every solve returns `LinearSolveDiagnostics` containing:

- requested and selected backend;
- selected preconditioner;
- convergence state, iteration count, and failure reason;
- final absolute and relative residuals plus residual history;
- setup and solve times;
- reduced matrix shape and nonzero count;
- whether dense storage was materialized.

A backend failure is returned as a result with `solution is None`; it is not hidden by an
automatic fallback. This keeps Newton termination and benchmark comparisons reproducible.
