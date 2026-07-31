# Linear-solver backends

The global finite-element tangent is stored in the package's deterministic `CSRMatrix`
format. The linear-solver layer consumes that storage directly and keeps dense conversion
behind the explicit `dense` backend.

This first issue-20 slice implements and verifies the backend boundary. Wiring the backend
selection and diagnostics into the bulk, coupled-contact, and event-localized Newton drivers
is intentionally left for the next slice so the solver API can be reviewed independently.

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
from contact3d.linear_solver import LinearSolverOptions, solve_reduced_system

result = solve_reduced_system(
    tangent,
    free_dofs,
    -residual[free_dofs],
    options=LinearSolverOptions(backend="sparse_lu"),
)
```

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
