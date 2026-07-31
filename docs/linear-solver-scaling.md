# Medium coupled linear-solver scaling

Issue #20 requires more than isolated matrix tests: a coupled finite-strain
contact problem must run through the Newton drivers without converting the
global or reduced tangent to dense storage. The benchmark
`benchmarks/linear_solver_scaling.py` supplies that evidence.

## Model

The model contains two separately meshed compressible neo-Hookean blocks. Each
block is a structured grid of positively oriented `TET4` elements. The lower
face is fixed, the upper face receives a prescribed downward displacement, and
the coincident interface is coupled by one matching mortar oracle per surface
cell.

The oracle freezes overlap and normal geometry intentionally. This isolates
linear-system behavior from broad-phase, clipping, and moving-overlap costs,
which have independent production benchmarks. Facet mass matrices are scaled
by physical cell area, so refining the interface does not multiply its total
force artificially.

The default sequence is:

| surface resolution | nodes | elements | interfaces | total DOFs | free DOFs |
|---:|---:|---:|---:|---:|---:|
| 2 | 54 | 96 | 4 | 162 | 108 |
| 4 | 150 | 384 | 16 | 450 | 300 |
| 6 | 294 | 864 | 36 | 882 | 588 |

Each block has two element layers through its half-unit thickness. The combined
reference volume is one at every level.

## Backend matrix

The default run compares:

- `dense`: the retained NumPy verification oracle;
- `sparse_lu`: direct SciPy factorization of the free-free CSR block;
- `gmres_ilu`: nonsymmetric GMRES with an ILU preconditioner.

`bicgstab_ilu` is available through `--backends` for additional experiments.
Backend selection occurs entirely through `NewtonOptions.linear_solver`; the
mechanics and contact code are unchanged.

## Run

```bash
uv run python benchmarks/linear_solver_scaling.py \
  --output results/linear-solver-scaling
```

An optional BiCGSTAB comparison is:

```bash
uv run python benchmarks/linear_solver_scaling.py \
  --backends dense sparse_lu gmres_ilu bicgstab_ilu \
  --output results/linear-solver-scaling
```

## Tables

The benchmark writes:

- `models.csv`: mesh, interface, DOF, sparsity, and volume data;
- `backend-summary.csv`: one row per mesh/backend combination;
- `linear-iterations.csv`: every accepted or failed Newton linear solve;
- `summary.json`: settings, complete rows, and machine-readable acceptance
  results.

The per-iteration table records the selected backend and preconditioner,
iteration count, absolute and relative residuals, setup and solve times,
free-free dimensions and nonzeros, dense-materialization flag, failure reason,
nonlinear residual, line-search result, and contact-event state.

## Plots

The generated SVG figures are:

- total linear solve time versus free DOFs;
- total linear setup time versus free DOFs;
- accumulated linear iterations versus free DOFs;
- backend displacement difference relative to dense Newton;
- reduced CSR density versus free DOFs;
- nonlinear and linear residual histories for every backend on the largest
  model.

The script parses every generated SVG before reporting success.

## Acceptance

A default run succeeds only when:

1. every requested backend converges at every mesh level;
2. dense and sparse-LU displacement solutions agree to a relative tolerance of
   `1e-9`;
3. every non-dense backend reports zero dense materializations;
4. the largest problem contains at least 500 free DOFs.

The benchmark deliberately does not hide failed Krylov iterations or
preconditioner setup. Such failures remain in the CSV/JSON diagnostics and make
the command exit unsuccessfully.
