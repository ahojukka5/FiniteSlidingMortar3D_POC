# Sparse bulk equilibrium and Newton globalization

This layer turns the verified finite-strain `TET4` element into a small but complete
bulk equilibrium solver. It is deliberately independent of the contact residual so
that sparse assembly, constraints, loads, and globalization can be verified before
coupling the two nonlinear systems.

## Symbolic sparse assembly

`Tet4Sparsity.from_mesh` constructs the global CSR graph once. It records the CSR
position of every entry in each 12-by-12 element tangent. Subsequent evaluations
zero only the numerical value array and scatter element matrices directly into those
precomputed positions. The sparsity pattern is therefore deterministic and reusable
throughout Newton iterations and load steps.

`assemble_tet4_sparse` returns the same energy and internal residual as the dense
verification assembler, but its tangent is a `CSRMatrix`. The dense assembler remains
an independent regression oracle.

The current linear solve extracts the free-free block from CSR and calls
`numpy.linalg.solve`. This is intentional for the present verification-sized models:
the nonlinear and assembly interfaces are sparse, while the linear algebra boundary
is explicit. A scalable sparse direct or Krylov backend can replace that final step
without changing the residual, tangent, constraint, or history contracts.

## Essential boundary conditions

`DirichletConstraints` stores unique global displacement DOFs and their prescribed
values. Constraints are imposed strongly by keeping every accepted and trial state
on the feasible affine subspace. Newton equations are solved only on complementary
free DOFs. The full residual remains available, so constrained entries are the
reaction forces after convergence.

## Dead loads and total potential

`DeadLoad` stores a configuration-independent global nodal force vector. At load
factor `alpha`, the equilibrium residual and total potential are

```math
R(u;\alpha) = f_{\mathrm{int}}(u) - \alpha f_{\mathrm{ext}},
```

```math
\Pi(u;\alpha) = U(u) - \alpha f_{\mathrm{ext}}^T u.
```

The potential is reported in every equilibrium evaluation. The line search uses the
free-residual merit function so it remains directly applicable when contact terms are
added later.

## Newton and Armijo line search

For free DOFs `f`, one Newton direction solves

```math
K_{ff}\,\Delta u_f = -R_f.
```

The line search uses

```math
\phi(u) = \frac{1}{2}\lVert R_f(u)\rVert^2
```

and accepts the first admissible step satisfying

```math
\phi(u+s\Delta u)
\le
\phi(u) - c\,s\lVert R_f(u)\rVert^2.
```

Trial states that invert an element are rejected and the step is reduced. The solver
does not silently regularize deformation gradients or tangents. Singular tangents,
line-search failure, and iteration exhaustion are returned as explicit
`termination_reason` values in `NewtonResult`.

`solve_load_steps` uses the preceding converged displacement as the predictor for a
strictly increasing sequence of load factors. Step-size adaptation is intentionally
left for the coupled contact driver.

## Verification benchmark

`benchmarks/nonlinear_equilibrium.py` constructs the 12-element cube-star mesh and a
manufactured affine deformation

```math
F =
\begin{bmatrix}
1 & 0 & 0.35\\
0 & 1 & 0.06\\
0 & 0 & 0.78
\end{bmatrix}.
```

The bottom face is fixed. External nodal dead loads are generated from the exact
affine state on free DOFs, giving a known nonlinear equilibrium solution while still
requiring Newton to recover it from the undeformed configuration.

The direct full-load solve requires step reductions down to `0.125`, converges in
eight accepted iterations, and recovers the exact displacement to roundoff. Ten nonzero load increments plus the zero-load state also converge and provide a nonlinear load-displacement path.
The benchmark writes:

- `summary.json` with mesh, material, sparsity, convergence, and balance metrics;
- `newton-history.csv` with every accepted nonlinear iteration;
- `load-steps.csv` with every converged load level;
- `newton-residual.svg` for residual convergence;
- `load-displacement.svg` for the nonlinear response path;
- `sparsity-pattern.svg` for visual inspection of the assembled CSR graph.

## Remaining boundary

This slice contains bulk-only equilibrium. The next coupling layer must add contact
residuals and tangents into the same global free-DOF system, carry accepted
augmented-Lagrange state between outer iterations, and define cutback rules for
contact topology, active-set, and KKT failures.
