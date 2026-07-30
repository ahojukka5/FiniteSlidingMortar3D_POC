# Analytical standard-mortar operator linearization

This layer closes the smooth geometric derivative of the standard-mortar operators. Every quadrature point already provides

- the slave shape vector `Ns` and its coordinate Jacobian;
- the master shape vector `Nm` and its coordinate Jacobian;
- the physical pallet weight `w` and its coordinate Jacobian.

The local operators are

```math
D = \sum_q w_q\,N_q^s (N_q^s)^T,
```

```math
M = \sum_q w_q\,N_q^s (N_q^m)^T.
```

## Local product rule

For one coordinate variation,

```math
\mathrm dD = \sum_q
\left[
\mathrm dw_q\,N_q^s(N_q^s)^T
+ w_q\,\mathrm dN_q^s(N_q^s)^T
+ w_q\,N_q^s(\mathrm dN_q^s)^T
\right],
```

```math
\mathrm dM = \sum_q
\left[
\mathrm dw_q\,N_q^s(N_q^m)^T
+ w_q\,\mathrm dN_q^s(N_q^m)^T
+ w_q\,N_q^s(\mathrm dN_q^m)^T
\right].
```

`integrate_facet_pair_linearized` evaluates these expressions for all local coordinate columns. Its diagnostics verify

```math
\sum_B D_{AB}=\sum_C M_{AC},
```

and

```math
\sum_B \mathrm dD_{AB}=\sum_C \mathrm dM_{AC}.
```

The sum of either local matrix equals the projected overlap area, and the sum of either operator Jacobian equals the overlap-area Jacobian.

## Global scatter

`analytical_mortar_weight_jacobian` first freezes the set of currently integrated facet pairs. For every pair, local columns are ordered as

```text
slave facet node-major xyz, then master facet node-major xyz.
```

They are scattered into the global ordering

```text
all slave node-major xyz, then all master node-major xyz.
```

Shared nodes and repeated facet support therefore accumulate into the same global tensor entries. The value matrices rebuilt from the analytical quadrature chain are compared with the ordinary residual assembly; `value_consistency_error` exposes any divergence between the two paths.

## Contact tangent

`moving_mortar_contact_tangent` now uses the analytical operator Jacobian by default. The previous centered-difference operator derivative remains available with

```python
moving_mortar_contact_tangent(pair, geometry_jacobian="numerical")
```

This retained path is a verification oracle, not an automatic fallback. Topology events, singular inverse maps, and degenerate pallets remain explicit errors so a Newton iteration cannot silently switch derivative branches.

## Remaining nonsmooth boundary

The smooth branch is analytically complete. Exact on-edge, edge-on-edge, zero-area, and appearing/disappearing overlap states still require an explicit generalized derivative and event policy before production nonlinear solves.
