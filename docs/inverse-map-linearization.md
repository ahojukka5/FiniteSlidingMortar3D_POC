# Inverse-map and quadrature linearization

The projected overlap pallets define physical quadrature points in the slave
projection plane. Each point must be mapped back to both facet parent domains
before the standard-mortar operators can be evaluated.

## Implicit inverse map

For either a projected `TRI3` or `QUAD4` facet, the parent coordinate
\(\boldsymbol\xi\) satisfies

```math
\mathbf r(\boldsymbol\xi)
=
\sum_A N_A(\boldsymbol\xi)\,\mathbf x_A
-
\mathbf x_q
=
\mathbf 0.
```

The two-dimensional parent Jacobian is

```math
\mathbf J
=
\frac{\partial\mathbf r}{\partial\boldsymbol\xi}
=
\sum_A \mathbf x_A \otimes
\frac{\partial N_A}{\partial\boldsymbol\xi}.
```

On one smooth, nonsingular inverse-map branch,

```math
\mathrm d\boldsymbol\xi
=
\mathbf J^{-1}
\left(
\mathrm d\mathbf x_q
-
\sum_A N_A\,\mathrm d\mathbf x_A
\right).
```

`inverse_map_2d_linearized` applies this formula after solving the base inverse
map. A singular projected facet or a nonconvergent bilinear inverse map raises
`InverseMapTopologyError`; the implementation does not regularize a folded or
singular parametric map silently.

The shape-value derivative follows directly:

```math
\mathrm dN_A
=
\frac{\partial N_A}{\partial\boldsymbol\xi}
\cdot
\mathrm d\boldsymbol\xi.
```

Two consistency identities are checked for every result:

```math
\sum_A N_A = 1,
\qquad
\sum_A \mathrm dN_A = 0,
```

and

```math
\sum_A \left(
\mathrm dN_A\,\mathbf x_A
+
N_A\,\mathrm d\mathbf x_A
\right)
=
\mathrm d\mathbf x_q.
```

## Moving pallet quadrature

For a barycentric pallet quadrature coordinate \(\boldsymbol\lambda_q\),

```math
\mathbf x_q
=
\sum_{a=1}^{3}
\lambda_{qa}\,\mathbf p_a,
\qquad
\mathrm d\mathbf x_q
=
\sum_{a=1}^{3}
\lambda_{qa}\,\mathrm d\mathbf p_a.
```

The normalized triangle rules have weights whose sum is one. Therefore the
physical integration factor and its derivative are

```math
w_q = \widehat w_q A_p,
\qquad
\mathrm dw_q = \widehat w_q\,\mathrm dA_p.
```

`linearize_facet_quadrature` composes the complete smooth chain:

1. slave-defined projection plane;
2. projected slave and master vertices;
3. frozen convex-clipping operations;
4. centroid-fan pallet vertices and signed areas;
5. pallet quadrature points;
6. slave and master inverse parent maps;
7. slave and master shape values;
8. physical integration weights.

All Jacobian columns use the established ordering: every slave coordinate first,
then every master coordinate, with node-major `xyz` components.

## Verification boundary

Direct `TRI3` and warped `QUAD4` inverse maps are compared column-by-column
against centered differences. The full facet-chain regression rebuilds
projection coordinates, replays the frozen clipping topology, reconstructs the
centroid fan, and compares quadrature points, both parent maps, both shape-value
sets, and physical weights.

The next analytical slice can assemble the local operator derivatives directly:

```math
D_{AB}
=
\sum_q w_q N_A^s N_B^s,
\qquad
M_{AC}
=
\sum_q w_q N_A^s N_C^m.
```

Topology changes, clipping events, degenerate pallets, and singular inverse maps
remain outer nonsmooth events rather than arbitrary derivatives inside one
Newton branch.
