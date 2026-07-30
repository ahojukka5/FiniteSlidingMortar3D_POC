# Centroid-fan pallet linearization

The convex intersection polygon is divided into triangular pallets by connecting
its arithmetic centroid to every polygon edge. Once clipping topology is frozen,
this construction is smooth as long as the intersection remains counterclockwise
and every pallet retains positive signed area.

## Polygon and centroid derivatives

For intersection vertices \(\mathbf p_i\), the fan center is

```math
\mathbf c = \frac{1}{n}\sum_{i=1}^{n}\mathbf p_i,
\qquad
\mathrm d\mathbf c = \frac{1}{n}\sum_{i=1}^{n}\mathrm d\mathbf p_i.
```

The signed polygon area is evaluated with the shoelace formula,

```math
A = \frac{1}{2}\sum_i \mathbf p_i \times \mathbf p_{i+1},
```

and differentiated directly:

```math
\mathrm dA = \frac{1}{2}\sum_i
\left(
\mathrm d\mathbf p_i \times \mathbf p_{i+1}
+ \mathbf p_i \times \mathrm d\mathbf p_{i+1}
\right).
```

`polygon_signed_area_linearized` exposes this value and derivative independently
of the pallet decomposition.

## Pallet derivatives

Pallet \(i\) uses vertices

```math
(\mathbf c,\mathbf p_i,\mathbf p_{i+1}).
```

Its signed area is

```math
A_i = \frac{1}{2}
(\mathbf p_i-\mathbf c)\times(\mathbf p_{i+1}-\mathbf c).
```

The analytical derivative is

```math
\mathrm dA_i = \frac{1}{2}\left[
(\mathrm d\mathbf p_i-\mathrm d\mathbf c)
\times(\mathbf p_{i+1}-\mathbf c)
+ (\mathbf p_i-\mathbf c)
\times(\mathrm d\mathbf p_{i+1}-\mathrm d\mathbf c)
\right].
```

`linearize_centroid_fan` returns the center, every pallet vertex tensor, every
signed-area derivative, the total overlap-area derivative, and two consistency
diagnostics:

```math
\sum_i A_i = A,
\qquad
\sum_i \mathrm dA_i = \mathrm dA.
```

Both identities hold to roundoff and provide a strong verification invariant for
the next quadrature and mortar-operator derivatives.

## Facet-pair chain and event boundary

`linearize_facet_pallets` composes the analytical projection-plane,
projected-vertex, frozen-clipping, and centroid-fan layers. Jacobian columns remain
ordered as all slave coordinate DOFs followed by all master coordinate DOFs.

An empty intersection is a smooth no-overlap branch with zero pallets and zero
area derivative. One- or two-vertex intersections, clockwise polygons, and
zero-area or inverted pallets raise `PalletTopologyError`. These states are outer
geometry events and are not assigned an arbitrary smooth derivative.

The remaining analytical layers are the inverse parent maps, quadrature-point
shape values, and the assembled `D` and `M` operator Jacobians.
