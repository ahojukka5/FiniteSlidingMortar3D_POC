# Analytical projection-plane linearization

This slice replaces the first part of the numerical moving-overlap oracle with analytical derivatives. It covers the non-mortar center plane and the coordinates of slave and master vertices projected into that plane. Clipping, pallet, inverse-map, and quadrature derivatives remain separate layers.

## Center frame

At the facet center, let the two covariant base vectors be

```math
\mathbf a_1 = \sum_A N_{A,1}\mathbf x_A,
\qquad
\mathbf a_2 = \sum_A N_{A,2}\mathbf x_A.
```

The implemented orthonormal frame is

```math
\mathbf e_1 = \frac{\mathbf a_1}{\|\mathbf a_1\|},
\qquad
\mathbf n = \frac{\mathbf a_1\times\mathbf a_2}
                  {\|\mathbf a_1\times\mathbf a_2\|},
\qquad
\mathbf e_2 = \frac{\mathbf n\times\mathbf e_1}
                  {\|\mathbf n\times\mathbf e_1\|}.
```

The plane origin is the facet-center interpolation

```math
\mathbf x_c = \sum_A N_A\mathbf x_A.
```

For any nonzero vector `v`, normalization is differentiated with

```math
\mathrm d\widehat{\mathbf v}
= \frac{\mathbf I-\widehat{\mathbf v}\otimes\widehat{\mathbf v}}
       {\|\mathbf v\|}\,\mathrm d\mathbf v.
```

`facet_projection_plane_jacobian` applies this identity through the two covariant vectors, their cross product, and the final in-plane cross product. Every returned tensor uses axes `(frame_component, facet_node, node_component)`.

## Projected vertices

For a point `x` and relative vector

```math
\mathbf r = \mathbf x-\mathbf x_c,
```

the projected coordinates are

```math
\bar x_1 = \mathbf r\cdot\mathbf e_1,
\qquad
\bar x_2 = \mathbf r\cdot\mathbf e_2.
```

Their variations are

```math
\mathrm d\bar x_1
= (\mathrm d\mathbf x-\mathrm d\mathbf x_c)\cdot\mathbf e_1
+ \mathbf r\cdot\mathrm d\mathbf e_1,
```

```math
\mathrm d\bar x_2
= (\mathrm d\mathbf x-\mathrm d\mathbf x_c)\cdot\mathbf e_2
+ \mathbf r\cdot\mathrm d\mathbf e_2.
```

`project_to_plane_jacobian` deliberately returns two variable groups:

- derivatives caused by motion of the non-mortar plane-defining nodes;
- direct derivatives caused by motion of the projected points.

For slave vertices projected onto their own facet plane, the groups share the same nodal variables and are added. For master vertices, they remain separate and map naturally to slave and master blocks of the contact tangent.

## Verification boundary

All frame fields and projected coordinates are checked column by column against centered differences for both `TRI3` and warped `QUAD4` facets. A randomized 200-configuration sweep produced maximum absolute column errors of `7.73e-10` for the plane frame and `1.31e-9` for self-projected vertices in the development environment.

The next analytical layer is topology-frozen convex clipping. Each intersection vertex must retain its generating edge pair so its derivative can be evaluated without reclassifying the clipping topology inside one Newton derivative.
