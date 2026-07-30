# Moving-overlap tangent decomposition

The projected overlap makes the standard-mortar operators configuration dependent:

```math
D = D(\mathbf x^s,\mathbf x^m), \qquad
M = M(\mathbf x^s,\mathbf x^m).
```

Facet pairs and unilateral activity are frozen during one smooth derivative evaluation. Projection, clipping, pallet geometry, inverse maps, quadrature points, shape values, physical weights, and the assembled operators are differentiated analytically. The centered-difference operator derivative remains an independent oracle.

## Operator derivative

`analytical_mortar_weight_jacobian` returns

```math
\mathrm dD, \qquad \mathrm dM,
```

with one derivative column per slave and master coordinate degree of freedom. The derivative-level partition-of-unity identity is checked directly:

```math
\sum_B \mathrm dD_{AB} - \sum_C \mathrm dM_{AC} = 0.
```

The same operator derivative must vanish under a common rigid translation of both surfaces. The numerical function `numerical_mortar_weight_jacobian` is retained solely for centered-difference verification.

## Complete smooth residual derivative

With

```math
\mathbf q_A = \sum_B D_{AB}\mathbf x^s_B
             - \sum_C M_{AC}\mathbf x^m_C,
\qquad
 a_A = \sum_B D_{AB},
```

the moving terms are

```math
\mathrm d\mathbf q_A =
\sum_B \mathrm dD_{AB}\mathbf x^s_B
+ \sum_B D_{AB}\mathrm d\mathbf x^s_B
- \sum_C \mathrm dM_{AC}\mathbf x^m_C
- \sum_C M_{AC}\mathrm d\mathbf x^m_C,
```

```math
\mathrm da_A = \sum_B \mathrm dD_{AB}.
```

For the area-normalized normal gap,

```math
g_A = \frac{\mathbf m_A\cdot\mathbf q_A}{a_A},
```

```math
\mathrm dg_A =
\frac{(\mathrm d\mathbf m_A\cdot\mathbf q_A
+ \mathbf m_A\cdot\mathrm d\mathbf q_A)a_A
- (\mathbf m_A\cdot\mathbf q_A)\mathrm da_A}{a_A^2}.
```

The force distributions also move:

```math
\mathrm d\mathbf f^s = D^T\mathrm d\mathbf t + (\mathrm dD)^T\mathbf t,
```

```math
\mathrm d\mathbf f^m = -M^T\mathrm d\mathbf t - (\mathrm dM)^T\mathbf t.
```

`moving_mortar_contact_tangent` assembles these terms with the analytical Appendix A normal derivative and analytical penalty law. The default geometry path is analytical; `geometry_jacobian="numerical"` selects the retained verification oracle.

## Nonsmooth boundary

The analytical result describes one smooth overlap branch. Broad-phase changes, clipping classifications, zero-area pallets, singular inverse maps, and unilateral activation changes remain outer events. They are never hidden by an automatic numerical fallback.
