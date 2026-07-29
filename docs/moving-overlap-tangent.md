# Moving-overlap tangent decomposition

The projected overlap makes the standard-mortar operators configuration dependent:

```math
D = D(\mathbf x^s,\mathbf x^m), \qquad
M = M(\mathbf x^s,\mathbf x^m).
```

This slice adds a verification-level derivative of those operators while retaining the current facet-pair set. The overlap polygon, pallet decomposition, inverse maps, and quadrature values are rebuilt for each centered perturbation. Broad-phase candidates and unilateral activity remain frozen, so the result describes one smooth Newton branch rather than a topology or active-set event.

## Operator derivative

`numerical_mortar_weight_jacobian` returns

```math
\mathrm dD, \qquad \mathrm dM,
```

with one derivative column per slave and master coordinate degree of freedom. The derivative-level partition-of-unity identity is checked directly:

```math
\sum_B \mathrm dD_{AB} - \sum_C \mathrm dM_{AC} = 0.
```

The same operator derivative must vanish under a common rigid translation of both surfaces.

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

`moving_mortar_contact_tangent` assembles these terms together with the analytical Appendix A normal derivative and analytical penalty law.

## Verification and remaining boundary

For smooth partial-overlap configurations, the decomposed tangent is compared against `numerical_contact_tangent` with frozen facet pairs and active rows. A 30-case randomized warped-`QUAD4` sweep produced a maximum relative Frobenius error of `1.35e-10` in the development environment.

The geometric operator derivative in this slice is deliberately numerical. It establishes the exact tensor interface and force-law assembly that the fully analytical Section 4 / Appendix B implementation must match. The next slices will replace the numerical columns with derivatives of the projection plane, projected vertices, clipping intersections, pallets, inverse maps, and quadrature shape values.
