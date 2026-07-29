# Consistent linearization: first analytical slice

This note records the exact boundary of the first analytical contact tangent. It differentiates the smooth penalty-force law while freezing the current mortar operators `D` and `M`, the integrated facet-pair set, and the unilateral active rows. The projected-overlap derivatives from Section 4 and Appendix B remain a later slice.

## Appendix A nominal-normal derivative

For a slave node `A`, the implemented Puso–Laursen nominal normal is

```math
\mathbf m_A = s\,\frac{1}{\left\|\sum_j \mathbf n'_{j}(0)\right\|}
\sum_j \frac{\mathbf v_j \times \mathbf v_{j+1}}
{\left\|\mathbf v_j(0) \times \mathbf v_{j+1}(0)\right\|},
```

where `s` is the surface orientation sign. Both denominators are fixed in the reference configuration. Therefore

```math
\mathrm d\mathbf m_A = s\,\frac{1}{\left\|\sum_j \mathbf n'_{j}(0)\right\|}
\sum_j \frac{
\mathrm d\mathbf v_j \times \mathbf v_{j+1}
+ \mathbf v_j \times \mathrm d\mathbf v_{j+1}}
{\left\|\mathbf v_j(0) \times \mathbf v_{j+1}(0)\right\|}.
```

`averaged_nodal_normal_jacobian` assembles this tensor directly from the attached facet corners. A two-quad warped-surface test compares all 324 entries against centered differences.

## Fixed-operator contact tangent

With frozen mortar matrices, the weighted gap vector, normalized normal gap, pressure, and traction are

```math
\mathbf q_A = \sum_B D_{AB}\mathbf x^s_B - \sum_C M_{AC}\mathbf x^m_C,
```

```math
g_A = \frac{\mathbf m_A\cdot\mathbf q_A}{a_A},
\qquad a_A = \sum_B D_{AB},
```

```math
p_A = \epsilon_N\,\chi_A g_A,
\qquad
\mathbf t_A = p_A\mathbf m_A,
```

where `chi_A` is the frozen active-row indicator. Their derivatives are

```math
\mathrm d\mathbf q_A = \sum_B D_{AB}\mathrm d\mathbf x^s_B
- \sum_C M_{AC}\mathrm d\mathbf x^m_C,
```

```math
\mathrm dg_A = \frac{
\mathrm d\mathbf m_A\cdot\mathbf q_A
+ \mathbf m_A\cdot\mathrm d\mathbf q_A}{a_A},
```

```math
\mathrm d\mathbf t_A =
\epsilon_N\chi_A\,\mathrm dg_A\,\mathbf m_A
+ p_A\,\mathrm d\mathbf m_A.
```

The nodal-force derivatives follow from the same frozen distributions used by the residual:

```math
\mathrm d\mathbf f^s = D^T\mathrm d\mathbf t,
\qquad
\mathrm d\mathbf f^m = -M^T\mathrm d\mathbf t.
```

## Verification boundary

`numerical_contact_tangent(..., freeze_weights=True)` reuses the base-state `D` and `M` matrices while recomputing normals and the force law at each centered perturbation. On the warped active-contact regression, the analytical and numerical matrices agree to a relative Frobenius error below `2e-8`; a 50-case randomized affine-quad sweep produced a maximum relative error of `1.57e-10` in the development environment.

This is not yet the full Puso–Laursen tangent. The next slice must differentiate the projection plane, projected vertices, clipping polygon, pallet geometry, inverse maps, quadrature shape values, and consequently `D` and `M` themselves.
