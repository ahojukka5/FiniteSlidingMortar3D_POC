# Frictionless mortar contact residual

## Discrete operators

For every proximate non-mortar facet `k` and mortar facet `l`, the projected overlap kernel integrates the local operators on common physical quadrature points. The local contributions are scattered into global matrices

```text
D_AB = integral N_A^s N_B^s dGamma
M_AC = integral N_A^s N_C^m dGamma.
```

The same quadrature points and weights are used for both matrices. Therefore each non-mortar row satisfies

```text
sum_B D_AB = sum_C M_AC
```

up to floating-point roundoff. This identity makes the assembled interface forces exactly self-equilibrated in translation.

Facet-pair discovery is deliberately independent of projected node, edge, or vertex ownership. Every pair whose current axis-aligned bounding boxes are inside the search band is clipped and integrated independently. This carries the principal robustness lesson from the repaired two-dimensional implementation into the three-dimensional design.

## Appendix A nodal normals

At each non-mortar node, oriented corner area vectors from all attached facets are accumulated. A current corner vector is divided by the magnitude of its reference counterpart before accumulation, and the resulting nodal sum is divided by the magnitude of the reference nodal sum. This is the nominal current normal of Puso and Laursen's Eqs. (A.3)-(A.4); it is generally not unit length after deformation.

## Gap and penalty law

The exact weighted mortar gap vector is

```text
gbar_A = sum_B D_AB x_B^s - sum_C M_AC x_C^m.
```

The implementation records both the vector and its nominal normal component

```text
gbar_n,A = m_A dot gbar_A.
```

The paper treats this quantity as a weighted volume-like constraint. For this verification slice, the penalty parameter is made independent of the nodal tributary area by defining

```text
w_A = sum_B D_AB
g_n,A = gbar_n,A / w_A
p_A = epsilon_n max(g_n,A, 0).
```

Rows with zero support have zero gap and pressure. This area normalization is an explicit implementation boundary, not a claim that the paper uses the same regularization. The later augmented-Lagrange multiplier law can consume `gbar_n` directly while reusing the same geometry and force assembly.

## Interface forces

With nodal traction vectors `t_A = p_A m_A`, the two sides receive

```text
f^s = D^T t
f^m = -M^T t.
```

The row consistency identity gives `sum f^s + sum f^m = 0` by construction. The evaluation object reports both force and moment balance. The 2004 formulation omits terms that would be required for exact angular-momentum conservation on arbitrary nonconforming curved interfaces, so moment balance is a diagnostic rather than a guaranteed identity.

## Numerical tangent oracle

The dense centered-difference tangent is intended only for verification-sized models. By default it freezes

- the set of integrated facet pairs; and
- the unilateral active rows.

This isolates the smooth current branch from broad-phase, overlap-topology, and contact-activation events. Phase 3 will compare every analytical geometric derivative against this oracle before the nonlinear solver is introduced.
