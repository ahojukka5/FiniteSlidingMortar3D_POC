# Projected augmented-Lagrange normal contact

Puso and Laursen formulate nodal mortar constraints with a scalar normal gap and a nonnegative nodal pressure. Their numerical examples use augmented-Lagrange enforcement, while Section 4 presents the penalty pressure and states that the multiplier extensions follow from the same force linearization.

This implementation keeps the multiplier update separate from the equilibrium solve. That separation is important: multipliers are fixed during one inner Newton solve and are projected only after an accepted equilibrium state.

## Sign convention

The package reports positive `normal_gaps` for penetration. The nodal Kuhn–Tucker conditions are therefore

```math
g_A \le 0,
\qquad
\lambda_A \ge 0,
\qquad
\lambda_A g_A = 0.
```

Here `lambda` has pressure units because the mortar gap is normalized by the mortar row area before enforcement.

## Inner pressure law

For accepted multiplier state `lambda^k` and penalty parameter `kappa`, the trial and projected pressures are

```math
p_A^{\mathrm{trial}} = \lambda_A^k + \kappa g_A,
```

```math
p_A = \max(0, p_A^{\mathrm{trial}}).
```

`evaluate_augmented_lagrange` evaluates this pressure without changing the accepted multiplier state. When an active set is supplied explicitly, the routine evaluates the selected smooth branch without reprojecting it. This is used only for tangent verification and Newton linearization.

A zero multiplier state reproduces the existing penalty law exactly.

## Accepted augmentation

After the current equilibrium problem has converged, `augment_multipliers` performs

```math
\lambda_A^{k+1}
= \max(0, \lambda_A^k + \kappa g_A).
```

Rows with no mortar support receive zero multiplier. The update is explicit and returns a new immutable state, the multiplier increment, and diagnostics before and after augmentation. No multiplier changes occur implicitly during residual evaluation.

## KKT diagnostics

`kkt_diagnostics` reports separate nodal residual blocks:

```math
r_A^{\mathrm{gap}} = \max(g_A,0),
```

```math
r_A^{\mathrm{dual}} = \max(-\lambda_A,0),
```

```math
r_A^{\mathrm{comp}} = |\lambda_A g_A|,
```

```math
r_A^{\mathrm{proj}}
= \left|\lambda_A
- \max(0,\lambda_A+\kappa g_A)\right|.
```

Unsupported multiplier magnitude is reported separately. The projection residual is a fixed-point form of the complete complementarity system and is useful as an outer augmentation stopping criterion. Gap, complementarity, projection, and multiplier tolerances remain separate because their physical units differ.

## Consistent tangent

During an inner Newton solve, `lambda^k` is fixed. On a frozen active branch,

```math
\mathrm dp_A = \kappa\,\mathrm dg_A.
```

The multiplier offset still contributes through the current traction:

```math
\mathbf t_A = p_A \mathbf m_A.
```

Consequently it enters both the nominal-normal term and the moving `D`/`M` force-distribution terms. `augmented_lagrange_contact_tangent` combines these terms with the existing analytical geometry Jacobian while keeping the accepted multipliers fixed. The retained numerical oracle verifies the complete assembled derivative independently.

`numerical_augmented_lagrange_tangent` keeps multipliers fixed and can freeze facet pairs, active rows, and mortar weights. It is an independent centered-difference oracle for the analytical augmented tangent.

## Outer coupled driver

`solve_augmented_contact` now applies the projected law to complete bulk/contact boundary-value problems. Every outer iteration first converges the global displacement equilibrium with all accepted multipliers fixed. The KKT blocks are then checked on every interface. Only a failed KKT check permits the next projected multiplier update.

The driver records

- every fixed-multiplier Newton result;
- contact-branch restarts inside those Newton solves;
- penetration, complementarity, and projection maxima;
- active-row counts and maximum pressure; and
- the largest accepted multiplier increment.

The returned multiplier tuple always corresponds to the returned equilibrium state. When the maximum augmentation count is reached, the code does not expose a projected state that has not subsequently been equilibrated.

## Nonsmooth boundary

A trial pressure exactly equal to zero is an active-set event. Broad-phase changes, clipping events, zero-area pallets, singular inverse maps, and multiplier projection events remain outside one smooth derivative. The implementation does not silently switch to a numerical fallback at these states.

The coupled Newton driver either restarts from an accepted state on the newly detected branch or rejects the branch-crossing line-search trial, according to the selected event policy. The policy moves between smooth branches; it does not assign an arbitrary derivative at the event itself.
