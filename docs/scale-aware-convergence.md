# Scale-aware convergence and interface-local penalties

The coupled solver carries dimensional residuals because they are needed for physical
interpretation, but absolute tolerances expressed directly in force, length, or pressure
units cannot be compared across unit systems or models of different size. This layer adds
reference scales and dimensionless diagnostics without changing the contact residual or its
analytical tangent.

## Problem scales

For a neo-Hookean material with shear modulus \(\mu\) and bulk modulus \(K\), the pressure
scale is Young's modulus reconstructed as

\[
E_* = \frac{9K\mu}{3K+\mu}.
\]

The global length scale \(L_*\) is the diagonal of the reference-mesh bounding box. The
force and energy scales are

\[
F_* = E_* L_*^2,
\qquad
W_* = E_* L_*^3.
\]

The implementation also compares these values with the sum of the interface force and
energy scales and retains the larger positive value. Thus a model dominated by a broad
contact interface is not normalized by an artificially small bulk extent.

## Interface scales

Every penalty-controlled interface exposes positive reference tributary areas \(A_i\) for
its slave rows. Production mortar interfaces obtain these by distributing each reference
`TRI3` or `QUAD4` facet area equally among its facet nodes. Define

\[
A_\Gamma = \sum_i A_i,
\qquad
L_\Gamma = \sqrt{A_\Gamma},
\qquad
h_i = \sqrt{A_i}.
\]

The interface scales are

\[
p_* = E_*,
\quad
F_\Gamma = p_* A_\Gamma,
\quad
W_\Gamma = F_\Gamma L_\Gamma,
\quad
\kappa_* = \frac{p_*}{\operatorname{median}(h_i)}.
\]

`normal_penalty / kappa_*` is reported as a dimensionless conditioning indicator. The
adaptive controller bounds an increased scalar interface penalty between factors of
\(p_*/\max h_i\) and \(p_*/\min h_i\). In normalized mode the dimensionless
maximum factor is the active upper bound; the legacy dimensional `maximum_penalty` cap is
retained only when normalized scaling is disabled.

## Normalized residuals

For a dimensional free equilibrium residual \(r\),

\[
\widehat r = \frac{\lVert r\rVert}{F_*}.
\]

For one interface, the KKT maxima are normalized according to their physical units:

\[
\widehat g = \frac{g}{L_\Gamma},
\qquad
\widehat\lambda = \frac{\lambda}{p_*},
\qquad
\widehat c = \frac{|\lambda g|}{p_* L_\Gamma},
\qquad
\widehat r_{\rm proj} = \frac{r_{\rm proj}}{p_*}.
\]

`ScaleAwareAugmentedContactResult` retains dimensional quantities and adds normalized
Newton and augmentation histories. Scale-aware stopping is opt-in through
`ScaleAwareConvergenceOptions(enabled=True)`. Leaving it disabled preserves the previous
dimensional tolerances exactly.

## Interface-local penalty updates

A failed augmented solve is eligible for penalty escalation only when it ends because the
augmentation limit was reached. The controller evaluates every interface independently:

1. compute dimensional and normalized maximum penetration;
2. compare against the configured dimensional or normalized target;
3. increase only unresolved interfaces;
4. clamp each proposal to its material/mesh-derived bounds;
5. retry the same path state with the equilibrated displacement and multiplier predictor;
6. commit the changed penalties only when the retried candidate converges.

Each adaptive attempt records the dimensional and normalized equilibrium residual,
per-interface penetrations, penalty ratios before and after the attempt, and an explicit
reason for every penalty change. A later cutback restores the complete last accepted
problem, including its interface-local penalties.

## Unit transformation

Under a change of length and pressure units

\[
x' = a x,
\qquad
p' = b p,
\]

force, energy, and normal penalty transform as

\[
F' = b a^2 F,
\qquad
W' = b a^3 W,
\qquad
\kappa' = \frac{b}{a}\kappa.
\]

All normalized Newton, KKT, penetration, and penalty-ratio diagnostics are invariant under
this transformation. The committed regression compares an `m-Pa` representation with an
`mm-MPa` representation and verifies identical decisions: the resolved interface remains
unchanged while only the under-resolved fine interface receives a bounded increase.
