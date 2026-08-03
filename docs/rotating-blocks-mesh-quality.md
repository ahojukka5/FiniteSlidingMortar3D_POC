# Rotating-blocks mesh-quality monitor

The rotating-blocks benchmark evaluates every accepted nonlinear state for bulk
mesh validity. This separates a contact-topology or nonlinear-solver failure from a
state in which the deformable support mesh has become singular or excessively
distorted.

## Recorded measures

Each TET4 element evaluation provides the deformation-gradient determinant

```math
J_e = \det(F_e)
```

and the neo-Hookean strain-energy density `psi_e`. The determinant is already
scale-independent: `J_e = 1` is the reference volume, `0 < J_e < 1` is local
compression, and `J_e <= 0` is singular or inverted.

Energy is normalized by the material shear modulus,

```math
\widehat{\psi}_e = \psi_e / \mu,
```

so the quality thresholds do not depend on the selected engineering units or on a
uniform rescaling of the material stiffness.

For every accepted state, `tables/mesh-quality.csv` records:

- the minimum `J_e`, global element index, body name, and body-local index;
- minimum and maximum energy density;
- the maximum normalized energy density and responsible element;
- continuation and phase parameters; and
- the resulting `accepted`, `warning`, or `failed` classification.

The combined summary retains the worst continuation state for both Jacobian and
energy measures. A gate failure therefore identifies the element, body, body-local
index, and continuation parameter rather than only reporting a global minimum.

## Versioned limits

Quick and full profiles currently use the same physical limits:

| Measure | Warning | Failure |
| --- | ---: | ---: |
| minimum deformation Jacobian | `0.50` | `0.05` |
| maximum `psi / mu` | `0.50` | `5.0` |

Any `J_e <= 0` is an inversion and always fails independently of the configurable
near-singularity limit. Warning states remain valid benchmark states but are counted
and exposed in the summary.

Threshold changes must update the schema, focused tests, and this document in the
same pull request. The limits are numerical acceptance data; machine-dependent
runtime values are not involved.

## Refinement comparison

The monitor evaluates every configured continuation-resolution level. Medium and
fine minimum-Jacobian and maximum-normalized-energy histories are interpolated to
the existing common refinement grid. The current absolute agreement limits are
`0.05` for minimum Jacobian and `0.10` for normalized energy density.

The comparison is written to `tables/refinement-mesh-quality.csv` and visualized in
`plots/mesh-quality-refinement.svg`. Adaptive cutbacks remain part of the underlying
solver evidence and do not alter the requested refinement grid.

## Checkpoint fields and gate behavior

The existing volume VTU checkpoint files expose cell arrays named `jacobian` and
`energy_density`. The monitor adds full accepted-state histories rather than relying
on the bounded checkpoint set.

`mesh-quality.json`, both quality tables, and the two SVG plots are required by the
artifact manifest. The common acceptance gate includes production quality, the
minimum-Jacobian limit, and refinement-quality agreement. All other gate categories
are still evaluated before a nonzero command exit.

This evidence covers the current finite-strain TET4, compressible neo-Hookean,
frictionless normal-contact benchmark. It does not establish quality limits for
higher-order elements, plasticity, frictional heating, or remeshing.
