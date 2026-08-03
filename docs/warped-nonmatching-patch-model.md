# Warped nonmatching patch model family

Issue #23 requires a mesh-refinement study that distinguishes contact-integration
error from bulk discretization error on a warped, nonmatching interface. The
`benchmarks/warped_patch_model.py` factory establishes the deterministic problem
family used by that campaign.

## Relation to the reference patch example

Puso and Laursen's three-dimensional patch example shows two important limits:

1. flat affine interfaces satisfy the patch test exactly;
2. arbitrarily warped quadrilateral faces introduce an integration-dependent
   pressure error, while the mortar result remains substantially more accurate
   and smoother than node-on-segment contact.

The present family therefore does not claim that a fixed curved interface should
transmit a globally homogeneous stress exactly. Instead, the distortion amplitude
scales with the largest in-plane cell width. Every finite mesh remains nonplanar,
but the sequence approaches the analytically controlled flat-interface limit.
This makes observed rates and asymptotic bias meaningful.

## Refinement profiles

The physical footprint is the unit square. Both blocks are one unit thick and
are discretized with positively oriented `TET4` elements. Their contact meshes
are independent:

| Profile | Lower cells | Upper cells | Characteristic size |
| --- | --- | --- | --- |
| `coarse` | `2 x 2 x 1` | `3 x 2 x 1` | `1/2` |
| `medium` | `3 x 3 x 2` | `4 x 3 x 2` | `1/3` |
| `fine` | `4 x 4 x 2` | `5 x 4 x 2` | `1/4` |

The upper grid also receives a boundary-preserving interior skew. Consequently,
refinement changes both the facet count and the overlap-pair pattern rather than
merely subdividing coincident grids.

The interface height is sampled from a smooth two-mode function. Its amplitude
is `warp_ratio * h`, where `h` is the profile's characteristic size. The upper
surface is the same function translated by the initial gap.

## Surface interpolation and mortar bias

The factory supports four independent surface families:

- `quad-quad`;
- `tri-quad`;
- `quad-tri`;
- `tri-tri`.

Triangular surfaces use alternating cell diagonals so that a single diagonal
orientation is not built into the convergence result. Every family can be built
with either the lower or upper body as the non-mortar/slave side. The physical
meshes and boundary conditions remain unchanged when the bias is reversed.

## Boundary-value problem

All lateral displacement components are constrained. The lower outer face is
fixed vertically and the upper outer face receives a prescribed compression.
The compression exceeds the initial gap, leaving a small axial strain after the
continuous surfaces meet. There are no dead loads.

For the flat-interface limit, the piecewise affine vertical displacement is
known exactly. It:

- satisfies every prescribed degree of freedom;
- closes the continuous interface without penetration;
- produces the same homogeneous deformation gradient in both blocks;
- defines a positive compressive first-Piola reference pressure through the
  package's neo-Hookean material law.

The model exposes this field through `manufactured_displacement()` and the
reference traction through `reference_pressure()`. These quantities will serve
as initial data and error oracles for the solve-and-report stage.

## Validation in this slice

The accompanying tests verify:

- monotone mesh refinement with nonmatching interface counts;
- positive reference determinants at all levels;
- all `TRI3`/`QUAD4` combinations;
- both mortar-side choices and their global node maps;
- nonplanarity at every finite level and decreasing warp amplitude;
- exact satisfaction of prescribed values by the manufactured field;
- pointwise closure of the analytical continuous interface;
- positive, profile-independent flat-limit pressure and reaction.

## Remaining work for issue #23

This model-factory slice intentionally leaves issue #23 open. The next slice will
run the augmented-Lagrange contact solve over the profile, surface-family, and
bias matrix; write pressure, gap, displacement, reaction, overlap, balance, and
KKT fields; estimate observed rates; and add bounded quick-regression thresholds
plus publication-oriented full output.

## Reference

M. A. Puso and T. A. Laursen, “A mortar segment-to-segment contact method for
large deformation solid mechanics,” *Computer Methods in Applied Mechanics and
Engineering* 193 (2004), 601–629.
