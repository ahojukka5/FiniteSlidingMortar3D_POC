# Sandwiched-beam bending benchmark

This benchmark follows the first three-dimensional example of Puso and
Laursen, *A mortar segment-to-segment contact method for large deformation
solid mechanics* (2004), Figures 4 and 5. Their example presses two
`10 x 1` beams together with ambient pressure `p = 0.1` and then bends the
pair with an end moment. A conforming beam bends normally, whereas a
symmetric node-on-segment discretization locks when all interface constraints
are active. Their mortar models retain the bending response for either
non-mortar-side choice.

The repository model is a precisely documented unit-width extrusion of that
configuration:

- length `10`, width `1`, and thickness `1` for each beam;
- compressible neo-Hookean material with `E = 1` and `nu = 0`;
- coincident, nonmatching `QUAD4` contact surfaces at `z = 1`;
- ambient pressure `0.1` on the two exterior faces;
- a self-equilibrated axial traction couple producing end moment `0.12`;
- symmetry constraints at `x = 0` and `y = 0`, plus one vertical anchor;
- biased standard mortar contact with either the upper or lower beam as the
  slave side.

The load path first ramps the ambient pressure over parameter interval
`[0, 0.25]`. It then holds that pressure fixed while ramping the end moment
over `[0.25, 1]`. This separates establishment of supported compressive
contact from the bending response.

## Model family

`benchmarks/sandwiched_beam_model.py` defines three nonmatching contact mesh
levels and one monolithic reference mesh per level:

| level | lower cells | upper cells | monolithic cells |
| --- | --- | --- | --- |
| coarse | `4 x 1 x 1` | `5 x 2 x 1` | `6 x 2 x 2` |
| medium | `8 x 2 x 2` | `11 x 3 x 2` | `12 x 3 x 4` |
| fine | `16 x 4 x 3` | `21 x 5 x 3` | `24 x 5 x 6` |

The monolithic mesh has no internal contact boundary and supplies the
locking-free bulk reference. The two contact bodies retain duplicate nodes
on the coincident interface so that all coupling is transferred through the
production mortar operator.

The model factory validates:

- positive reference determinants for every `TET4`;
- exact contact-surface-to-bulk mappings;
- nonmatching `QUAD4` surface spaces;
- correct upper/lower slave-side reversal;
- zero net force for the ambient-pressure pair and end couple;
- the requested pressure resultant and bending moment;
- deterministic staged path values and symmetry constraints.

## Claim boundary

This model-family slice establishes geometry, boundary data, load resultants,
refinement levels, slave-side choices, and the monolithic reference. It does
**not** yet claim that the nonlinear contact solution follows the reference
or that locking is absent. Those claims require the next solver and artifact
slice:

1. solve both slave-side choices with the event-aware adaptive augmented path;
2. solve the monolithic reference through the bulk equilibrium driver;
3. measure moment-displacement, curvature, pressure, gap, KKT, reaction, and
   force/moment-balance histories;
4. export selected VTK states and deterministic plots;
5. keep full three-level publication runs behind an explicit manual command,
   while automatic CI runs only unit and bounded smoke checks.
