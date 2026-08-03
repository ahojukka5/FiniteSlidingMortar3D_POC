# Rotating-blocks benchmark interpretation

This note defines what the rotating-blocks benchmark measures, how each result
must be interpreted, and which conclusions the evidence does not support. The
benchmark is a controlled precursor to cylindrical ironing: it combines a
deformable support, a prescribed rigid boundary motion, nonmatching mortar
surfaces, changing projected overlap topology, augmented frictionless contact,
and event-localized adaptive continuation.

A successful result means that this specific model and solver policy completed
the documented path and satisfied the versioned numerical criteria. It is not a
general proof of frictional contact, curved-surface objectivity, generalized
derivatives at exact special states, or production-scale performance.

## Physical model

### Geometry and meshes

The lower body occupies

```text
x in [-1.00, 1.00]
y in [-1.00, 1.00]
z in [ 0.00, 0.50]
```

and is deformable. The upper body occupies

```text
x in [-0.65, 0.65]
y in [-0.35, 0.35]
z in [ 0.52, 0.82]
```

and every upper node follows the prescribed rigid-body path. The initial normal
separation is therefore `0.02`. The rotation pivot is `(0, 0, 0.52)`, the center
of the upper contact plane.

Both bodies use positively oriented total-Lagrangian `TET4` elements. Every
structured hexahedral cell is split into six tetrahedra. The quick and full
profiles change only the lower-body resolution:

| Profile | Lower cells | Upper cells | Lower TET4 | Upper TET4 |
| --- | ---: | ---: | ---: | ---: |
| `quick` | `4 x 4 x 2` | `3 x 2 x 1` | 192 | 36 |
| `full` | `8 x 8 x 4` | `3 x 2 x 1` | 1536 | 36 |

The upper bottom surface is the non-mortar slave side. The lower top surface is
the master side. Their QUAD4 grids are nonmatching. The upper footprint is
rectangular, so a centered in-plane rotation changes the overlap polygons and
facet-pair set even without translation.

### Material, contact, and scales

The bulk material is the compressible logarithmic neo-Hookean model constructed
from

```text
Young modulus E = 210
Poisson ratio nu = 0.30
```

The code derives the shear modulus, bulk modulus, and first Lame parameter from
those values. The same material is assigned to the complete TET4 mesh, although
the upper block has no free deformable degree of freedom because all upper nodes
are prescribed.

The contact model is frictionless normal mortar contact with:

```text
initial normal penalty = 3200
search distance        = 0.12
triangle quadrature    = 7 points
```

The production solver may increase the normal penalty through its bounded,
interface-local, scale-aware policy. Such updates are solver actions, not changes
to the reference benchmark geometry. The search distance is a candidate-discovery
radius and is not an admissible penetration or gap tolerance.

### Boundary conditions and prescribed motion

All nodes on the lower bottom face are fixed in all three components. Every node
of the upper block is controlled in all three components. No dead nodal load is
applied; the load is introduced entirely through the prescribed upper-body
motion, and reactions are recovered on constrained degrees of freedom.

Let `s` denote the absolute continuation parameter. The path is:

1. `0 <= s <= 0.25`: translate the upper body by a linearly interpolated
   `(0, 0, -0.04)`;
2. `0.25 <= s <= 1`: hold the full compression, rotate from zero to `pi/2`
   about the global z axis through the pivot, and add a linearly interpolated
   tangential translation from zero to `(0.10, 0, 0)`.

The compression consumes the initial `0.02` separation and prescribes another
`0.02` of downward motion relative to the undeformed lower reference surface.
The lower-body deformation and pressure distribution are solved from equilibrium.

The path records `parameter`, `phase_index`, `phase_parameter`, rotation angle,
and translation components. `phase_index = 0` denotes compression and
`phase_index = 1` denotes rotation plus tangential translation. An interior phase
boundary belongs to the following phase.

## Interpretation layers

The evidence should be read in four separate layers.

1. **Geometry:** reference dimensions, mesh mappings, projected polygons, facet
   pairs, support rows, and topology signatures.
2. **Formulation:** finite-strain bulk residuals, frictionless mortar residuals,
   augmented multipliers, contact pressure, KKT measures, reactions, and balance.
3. **Solver policy:** Newton, line search, augmentation, event localization,
   adaptive cutback, penalty retry, and linear backend selection.
4. **Benchmark policy:** quick/full resolutions, checkpoint selection, numerical
   thresholds, repeated scans, and load-step refinement comparisons.

A topology event is a geometry or active-set change. A cutback is a continuation
policy action. A penalty retry is an enforcement-policy action. They must not be
reported as the same phenomenon.

## Metric dictionary

### Path and accepted-state metrics

`tables/accepted-steps.csv` contains one row for each accepted nonlinear state.

| Metric | Definition |
| --- | --- |
| `accepted_step` | One-based ordinal of the accepted state. |
| `parameter` | Absolute continuation coordinate `s`. |
| `phase_index` | Zero for compression and one for rotation/translation. |
| `phase_parameter` | Local interpolation coordinate inside the active phase. |
| `rotation_angle` | Prescribed upper-body angle about the global z axis. |
| `reaction_x/y/z` | Sum of constrained residual components over all controlled upper-body nodes. |
| `reaction_norm` | Euclidean norm of the complete constrained reaction vector retained by the accepted step. |
| `maximum_pressure` | Largest accepted slave nodal projected normal pressure. |
| `overlap_area` | Sum of integrated projected intersection areas over all active slave/master facet pairs. |
| `facet_pairs` | Number of integrated slave/master facet pairs in the accepted topology signature. |
| `supported_rows` | Number of slave mortar rows with positive geometric integration support. |
| `active_rows` | Number of supported slave rows on the active pressure branch. |
| `inner_converged` | Whether the accepted inner equilibrium solve converged. |
| `inner_termination_reason` | Typed termination reason returned by the inner solve. |

`reaction_x/y/z` are controlled-boundary reactions, not contact tractions sampled
at three points. Their signs follow the global residual convention. Comparisons
must preserve that convention rather than replacing values by magnitudes.

### Adaptive attempts and solver diagnostics

`tables/attempts.csv` contains accepted attempts, cutbacks, and penalty retries.
Its interval is defined by `start_parameter`, `target_parameter`, and `step_size`.
`action` states what the adaptive driver did with the attempted state.

The normalized equilibrium residual is the scale-aware free-equilibrium residual
reported by the production solver. The normalized maximum penetration is the
accepted maximum penetration measure divided by the solver's physical gap scale.
The benchmark gate uses the maximum values over accepted attempts; rejected
attempts remain diagnostics and do not become accepted KKT evidence.

`tables/solver-diagnostics.csv` records deterministic work counters:

- augmentations, Newton iterations, and line-search iterations;
- linear solves, linear iterations, and linear failures;
- event-localization batches and atomic events;
- selected backends and preconditioners;
- maximum matrix nonzero count and dense materializations.

The worst accepted and worst rejected/retried attempts are selected by the
lexicographic maximum of Newton iterations, linear iterations, event-localization
batches, and attempt index. Wall time, linear setup time, and linear solve time
are provenance only. They are not numerical acceptance metrics.

### Projected overlap and topology

Projected overlap is constructed by projecting each proximate slave/master facet
pair to the non-mortar center plane, clipping the two polygons, triangulating the
intersection, and integrating both sides at common physical quadrature points.

`tables/facet-pairs.csv` reports the integrated overlap area for each selected
checkpoint and facet pair. `tables/overlap-regions.csv` reports independently
inspectable slave, master, and clipped intersection polygons. `projected_area`
is the unsigned shoelace area in the local two-dimensional projection plane.
`region_kind` is zero for slave, one for master, and two for intersection.

A topology signature includes the integrated facet-pair set, supported rows,
active rows, and geometry tokens describing clipping and pallet construction.
`tables/events.csv` records localized atomic changes in absolute continuation
coordinates. Event kinds mean:

- `pair_entry` or `pair_exit`: an integrated slave/master facet pair appears or
  disappears;
- support activation or release: a slave mortar row gains or loses geometric
  integration support;
- pressure activation or release: an active pressure row changes branch;
- clipping transition: the clipped polygon vertex/edge construction changes;
- pallet transition: the centroid-fan triangulation changes.

An event location is the selected absolute continuation parameter returned by
the event-localization state machine. It is not the Newton fraction alone and it
is not the midpoint of a kinematic scan bracket.

The kinematic topology oracle holds all free bulk degrees of freedom at zero and
evaluates only prescribed geometry. It proves that the path crosses deterministic
geometry branches; it does not prove nonlinear equilibrium. The production event
history is equilibrium-dependent and is interpreted separately.

### Pressure metrics

For slave row `i`, let `A_i` be the mortar row area and `p_i` the projected
normal pressure. `tables/pressure-nodes.csv` records the current coordinate,
`A_i`, normal gap, `p_i`, augmented multiplier, support/activity flags, and the
nodal pressure measure `p_i A_i`.

The pressure resultant and aggregate measures are

```text
P        = sum_i p_i A_i
mean     = sum_i p_i A_i / sum_i A_i
variance = sum_i A_i (p_i - mean)^2 / sum_i A_i
rms      = sqrt(sum_i A_i p_i^2 / sum_i A_i)
L2       = sqrt(sum_i A_i p_i^2)
centroid = sum_i p_i A_i x_i / P
```

The centroid is omitted when `P` is numerically zero. Unsupported rows must have
zero pressure and zero accepted multiplier. The pressure resultant is compared
with the normal projection of the assembled slave contact force. Tangential
contact force and slave/master force cancellation are diagnostics for accidental
friction or node-mapping errors.

The implementation uses a penetration-positive normal-gap convention. A positive
reported gap is penetration; `maximum_separation` is the maximum of `-normal_gap`.

### Reactions, force balance, and moment balance

The applied resultant is the accepted dead-load vector, which is zero for this
prescribed-motion benchmark. The reaction resultant is the residual on constrained
degrees of freedom. Their sum is the global force-balance error. Bulk and contact
forces are internal action-reaction systems and are not added again to that global
resultant.

The interface residual is split into slave and master blocks. Their sum is the
contact force-cancellation error. Moments use current nodal coordinates and are
computed about both the global origin and the current translated rigid-motion
pivot.

Normalized force errors divide the imbalance by the largest total variation of
the relevant nodal force systems. Normalized moment errors use the largest of the
nodal moment total variation and force scale times current geometric length scale.
`tables/force-moment-balance.csv` retains the dimensional resultants, moments,
scales, and normalized errors.

### KKT and enforcement metrics

The rotating-blocks gate currently exposes two aggregate KKT-related measures:

- `normalized_equilibrium_residual`: the maximum scale-aware free-equilibrium
  residual over accepted attempts;
- `normalized_penetration`: the maximum scale-aware penetration measure over
  accepted attempts.

The complete solver evidence also retains augmented multipliers, support and
active branches, penalty updates, and inner termination reasons. The common gate
does not replace those histories with a single scalar. Passing the two aggregate
limits means only that the accepted path met the configured equilibrium and
penetration tolerances.

### Contact-retention metrics

The retention monitor applies to accepted rotation-phase states. Structural
contact requires

```text
overlap_area >= 1e-12
supported_rows >= 1
```

and load-bearing contact requires

```text
active_rows >= 1
normal_reaction >= 1e-12
```

Here `normal_reaction = sum_i p_i A_i`. A single non-load-bearing accepted state
may be classified as a localized transition only when overlap and support remain,
both neighboring accepted states are load bearing, a localized event lies in the
neighboring interval, and that interval is no larger than two requested path
increments. Consecutive non-load-bearing states are sustained contact loss.

### Mesh-quality metrics

For each TET4 element `e`, the monitor records

```text
J_e       = det(F_e)
psi_hat_e = psi_e / mu
```

where `F_e` is the deformation gradient, `psi_e` is strain-energy density, and
`mu` is the material shear modulus. `J_e <= 0` is singular or inverted. The
accepted-state table identifies the global element, body, and body-local element
responsible for the minimum Jacobian and maximum normalized energy density.

The current warning/failure values are:

| Measure | Warning | Failure |
| --- | ---: | ---: |
| minimum `J_e` | `0.50` | `0.05` |
| maximum `psi_e / mu` | `0.50` | `5.0` |

Warnings remain visible accepted states. Failure values enter the benchmark gate.

### Load-step refinement metrics

The quick profile requests 8, 16, and 32 continuation steps. The full profile
requests 32, 64, and 128. For each level, the requested increment is fixed as the
initial and maximum increment; adaptive cutbacks below it remain enabled and are
reported separately.

Medium and fine accepted histories are interpolated to the common uniform grid
defined by the fine requested resolution. The primary response fields are
controlled reactions in x/y/z, maximum pressure, and total overlap area. For each
field, the table contains medium value, fine value, absolute error, and relative
error. Relative errors are scaled by the maximum magnitude of the complete fine
history so zero crossings do not create artificial singular errors.

Topology events are grouped by event kind, entity, and interface. Ordered
occurrences are paired. Count mismatches are explicit failures; matched locations
use absolute continuation-parameter error.

Pressure refinement additionally compares nodal pressure, multiplier, gap, row
area, aggregate resultants and moments, norms, variance, supported area, and
centroid coordinates. Mesh-quality refinement compares minimum-Jacobian and
maximum-normalized-energy histories.

### Checkpoint metrics

The bounded VTK evidence requests six physical regimes:

1. `pre-contact`: reference state at `s = 0`;
2. `first-contact`: earliest accepted state with overlap, support, and active
   pressure;
3. `compressed`: accepted state nearest `s = 0.25`;
4. `quarter-rotation`: rotation-phase state nearest 25 percent of the rotation;
5. `half-rotation`: rotation-phase state nearest 50 percent of the rotation;
6. `final`: accepted state at the completed prescribed motion.

The checkpoint table records the selection rule, target parameter, selected
accepted-step index, selected parameter, selection error, maximum displacement,
maximum pressure, and overlap area. A missing regime is recorded with a typed
reason rather than silently omitted.

Each available checkpoint contains:

- `volume.vtu`: displacement, reaction, external load, contact force, element
  Jacobian, and energy density;
- `slave-contact.vtp`: normal gap, pressure, multiplier, support, activity, and
  slave contact force;
- `master-contact.vtp`: master contact force and overlap area per master facet;
- `projected-overlap.vtp`: slave, master, and clipped polygons in the projection
  plane.

## Plot dictionary

Every SVG is a visualization of values retained in a table or checkpoint; no plot
introduces a separate acceptance quantity.

| Plot | Series and interpretation |
| --- | --- |
| `overlap-area.svg` | Total projected intersection area versus continuation parameter. |
| `maximum-pressure.svg` | Maximum slave nodal projected pressure versus continuation parameter. |
| `controlled-reactions.svg` | Summed controlled-boundary reaction components versus continuation parameter. |
| `deformation.svg` | Maximum nodal displacement at selected physical checkpoints. |
| `pressure-redistribution.svg` | Slave nodal pressure profiles for selected checkpoints. |
| `event-locations.svg` | Localized atomic event categories at their absolute continuation parameters. |
| `final-projected-overlap.svg` | Slave, master, and intersection polygons for the final checkpoint. |
| `force-balance.svg` | Normalized global and interface force errors for accepted states. |
| `moment-balance.svg` | Normalized origin- and pivot-based moment errors for accepted states. |
| `balance-worst-states.svg` | Continuation coordinates of the worst balance metrics. |
| `contact-retention-metrics.svg` | Overlap, support/activity, normal reaction, and gap diagnostics during rotation. |
| `contact-retention-status.svg` | Accepted, localized-transition, or failed retention classification. |
| `mesh-quality.svg` | Minimum Jacobian and maximum normalized energy density over accepted states. |
| `mesh-quality-refinement.svg` | Medium/fine mesh-quality histories and their differences. |
| pressure aggregate and centroid plots | Pressure resultant, norms, supported area, and centroid histories from the pressure tables. |
| pressure refinement plots | Medium/fine nodal and aggregate pressure errors on the common path grid. |

## Acceptance semantics

The common gate evaluates every category before returning. Current quick/full
limits are:

| Criterion | Limit |
| --- | ---: |
| final parameter | `1.0 +/- 1e-12` |
| minimum rotation overlap | `1e-12` |
| minimum supported rows | `1` |
| normalized equilibrium residual | `1e-8` |
| normalized penetration | `1e-7` |
| normalized force error | `1e-7` |
| normalized moment error | `1e-7` |
| maximum medium/fine primary-field error | `5e-2` |
| repeated-scan absolute error | `1e-12` |
| repeated-scan relative error | `1e-10` |

The event-location limit is `0.125` in quick mode and `0.03125` in full mode.
The pressure, retention, and mesh-quality monitors add their own versioned
criteria to the same aggregate gate.

A gate pass means:

- the solver reached the final path coordinate with converged accepted states;
- rotation retained the required frictionless normal-contact evidence;
- accepted equilibrium, penetration, force, and moment measures met their limits;
- repeated kinematic topology scans were deterministic;
- medium/fine response and event histories met the configured comparison limits;
- pressure, retention, mesh-quality, checkpoint, and artifact evidence was
  complete.

Thresholds are benchmark policy. They are not universal discretization-error or
contact-formulation bounds.

## Expected transitions and diagnostic failure modes

The rectangular upper footprint rotates over a nonmatching master grid and then
translates in x. The expected history therefore contains repeated facet-pair
entries/exits, support changes, pressure-row branch changes, clipping transitions,
and pallet transitions while positive projected overlap remains.

Interpret common failures as follows:

- **No first-contact checkpoint:** compression did not produce an accepted
  pressure-bearing state or the accepted-state evidence is incomplete.
- **Overlap or support loss:** the geometric contact branch was lost; a
  non-load-bearing exception cannot repair missing geometry.
- **Activity or reaction loss with overlap retained:** contact may be crossing a
  localized branch, but only the bounded one-state exception is admissible.
- **Event-count mismatch:** medium and fine paths crossed different discrete
  histories or event attribution is unstable.
- **Event-location mismatch:** the same event was found at insufficiently
  converged path coordinates.
- **Force or moment imbalance:** reaction extraction, contact sign, node mapping,
  or accepted equilibrium may be inconsistent.
- **Pressure-resultant mismatch:** nodal pressure integration and assembled slave
  contact force disagree.
- **Near-zero or negative Jacobian:** the bulk state is singular or inverted;
  contact diagnostics from that state are not trustworthy.
- **Dense materialization in the full profile:** sparse execution silently fell
  back to a dense global matrix and violates the backend policy.
- **Excessive retries without progress:** the solver policy is cycling even when
  individual inner solves produce typed failures.

## Claim boundary and follow-up benchmarks

The benchmark supports claims only for the implemented frictionless normal-contact
formulation, compressible neo-Hookean `TET4` bulk model, prescribed rigid upper
boundary, current nonmatching QUAD4 interfaces, and implemented one-sided
event-localized restart policy.

It does not establish:

- Coulomb friction, stick/slip history, tangential traction, or frictional work;
- a unique generalized derivative at exact edge-on-edge or on-vertex states;
- curved-interface objectivity or pressure invariance under a full revolution;
- plasticity, remeshing, near-incompressible anti-locking, or higher-order bulk
  behavior;
- mesh convergence of the physical discretization, because the current primary
  refinement campaign refines continuation steps rather than both bulk meshes;
- machine-independent performance, because timings are provenance only.

Issue #65 isolates exact and perturbed edge-on-edge and on-vertex states before
stronger generalized-derivative claims are made. Issue #25 extends the verified
frictionless path to cylindrical ironing with severe local deformation and long
sliding distance. Issue #26 tests curved concentric interfaces and rigid-rotation
objectivity. Issue #71 registers the physically validated quick/full campaign in
the standardized benchmark suite.

## Commands and detailed notes

Run the complete bundle with:

```bash
uv run python benchmarks/rotating_blocks_bundle.py \
  --profile quick \
  --output results/rotating-blocks
```

Use `--profile full` for the publication-oriented mesh and path settings.

Detailed implementation notes remain authoritative for their individual data
contracts:

- [model](rotating-blocks-model.md);
- [solver integration](rotating-blocks-solver.md);
- [solver diagnostics](rotating-blocks-solver-diagnostics.md);
- [topology oracle](rotating-blocks-topology-oracle.md);
- [load-step refinement](rotating-blocks-refinement.md);
- [result bundle](rotating-blocks-result-bundle.md);
- [pressure redistribution](rotating-blocks-pressure.md);
- [force and moment balance](rotating-blocks-balance.md);
- [contact retention](rotating-blocks-contact-retention.md);
- [mesh quality](rotating-blocks-mesh-quality.md);
- [physical checkpoints](rotating-blocks-checkpoints.md);
- [acceptance gate](rotating-blocks-acceptance-gate.md).
