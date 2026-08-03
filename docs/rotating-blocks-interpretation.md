# Rotating-blocks benchmark interpretation

This note defines what the rotating-blocks benchmark measures, how its quick and
full outputs must be interpreted, and which conclusions the evidence does not
support. The benchmark is a controlled precursor to cylindrical ironing: it
combines a deformable support, prescribed rigid boundary motion, nonmatching
mortar surfaces, changing projected overlap topology, augmented frictionless
contact, and event-localized adaptive continuation.

A successful result means that this specific model and solver policy completed
the documented path and satisfied the applicable versioned numerical criteria. It
is not a general proof of frictional contact, curved-surface objectivity,
generalized derivatives at exact special states, or production-scale performance.

## Physical model

### Geometry and meshes

The deformable lower body occupies

```text
x in [-1.00, 1.00]
y in [-1.00, 1.00]
z in [ 0.00, 0.50]
```

The prescribed upper body occupies

```text
x in [-0.62, 0.68]
y in [-0.32, 0.38]
z in [ 0.521, 0.821]
```

The initial normal separation is `0.021`. The rotation pivot is
`(0.03, 0.03, 0.521)`, the center of the upper contact plane. Translating both the
upper body and pivot by `0.03` in x and y preserves the centered physical motion
while keeping slave and master grid lines distinct at the initial and final
orientations. Exact edge-on-edge and on-vertex states are isolated separately by
issue #65 rather than imposed at the first production Newton iteration.

The one-millimetre normal offset places rigid first contact at `s = 0.13125`,
strictly between both the quick and full nominal continuation nodes. The event
solver therefore localizes contact onset inside an increment instead of landing
exactly on the unilateral active-set switch.

Both bodies use positively oriented total-Lagrangian `TET4` elements. Every
structured hexahedral cell is split into six tetrahedra.

| Profile | Lower cells | Upper cells | Lower TET4 | Upper TET4 |
|---|---:|---:|---:|---:|
| `quick` | `2 x 2 x 1` | `3 x 2 x 1` | 24 | 36 |
| `full` | `8 x 8 x 4` | `3 x 2 x 1` | 1536 | 36 |

The upper bottom surface is the non-mortar slave side. Its six QUAD4 facets use
12 nodes. The lower top surface is the master side, with four facets and nine
nodes in quick mode or 64 facets and 81 nodes in full mode. The interfaces are
therefore nonmatching in both profiles.

The rectangular upper footprint rotates over the master grid and translates
`0.10` in x. This causes repeated clipping, pallet, support, pressure, and—when
the broad-phase set changes—facet-pair transitions while the bodies remain in
contact.

### Material and contact

The bulk material is the compressible logarithmic neo-Hookean model constructed
from

```text
Young modulus E = 210
Poisson ratio nu = 0.30
```

The complete TET4 mesh receives the same material. Every upper node is prescribed,
so the upper block contributes geometry, reactions, and visualization but no free
deformable mode.

The contact model is biased single-pass standard mortar with frictionless normal
contact:

```text
initial normal penalty = 3200
search distance        = 0.12
triangle quadrature    = 7 points
```

The production solver may increase the normal penalty through its bounded,
interface-local, scale-aware policy. The search distance controls candidate
discovery; it is not an admissible penetration or gap tolerance.

### Boundary conditions and path

All lower-bottom nodes are fixed in all three components. Every upper node is
controlled in all three components. No dead nodal load is applied; loading is
introduced through prescribed motion and reactions are recovered on constrained
degrees of freedom.

Let `s` be the absolute continuation parameter:

1. `0 <= s <= 0.25`: translate the upper body by `(0, 0, -0.04)`;
2. `0.25 <= s <= 1`: hold the compression, rotate through `pi/2` about the
   global z axis through the pivot, and add `(0.10, 0, 0)` translation.

The compression consumes the `0.021` separation and imposes another `0.019` of
approach relative to the undeformed lower reference surface. The lower-body
deformation and pressure distribution follow from nonlinear equilibrium.

## Evidence profiles

### Standardized quick campaign

`benchmarks/rotating_blocks_quick.py` performs one production quick solve. It
writes:

- accepted-state, adaptive-attempt, solver-diagnostic, and topology-event tables;
- force and moment balance and contact-retention evidence;
- response, overlap, pressure, event, balance, and retention plots;
- one complete final-state VTK checkpoint;
- a manifest, a quick acceptance gate, and checked non-timing golden metrics.

The quick campaign deliberately omits repeated topology scans and the
coarse/medium/fine continuation-refinement campaign. It is a bounded physical CI
exercise, not the publication discretization and not a synthetic solver stub.

### Full publication campaign

`benchmarks/rotating_blocks_bundle.py --profile full` executes the full mesh and
path settings and adds:

- repeated clean topology scans;
- 32/64/128-step continuation-refinement comparisons;
- pressure-redistribution and mesh-quality refinement evidence;
- all six requested physical VTK checkpoints;
- the complete aggregate acceptance gate and publication result bundle.

Machine-dependent timings are provenance only in both profiles.

## Interpretation layers

Read the evidence in four separate layers:

1. **Geometry:** dimensions, mappings, projected polygons, facet pairs, support
   rows, and topology signatures.
2. **Formulation:** finite-strain bulk residuals, frictionless mortar residuals,
   augmented multipliers, pressure, KKT measures, reactions, and balance.
3. **Solver policy:** Newton, line search, augmentation, event localization,
   adaptive cutback, penalty retry, and linear backend selection.
4. **Benchmark policy:** quick/full meshes, requested path resolution, checkpoint
   selection, tolerances, repetition, and refinement comparisons.

A topology event is a geometry or active-set change. A cutback is a continuation
policy action. A penalty retry is an enforcement-policy action. They are recorded
separately.

## Core metrics

### Accepted states

`tables/accepted-steps.csv` contains one row for each accepted nonlinear state.
Important fields are:

| Metric | Meaning |
|---|---|
| `parameter` | Absolute continuation coordinate `s`. |
| `phase_index` | Zero for compression, one for rotation/translation. |
| `rotation_angle` | Prescribed upper-body angle. |
| `reaction_x/y/z` | Sum of controlled upper-boundary reactions. |
| `maximum_pressure` | Largest accepted slave nodal pressure. |
| `overlap_area` | Total integrated projected intersection area. |
| `facet_pairs` | Integrated slave/master facet-pair count. |
| `supported_rows` | Slave rows with positive geometric integration support. |
| `active_rows` | Supported rows on the active pressure branch. |
| `inner_converged` | Inner equilibrium convergence status. |

Reaction signs follow the global residual convention; they must not be replaced
by magnitudes during comparisons.

### Attempts and solver work

`tables/attempts.csv` retains accepted attempts, adaptive cutbacks, and penalty
retries. Rejected states remain diagnostics and never become accepted KKT
evidence. `tables/solver-diagnostics.csv` records deterministic counts for
augmentations, Newton and line-search iterations, linear solves and failures,
event localization, selected backends, matrix nonzeros, and dense
materializations. Wall time and linear setup/solve time are excluded from numeric
acceptance.

### Topology

Projected overlap is formed by projecting each proximate facet pair to the slave
center plane, clipping the two polygons, triangulating the intersection, and
integrating both sides at common physical quadrature points.

A topology signature contains ordered facet pairs, clipping and pallet tokens,
supported rows, and active rows. Event locations are absolute continuation
coordinates returned by the event-localization state machine, not raw Newton
fractions or topology-scan bracket midpoints.

The production gate requires at least two distinct localized transitions. Records
are deduplicated by event kind, interface, entity, and continuation coordinate, so
repeated augmentation records of one event do not count as separate transitions.
Pair-entry and pair-exit counts remain diagnostics: a coarse interface may retain
the same broad-phase candidates while clipping, pallet, support, or pressure
branches change.

The kinematic topology oracle holds free bulk degrees of freedom at zero. It
proves that the prescribed path crosses deterministic geometry branches; it does
not prove nonlinear equilibrium. Production events are equilibrium dependent and
are reported separately.

### Pressure

For slave row `i`, with mortar area `A_i` and pressure `p_i`, the aggregate
pressure measures include

```text
resultant = sum_i p_i A_i
mean      = sum_i p_i A_i / sum_i A_i
variance  = sum_i A_i (p_i - mean)^2 / sum_i A_i
rms       = sqrt(sum_i A_i p_i^2 / sum_i A_i)
L2        = sqrt(sum_i A_i p_i^2)
centroid  = sum_i p_i A_i x_i / resultant
```

Unsupported rows must have zero pressure and zero accepted multiplier. The
pressure resultant is compared with the normal projection of the assembled slave
contact force. The implementation uses a penetration-positive normal-gap
convention.

### Force and moment balance

The applied resultant is zero for this prescribed-motion benchmark. The reaction
resultant is the constrained residual. The interface residual is split into slave
and master blocks, which must cancel. Moments use current nodal coordinates and
are evaluated about the global origin and the translated pivot. Errors are
normalized by force and moment total-variation scales rather than raw magnitudes.

### Scale-aware KKT limits

The production solver uses:

| Measure | Normalized limit |
|---|---:|
| free equilibrium residual | `1e-8` |
| maximum penetration | `1e-7` |
| complementarity | `1e-7` |
| multiplier admissibility | `1e-7` |
| projection residual | `1e-5` |

The projection residual is pressure-valued and includes the penalty-to-pressure
scale ratio multiplying normalized gap. Its `1e-5` limit therefore does not relax
the independent `1e-7` penetration requirement; it prevents that requirement
from being multiplied again by the penalty ratio at contact onset.

### Contact retention

During accepted rotation states, structural contact requires positive overlap and
at least one supported row. Load-bearing contact requires at least one active row
and positive normal reaction. A single non-load-bearing state may be classified as
a localized transition only when load-bearing neighbors and a localized event
bracket it within the documented path interval. Consecutive non-load-bearing
states are sustained contact loss.

### Mesh quality and refinement

For every TET4 element, the full monitor records

```text
J_e       = det(F_e)
psi_hat_e = psi_e / mu
```

Nonpositive `J_e` is singular or inverted. The full campaign compares minimum
Jacobian and maximum normalized energy histories along with reactions, pressure,
overlap, and event locations on common path coordinates.

## Checkpoints

The full campaign requests:

1. `pre-contact`;
2. `first-contact`;
3. `compressed` near `s = 0.25`;
4. `quarter-rotation`;
5. `half-rotation`;
6. `final` at `s = 1`.

Every available checkpoint contains volume, slave-contact, master-contact, and
projected-overlap VTK files. Missing regimes are represented by typed selection
records rather than silently omitted. The standardized quick campaign exports the
same four-file contract for the final state only while retaining selection records
for all six regimes.

## Acceptance meaning

A quick gate pass means that the bounded production model reached the requested
final motion, accepted converged states, met equilibrium and penetration limits,
localized repeated distinct topology transitions, retained contact, passed
force/moment balance, identified the requested physical regimes, and wrote a valid
final checkpoint and manifest.

A full gate pass additionally means deterministic repeated scans, acceptable
continuation-refinement differences, complete pressure and mesh-quality evidence,
and all six checkpoint exports. Thresholds are benchmark policy, not universal
contact-formulation error bounds.

## Failure interpretation

- **No first-contact regime:** compression did not create an accepted
  pressure-bearing state or its evidence is incomplete.
- **Overlap or support loss:** the geometric contact branch was lost.
- **Activity or reaction loss with overlap retained:** contact may be crossing a
  localized branch; only the bounded one-state exception is admissible.
- **Maximum augmentations:** inspect projection, complementarity, gap, multiplier,
  and event history separately before changing penalties or tolerances.
- **Event mismatch:** paths crossed different discrete histories or event
  attribution is unstable.
- **Force or moment imbalance:** reaction extraction, contact sign, node mapping,
  or accepted equilibrium may be inconsistent.
- **Pressure-resultant mismatch:** nodal pressure integration and assembled slave
  contact force disagree.
- **Nonpositive Jacobian:** the bulk state is invalid and its contact diagnostics
  are not trustworthy.
- **Dense full-profile matrix:** sparse execution silently fell back to a dense
  global matrix.
- **Retries without progress:** solver policy is cycling despite typed inner
  failures.

## Claim boundary

The benchmark supports claims only for frictionless normal contact, biased
single-pass standard mortar, compressible neo-Hookean `TET4` bulk mechanics,
nonmatching QUAD4 interfaces, a prescribed rigid upper boundary, and the current
one-sided event-localized restart policy.

It does not establish Coulomb friction, unique generalized derivatives at exact
special states, curved-interface objectivity, plasticity, near-incompressible
anti-locking, remeshing, higher-order elements, or machine-independent
performance. Issue #25 extends the path to cylindrical ironing; #26 covers curved
concentric interfaces; #65 isolates exact special states.

## Commands

Run the bounded integration evidence with:

```bash
uv run python benchmarks/rotating_blocks_quick.py \
  --output results/rotating-blocks-quick
```

Run the publication campaign with:

```bash
uv run python benchmarks/rotating_blocks_bundle.py \
  --profile full \
  --output results/rotating-blocks
```

Detailed contracts remain in the model, solver, diagnostics, topology-oracle,
refinement, result-bundle, pressure, balance, retention, mesh-quality, checkpoint,
and acceptance-gate notes.
