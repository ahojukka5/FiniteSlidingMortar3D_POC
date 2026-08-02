# Rotating-blocks kinematic topology oracle

The rotating-blocks nonlinear benchmark needs a geometry-level reference that is
independent of Newton convergence. The kinematic topology oracle evaluates the
same staged compression and rotation path while holding every free bulk degree of
freedom at zero and applying the prescribed rigid-body constraints exactly.

No equilibrium solve, line search, multiplier augmentation, penalty update, or
nonlinear stopping tolerance is used. The only numerical threshold is the
explicit contact-geometry tolerance used by clipping, overlap integration, and
mortar-row support detection.

## Scan state

For every absolute continuation parameter, the scanner:

1. evaluates the immutable staged boundary path;
2. applies its complete Dirichlet snapshot to a zero displacement vector;
3. creates zero multiplier states for the contact interfaces;
4. evaluates each mortar interface directly;
5. constructs the same `ContactTopologySignature` used by event-localized Newton.

Each interface sample records:

- integrated facet-pair set;
- projected total overlap area;
- supported mortar-row indices;
- active pressure-row indices;
- maximum projected pressure;
- geometry tokens `(slave facet, master facet, polygon vertices, pallets,
  orientation)`.

`contact_topology_signatures` is shared with the event-localization machinery, so
the oracle and nonlinear solver cannot silently adopt different definitions of a
contact branch.

## Transition brackets

Adjacent sampled signatures are compared with the production topology-difference
logic. Every changed interval stores its absolute left and right continuation
parameters and the atomic changes inside the interval:

- pair entry or exit;
- support activation or release;
- pressure activation or release;
- clipping vertex/edge transition;
- centroid-fan pallet transition.

These are deterministic brackets, not claims of an exact event location. The
nonlinear benchmark can use `expected-transitions.json` to diagnose missed,
additional, or displaced events and can refine selected brackets with the event
state machine.

The discrete sample history is hashed into a SHA-256 signature digest. Repeated
runs with the same profile, sample parameters, and geometry tolerance must produce
the same digest and transition table.

## Artifacts

Run the canonical quick oracle with:

```bash
uv run python benchmarks/rotating_blocks_topology_oracle.py \
  --profile quick \
  --output results/rotating-blocks-topology-oracle
```

The benchmark writes:

- `summary.json` — event counts, pair-count range, overlap range, and digest;
- `expected-transitions.json` — reusable absolute transition brackets;
- `sample-history.csv` — one row per parameter and interface;
- `transition-history.csv` — one row per changed interval;
- `overlap-area.svg` — projected overlap history;
- `topology-counts.svg` — facet-pair, support-row, and active-row histories;
- `transition-timeline.svg` — atomic event categories along the path;
- `manifest.json` — provenance and artifact registration.

The quick profile uses 65 samples by default. The full profile uses 129 samples
with the same physical geometry and final motion.

## Verification boundary

The oracle proves that the prescribed geometry crosses repeated deterministic
contact-topology changes. It does not prove nonlinear equilibrium, force balance,
KKT convergence, multiplier transport, or load-step convergence. Those claims
belong to the subsequent event-localized rotating-blocks solve and refinement
study.
