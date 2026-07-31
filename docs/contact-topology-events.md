# Contact-topology event localization

The analytical mortar tangent is valid while the discrete contact branch is fixed. A branch is not
only the unilateral active set. It also contains the integrated facet pairs, supported slave rows,
and the topology of every projected intersection polygon and centroid-fan pallet decomposition.

This note documents the first explicit event-state-machine layer. It replaces the previous Boolean
"branch changed" restart with typed, localized, reproducible event records.

## Branch signature

For every contact interface the event solver records

- integrated slave/master facet pairs;
- supported slave rows;
- active pressure rows;
- one geometry token per integrated pair:
  `(slave facet, master facet, intersection vertex count, pallet count, orientation sign)`.

The geometry token is intentionally discrete. Continuous overlap area, pressure, and gap remain
smooth state variables and are not used to decide whether a derivative branch changed.

## Atomic events

A transition is decomposed deterministically into atomic records:

- `pair_entry` and `pair_exit`;
- `clipping_vertex_edge` when polygon vertex count or orientation changes;
- `pallet_transition` when centroid-fan connectivity changes;
- `support_activation` and `support_release`;
- `pressure_activation` and `pressure_release`;
- `inverse_map_boundary` when the projected inverse map reports a recoverable singular event.

Clipping, pallet, and inverse-map exceptions are recoverable contact events. A nonpositive bulk
deformation Jacobian remains invalid geometry and is handled by the ordinary line search rather than
being relabelled as contact topology.

## Segment localization

Let `s` parameterize one Newton trial segment,

```text
u(s) = u_n + s Delta u,  0 <= s <= alpha.
```

The state machine receives valid left and right observations on different branches. It bisects the
segment using the predicate "same branch as the left observation" until the bracket width is below
`fraction_tolerance`. If the evaluator enters a recoverable singular band, the machine first
localizes its left boundary and then its first valid state on the other side.

The selected derivative branch is explicit:

- `right` is the production default and restarts Newton immediately after the event;
- `left` is available for verification and one-sided derivative studies.

A localized batch stores the final left/event/right fractions, selected fraction, selected branch,
and every atomic event in stable interface/kind/entity order.

## Newton integration

`solve_event_aware_coupled_equilibrium` retains the established smooth residual and tangent. When a
residual-only line-search trial is on another branch, it

1. localizes the first event on the accepted trial segment;
2. moves to the first valid point on the selected side;
3. records the typed event batch;
4. reassembles the analytical tangent on that new branch;
5. restarts Newton from the event state.

The structural event step is accepted independently of the old-branch Armijo model. Subsequent
smooth steps use the unchanged Armijo residual merit test. `event_policy="reject"` retains the old
behavior for comparison.

`solve_event_aware_augmented_contact` uses the same inner solver and retains event histories across
all multiplier augmentations.

## Determinism regression

The committed synthetic regression crosses pair, clipping, pallet, support, and pressure events with
5, 10, 20, and 40 subdivisions. Event locations are compared against the finest partition. The
state machine must select the right branch and reproduce the same atomic transition ordering.

Regenerate with

```bash
uv run python benchmarks/topology_event_regression.py \
  --output results/topology-events
```

## Current boundary

This slice localizes events inside Newton trial segments and derives geometry topology from the
production projected overlap. It does not yet replace the adaptive continuation controller's outer
cutback policy. The next slice must propagate event-aware augmented solves through adaptive mixed
paths and record absolute continuation parameters for every event.
