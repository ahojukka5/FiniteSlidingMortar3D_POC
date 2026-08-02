# Multiplier transport across support changes

The nodal augmented-Lagrange multiplier vector has the fixed dimension of the
slave surface, while projected mortar support changes as overlap topology moves.
An event-localized Newton restart must therefore transport the accepted multiplier
state onto the selected post-event support branch before evaluating the restarted
residual and tangent.

## Deterministic policy

For each interface, compare the supported-row masks on the localized left branch
and selected branch:

- rows supported on both branches retain their accepted multiplier;
- released rows are set to zero before the post-event restart;
- newly supported rows are initialized to zero;
- rows unsupported on both branches remain zero.

Zero initialization is deliberately history-free. A newly supported row receives
pressure only through the selected-branch gap and the next projected augmentation,
not through a multiplier retained from an earlier, disconnected support interval.

## KKT boundary

Before transport, `maximum_unsupported_before` records the largest multiplier that
would violate the selected branch's unsupported-row condition. After transport,
`maximum_unsupported_after` must be exactly zero. The ordinary contact evaluation
then recomputes penetration, complementarity, projection, and unsupported-row KKT
residuals with the transported state.

Transport does not increment the augmentation index. It changes only the support
representation of the already accepted multiplier state. A subsequent outer
augmentation retains its normal increment semantics.

## Solver propagation

`solve_event_aware_coupled_equilibrium` applies transport immediately after a
support-changing event is localized and before the selected displacement is
re-evaluated. The resulting states are returned by
`EventAwareCoupledNewtonResult`.

Both dimensional and scale-aware augmented solvers continue from those returned
states. The adaptive path already consumes the augmented result, so accepted,
cut-back, and penalty-retried attempts share the same transport policy.

## Diagnostics

Each changed interface produces a `MultiplierTransportRecord` containing:

- released, activated, and all changed row indices;
- multiplier values before and after transport on changed rows;
- maximum unsupported multiplier before and after transport;
- the versioned initialization rule name, currently `zero`.

`multiplier_transport_rows()` returns strict-JSON-compatible dictionaries for
benchmark event tables. The records remain separate from geometric topology-event
records because one localized event batch can update more than one interface.
