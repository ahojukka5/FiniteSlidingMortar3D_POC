# Continuation restart diagnostics

Long finite-sliding paths can cross many legitimate contact-topology events. A
large restart count is therefore not itself a failure. The diagnostic layer in
`contact3d.event_solver` separates four mechanisms:

1. **Localized topology restarts** are exact post-event Newton restarts retained
   in `AdaptiveTopologyEventBatch` records.
2. **Line-search cutbacks** are reductions inside a fixed Newton solve. Their
   count is the sum of `line_search_iterations` over every attempted solve.
3. **Adaptive cutbacks** reject a continuation target and reduce the requested
   path increment.
4. **Penalty retries** repeat the same continuation target after increasing one
   or more interface penalties.

`solve_event_aware_adaptive_contact_path` retains every attempted solver result,
including rejected and penalty-retried attempts. This permits line-search
statistics to remain aligned with the adaptive attempt table instead of being
available only for accepted states.

## Progress-aware loop rule

`analyze_restart_diagnostics` converts every atomic topology event into a record
containing:

- adaptive attempt and action;
- number of committed accepted steps;
- absolute continuation parameter;
- augmentation, event batch, interface, kind, and entity;
- selected left/right branch;
- the complete selected facet-pair, clipping, pallet, support, and activity
  signature.

A restart loop is reported only when the same event identity and selected
signature are localized repeatedly:

- at the same committed-step number; and
- within `parameter_tolerance` of the same absolute continuation parameter.

Accepted continuation progress resets the repetition sequence. The default
`repetition_limit` is three localizations. Consequently, the same pair transition
may occur many times along a rotating path without being classified as a loop,
provided the accepted path parameter advances.

The diagnostic termination value is `restart_loop`. It is evidence for a
benchmark acceptance gate; the diagnostic does not retroactively alter the
solver result or reject otherwise healthy event sequences.

## Machine-readable outputs

`RestartDiagnostics.summary()` provides aggregate counts and the first detected
loop. `count_rows()` groups atomic events by committed step, augmentation,
interface, and event kind. `attempt_rows()` aligns event batches, atomic events,
line-search cutbacks, adaptive cutbacks, and penalty retries with each attempted
solve.

The rotating-blocks result writer should include these outputs as separate JSON
and CSV artifacts. Timing information and future stable signature hashes are
outside this diagnostic. The selected signature is retained as deterministic
canonical JSON until the dedicated topology-signature work adds versioned hashes.
