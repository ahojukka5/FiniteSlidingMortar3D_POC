# Rotating-blocks solver diagnostics

The rotating-blocks production runner records one diagnostics row for every
adaptive attempt, including accepted steps, cutbacks, and penalty retries. The
schema is `contact3d-rotating-blocks-diagnostics/v1`.

## Deterministic work counters

Each row records the requested path interval, action, inner termination reason,
augmentation and Newton counts, line-search iterations, linear solves and
iterations, linear failures, and localized topology-event batches and atomic
events. The row also records selected linear backends, preconditioners, maximum
matrix nonzero count, and dense materializations.

The aggregate summary totals those counters and identifies the worst accepted
attempt and worst rejected or retried attempt. Worst means the lexicographic
maximum of Newton iterations, linear iterations, localized event batches, and
attempt index. This ordering is deterministic and is suitable for regression
comparisons.

## Backend policy

The quick profile requests the `auto` linear backend. The full profile requests
`sparse_lu`, and its diagnostics criterion requires zero dense matrix
materializations. A sparse run therefore cannot silently satisfy the benchmark
through dense fallback.

## Timing boundary

Wall-clock time and linear setup and solve times are retained as provenance.
The remaining wall time is reported as unattributed time, which includes Python
overhead, bulk and contact assembly, topology localization, and other solver
work not timed independently by the current core APIs.

Timing values are deliberately excluded from acceptance. Numerical acceptance,
backend selection, work counts, and worst-attempt identities do not depend on
machine-dependent durations.
