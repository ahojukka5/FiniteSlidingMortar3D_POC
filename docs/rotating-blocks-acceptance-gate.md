# Rotating-blocks acceptance gate

The rotating-blocks command writes all numerical and visualization evidence before
it evaluates the final benchmark result. It then exits with a nonzero status when
any required criterion fails. The failure message includes every failed metric,
its observed value, the comparison relation, and the configured limit.

The gate schema is `contact3d-rotating-blocks-acceptance-gate/v1`. The complete
criterion table is written to `tables/acceptance-gate.csv`, while the versioned
thresholds, deterministic repetition summary, failed-criterion names, and messages
are written to `acceptance-gate.json`. Both files are required by the main artifact
manifest and are embedded in `summary.json`.

## Evidence categories

The gate combines evidence already produced by the production benchmark:

- solver convergence and completion of the prescribed path to parameter `1.0`;
- retained projected overlap and at least one supported mortar row throughout all
  accepted rotation-phase states;
- scale-aware equilibrium and penetration residuals;
- global and interface force and moment balance;
- exact discrete and tolerance-based continuous comparison of two clean kinematic
  topology scans;
- medium/fine response and topology-event refinement agreement;
- complete pressure-redistribution evidence.

All criteria are evaluated before the result is returned. A failure in one category
does not suppress diagnostics from the remaining categories.

The current contact-retention criterion is deliberately a baseline gate: every
accepted rotation state must retain positive projected overlap and supported rows.
The dedicated contact-retention monitor extends this evidence with interval-aware
activity, reaction, gap, and neighboring-signature diagnostics.

## Versioned thresholds

Quick and full profiles select immutable threshold records. Both profiles currently
use:

| Metric | Limit |
| --- | ---: |
| final continuation parameter | `1.0 ± 1e-12` |
| minimum rotation overlap area | `1e-12` |
| minimum supported rows | `1` |
| normalized equilibrium residual | `1e-8` |
| normalized penetration | `1e-7` |
| normalized force error | `1e-7` |
| normalized moment error | `1e-7` |
| maximum medium/fine field error | `5e-2` |
| repetition absolute error | `1e-12` |
| repetition relative error | `1e-10` |

The topology-event location limit follows the existing path-refinement resolution:
`0.125` in quick mode and `0.03125` in full mode. Full mode is therefore stricter
for event localization while retaining the same physical and KKT limits.

Any threshold change must update the schema or its documented versioned policy,
focused tests, and the profile record in the same pull request. Machine-dependent
timings remain provenance only and are never gate criteria.

## Command behavior

The complete benchmark is run with:

```bash
uv run python benchmarks/rotating_blocks_bundle.py \
  --profile quick \
  --output results/rotating-blocks
```

Artifacts and the manifest are finalized even when the numerical gate fails. The
process then raises one aggregated error containing all failed criterion messages,
which makes CI failures actionable without rerunning the benchmark once per metric.
