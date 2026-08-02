# Rotating-blocks load-step refinement

The refinement study reruns one rotating-blocks mesh and physical motion with
three requested continuation resolutions. For the quick profile these are 8,
16, and 32 steps; for the full profile they are 32, 64, and 128 steps.

Each requested resolution fixes both the initial and maximum continuation step
to `1 / requested_steps`. Adaptive cutbacks remain enabled below that value.
The result summary records requested resolution and actual cutback count
separately, so solver recovery cannot be mistaken for planned path refinement.

## Common comparison path

Accepted states from the medium and fine runs are interpolated to the uniform
parameter grid defined by the fine requested resolution. The following fields
are compared:

- controlled-body reaction components in global x, y, and z;
- maximum slave nodal pressure;
- total projected overlap area.

The table records medium and fine values plus absolute and relative errors at
every common parameter. Relative errors use the fine value as reference with a
small denominator floor for zero crossings.

Topology events are grouped by event kind, entity, and interface. Ordered
occurrences are paired between the medium and fine runs. The comparison records
event-count mismatches explicitly and measures absolute continuation-parameter
error for matched occurrences.

## Acceptance boundary

The current gate requires:

- every coarse, medium, and fine production run to pass;
- every run to reach the final prescribed motion;
- the final facet-pair, support-row, and active-row counts to agree;
- all medium/fine relative field errors to remain at or below 5 percent;
- medium and fine event counts to match;
- matched event locations to differ by no more than two medium requested steps.

These are benchmark-level tolerances, not generalized contact-formulation
claims. Later result-bundle work can version stricter field-specific limits.

## Command

```bash
uv run python benchmarks/rotating_blocks_refinement.py \
  --profile full \
  --output results/rotating-blocks-refinement
```

The command writes `summary.json`, `field-comparison.csv`,
`event-comparison.csv`, and `refinement-error.svg`.
