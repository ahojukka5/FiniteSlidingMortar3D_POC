# Rotating-blocks deterministic repetition check

The repetition check runs the same rotating-blocks profile twice from independently
constructed model and path objects. It compares complete topology histories rather
than only final scalar metrics.

Discrete fields are compared exactly:

- phase names;
- ordered facet-pair signatures;
- clipping and pallet geometry tokens;
- supported and active row states;
- transition kinds, interfaces, entities, and details;
- frame and transition counts.

Continuous fields use explicit absolute and relative tolerances:

- absolute and phase continuation parameters;
- transition bracket endpoints;
- projected overlap area;
- maximum pressure.

The first mismatch is reported with its field name, frame or transition index,
absolute path parameter, interface, left and right values, and numeric errors when
applicable. Later mismatches are intentionally not hidden behind an aggregate pass
or fail flag.

The command is independent of mesh refinement and nonlinear equilibrium:

```bash
uv run python benchmarks/rotating_blocks_repetition.py \
  --profile quick \
  --output results/rotating-blocks-repetition
```

It constructs two clean kinematic scans and writes a versioned `summary.json` plus
an artifact manifest. The current check validates the solver-independent topology
oracle. The same comparison model can later be applied to accepted nonlinear
histories after the production rotating-blocks solve is established.
