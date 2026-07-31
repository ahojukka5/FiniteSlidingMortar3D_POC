# Benchmark artifact contract

The validation campaign uses a versioned artifact contract instead of letting each
benchmark invent unrelated JSON, CSV, and visualization conventions. The first contract is
identified by:

```text
contact3d-benchmark/v1
```

The implementation lives in `contact3d.benchmark_artifacts`. It is intentionally independent
of the mechanics formulation: benchmark code supplies arrays and records, while the artifact
layer validates, serializes, and registers them.

## Manifest

Every migrated benchmark writes `manifest.json` after all other files have been completed.
The manifest contains:

- `schema_version`: the artifact contract version;
- `benchmark`: a stable benchmark name;
- `provenance`: runtime, package, solver, and seed metadata;
- `artifacts`: a deterministic path-sorted list of registered files.

Each artifact record contains a relative `path`, a `kind`, and optionally a more specific
record schema. Supported kinds are `json`, `csv`, `vtu`, `vtp`, `svg`, and `other`.
Absolute paths and parent-directory traversal are rejected. Duplicate paths are rejected, and
`validate_benchmark_manifest(..., root=output)` verifies that every registered file exists.

The manifest is written only by `BenchmarkArtifactWriter.finalize`. Required artifact names
can be supplied at finalization so a benchmark fails rather than silently producing an
incomplete result directory.

## Provenance

`BenchmarkArtifactWriter` records:

- the benchmark name and deterministic seed;
- the current Git SHA, or `GITHUB_SHA` when running from a packaged environment;
- Python version, implementation, and executable;
- operating-system name, release, and machine architecture;
- installed `contact3d`, NumPy, and SciPy versions;
- the complete solver settings after dataclass and NumPy values are normalized to JSON.

A missing Git checkout or package metadata is represented explicitly as `unknown`; it is not
silently omitted.

## Record schemas

Specific JSON and CSV files carry independent schema names in their artifact records. The
current core suite uses:

- `contact3d-tet4-patch/v1` for the affine bulk patch summary;
- `contact3d-tangent-convergence/v1` for directional tangent studies;
- `contact3d-nonlinear-equilibrium/v1` for the nonlinear bulk summary;
- `contact3d-newton-iterations/v1` for bulk Newton and linear-solver rows;
- `contact3d-load-steps/v1` for continuation rows;
- `contact3d-coupled-mortar-patch/v1` for the matching coupled patch summary;
- `contact3d-augmentation-iterations/v1` for augmented-Lagrange outer iterations;
- `contact3d-coupled-newton-iterations/v1` for coupled Newton rows with flattened linear
  diagnostics;
- `contact3d-adaptive-policy/v1` for the adaptive-controller summary;
- `contact3d-adaptive-attempts/v1` for accepted, cut-back, and penalty-escalated attempts;
- `contact3d-adaptive-accepted-steps/v1` for the committed controller path;
- `contact3d-warped-adapter/v1` for the production contact summary;
- `contact3d-contact-nodes/v1` for slave-node gap, pressure, multiplier, support, and
  activity.

Later benchmark families can add compatible record schemas without changing the directory
manifest version. The manifest version changes only when the outer contract itself changes.

## VTK output

The artifact layer writes dependency-free ASCII XML VTK files.

### TET4 volumes

`write_tet4_vtu` writes an `UnstructuredGrid` with VTK tetrahedron cell type 10. The `Points`
array contains current coordinates. Point data always includes:

- `reference_coordinates`;
- `displacement`.

The affine patch writes `affine-patch.vtu` with nodal internal force and element Jacobian,
energy-density, and deformation-gradient-error fields.

The nonlinear bulk benchmark writes `deformed.vtu` with:

- nodal reaction and external load vectors;
- element Jacobian determinants;
- element strain-energy density.

The coupled mortar patch writes `deformed.vtu` with:

- nodal reactions, external loads, and assembled contact-force vectors;
- element body identifiers, Jacobian determinants, and strain-energy density.

Open these files in ParaView and color by `jacobian`, `energy_density`, or `body_id`, or apply
a Glyph filter to the force vectors.

### Contact surfaces and overlap regions

`write_surface_vtp` writes TRI3, QUAD4, or general polygon cells as `PolyData`.

The coupled mortar patch writes:

- `slave-contact.vtp` with normal, gap, pressure, multiplier, support, activity, and contact
  force;
- `master-contact.vtp` with normal, contact force, and interface area.

The warped nonmatching adapter writes:

- `slave-contact.vtp` with nodal normal gap, pressure, support, and active-state fields;
- `master-contact.vtp` with integrated overlap area per master facet;
- `projected-overlap.vtp` with slave, master, and intersection polygons in the projection
  plane.

`projected-overlap.vtp` uses numeric `region_kind` values:

| Value | Region |
|---:|---|
| 0 | projected slave polygon |
| 1 | projected master polygon |
| 2 | clipped intersection polygon |

It also records `pair_index` and `projected_area` on every polygon cell.

## Numeric regression checks

Golden results must be compared numerically rather than by raw JSON or CSV equality.
`compare_numeric_metrics` accepts dotted metric paths and one `NumericTolerance` per metric:

```python
from contact3d.benchmark_artifacts import NumericTolerance, compare_numeric_metrics

compare_numeric_metrics(
    actual,
    reference,
    {
        "direct_newton.final_residual_norm": NumericTolerance(absolute=1.0e-10),
        "direct_newton.reaction_balance_norm": NumericTolerance(
            absolute=1.0e-11,
            relative=1.0e-8,
        ),
    },
)
```

The allowed error is

\[
  \varepsilon_{\mathrm{abs}}
  + \varepsilon_{\mathrm{rel}}\lvert q_{\mathrm{reference}}\rvert.
\]

Missing, nonnumeric, nonfinite, or out-of-tolerance metrics raise
`BenchmarkArtifactError` with the failed metric paths.

## Standardized runner

Run every migrated benchmark and validate all manifests with:

```bash
uv run python benchmarks/run_standardized.py \
  --output results/standardized-benchmarks
```

Run one benchmark by repeating `--benchmark` as needed:

```bash
uv run python benchmarks/run_standardized.py \
  --benchmark coupled-mortar-patch \
  --benchmark adaptive-contact-policy \
  --output results/standardized-benchmarks
```

The runner currently covers the core patch, bulk, coupled, adaptive, and production-interface
families:

- `tet4-patch`;
- `nonlinear-equilibrium`;
- `coupled-mortar-patch`;
- `adaptive-contact-policy`;
- `warped-nonmatching-adapter`.

It writes `suite-summary.json` only after every subprocess succeeds and every manifest passes
schema and file-completeness validation.

## Remaining issue-22 work

The core acceptance families—patch, bulk, coupled, and adaptive—now use the common contract.
The remaining work is:

- migrate the mixed-path, onset, scale-aware, event, BVH, and solver-scaling benchmarks;
- move repeated SVG chart implementations behind common plotting helpers;
- add versioned KKT, event, facet-pair, and mesh-refinement row schemas;
- export complete coupled volume/contact states from the adaptive onset benchmarks;
- add checked-in tolerance specifications for selected golden regression metrics.
