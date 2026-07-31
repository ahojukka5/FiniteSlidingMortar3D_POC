# Incremental contact-facet BVH

The contact broad phase must find every slave/master facet pair whose current
axis-aligned bounding boxes are no farther apart than the configured search
band. Missing one pair changes the mortar integration domain, while returning
extra pairs only adds inexpensive zero-overlap integration attempts. The search
therefore prioritizes conservative completeness and deterministic results.

## Reference topology and current refit

Each immutable `ContactSurface` builds one `FacetAABBTree` from reference facet
centroids. The tree uses deterministic median splits:

1. choose the coordinate with the largest centroid extent;
2. sort by that coordinate and then by the original facet index;
3. split the ordered range at its integer midpoint;
4. stop when a leaf contains at most eight facets.

Only the topology and facet permutation are retained from this build. During a
contact evaluation, `FacetAABBTree.refit` recomputes current facet AABBs and then
updates leaf and internal-node bounds from the bottom upward. No centroid sort,
partition, or topology allocation is repeated.

The refitted tree stores both per-facet and per-node bounds. These arrays are the
broad-phase cache for that current configuration.

## Slave/master query

`discover_facet_pairs_with_diagnostics` computes all slave facet AABBs once and
queries them against the refitted master tree. A tree node is rejected only when
the Euclidean distance between the slave AABB and the node AABB exceeds the
search distance. Because a parent AABB encloses every descendant, this pruning
cannot remove a qualifying facet pair.

The exact facet-box distance test is repeated inside accepted leaves. The final
pair list is sorted lexicographically, so pair ordering does not depend on leaf
size, traversal order, or median-split details.

`discover_facet_pairs` is the production wrapper and returns only the ordered
pairs. `discover_facet_pairs_with_diagnostics` additionally returns:

- tree and leaf counts;
- visited nodes and leaves;
- exact facet-box tests;
- accepted pairs;
- the corresponding quadratic test count and tested fraction.

The query API is deliberately slave-against-master. It can later accept
self-contact adjacency filters without changing tree construction or the
contact-integration interface.

## Verification oracle

`discover_facet_pairs_brute_force` retains the original quadratic nested loop.
Tests compare BVH and oracle sets on randomly deformed nonmatching surfaces and
verify that different leaf sizes produce exactly the same ordered result.

A contact-level regression supplies the oracle pair set explicitly to mortar
assembly and checks that overlap areas, `D`, `M`, and the residual are bitwise
identical to the default BVH path. Existing smooth-branch tangent tests then
continue to exercise the same integrated pair sets.

## Scaling benchmark

Regenerate the committed reference study with:

```bash
uv run python benchmarks/broad_phase_scaling.py \
  --output results/broad-phase-scaling
```

The study uses warped, nonmatching pairs of structured surfaces with 64, 256,
576, and 1024 facets per side. It writes:

- `scaling.csv` with operation counts and median timings;
- `summary.json` with equivalence and growth metrics;
- `operation-scaling.svg` comparing BVH and quadratic facet tests;
- `refit-cost.svg` comparing refit time with topology-build time.

In the committed reference run, all pair sets agree with the oracle. The BVH
facet-test count grows with fitted exponent about 1.22 rather than 2.0, and the
largest model tests about 3.9% of the quadratic facet pairs. Current-coordinate
refit costs between about 69% and 80% of a complete topology rebuild on the
reference machine. Timing values are machine dependent; pair equivalence and
operation counts are the deterministic regression quantities.
