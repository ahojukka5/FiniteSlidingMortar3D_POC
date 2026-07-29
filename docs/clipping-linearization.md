# Topology-frozen clipping linearization

The projected slave and master facets are convex polygons in the non-mortar center plane. Their intersection is constructed with Sutherland–Hodgman clipping. The clipping map is piecewise smooth: inside/outside classifications and the number and lineage of output vertices change when a vertex reaches a clip edge or two relevant edges become parallel.

This implementation separates those outer topology events from the smooth Newton derivative.

## Frozen operation trace

`trace_clipping_topology` evaluates the base configuration and stores, for every clip edge:

- the input vertex count;
- the inside/outside state of every input vertex;
- every retained vertex;
- every intersection operation and the two input vertices that generate it.

`replay_clipping_topology` applies the same operations to a perturbed configuration without repeating the classifications. This gives a direct finite-difference oracle for one smooth clipping branch.

A vertex whose signed distance to a clip edge lies inside the configurable event band raises `ClippingTopologyError`. Likewise, a parallel or numerically singular line intersection is rejected. These states require a separate generalized derivative or event-handling policy; silently selecting one branch would reproduce the kind of support discontinuity found in the repaired two-dimensional implementation.

## Intersection derivative

For an input segment from `p` to `q` and a clip line from `a` to `b`, define

```math
\mathbf e = \mathbf b-\mathbf a,
```

```math
d_p = \mathbf e \times (\mathbf p-\mathbf a),
\qquad
d_q = \mathbf e \times (\mathbf q-\mathbf a),
```

```math
\lambda = \frac{d_p}{d_p-d_q},
\qquad
\mathbf x = \mathbf p + \lambda(\mathbf q-\mathbf p).
```

The derivative propagated by `linearize_clipping_topology` is

```math
\mathrm d\lambda =
\frac{(\mathrm dd_p)(d_p-d_q)
-d_p(\mathrm dd_p-\mathrm dd_q)}{(d_p-d_q)^2},
```

```math
\mathrm d\mathbf x =
\mathrm d\mathbf p
+\mathrm d\lambda(\mathbf q-\mathbf p)
+\lambda(\mathrm d\mathbf q-\mathrm d\mathbf p).
```

Both the moving subject edge and moving clip edge are included.

## Facet-pair chain

`linearize_facet_intersection` composes the projection-plane layer from the previous slice with clipping:

1. build the slave-defined projection frame and its analytical Jacobian;
2. project slave vertices and combine their direct and moving-frame terms;
3. project master vertices and retain separate slave-frame and master-point terms;
4. assemble one global Jacobian column ordering, slave coordinates first and master coordinates second;
5. trace and analytically differentiate the convex intersection polygon.

The result supplies the intersection vertices and a tensor with axes

```text
(intersection vertex, projected component, global coordinate DOF).
```

## Verification boundary

The clipping Jacobian is compared against centered differences that replay the frozen operation trace. A randomized sweep of 100 warped partial-overlap `QUAD4` facet pairs produced a maximum absolute column error of `5.154e-9` in the development environment. Common rigid translation cancels to machine precision.

This slice does not yet differentiate the centroid-fan pallets, signed areas, inverse parent maps, quadrature shape values, or assembled `D` and `M` operators. Exact on-edge and edge-on-edge states are deliberately diagnosed rather than assigned a generalized derivative.
