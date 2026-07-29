# Literature review and formulation choice

## Primary implementation target

Michael A. Puso and Tod A. Laursen, **“A mortar segment-to-segment contact method for large deformation solid mechanics”**, *Computer Methods in Applied Mechanics and Engineering* 193 (2004), 601–629, DOI `10.1016/j.cma.2003.10.010`.

The paper provides the first implementation target because it gives a complete geometric integration algorithm for frictionless three-dimensional large-sliding contact with linear tetrahedral and trilinear hexahedral surface facets:

1. use the designated non-mortar facet to define a center tangent plane;
2. orthogonally project both facets to that plane;
3. clip the two convex projected polygons;
4. split the intersection polygon into triangular pallets;
5. evaluate the two parent-coordinate inverse maps at common pallet quadrature points;
6. assemble the non-mortar/non-mortar matrix `D` and non-mortar/mortar matrix `M` from the same physical quadrature points.

Using identical physical quadrature for both sides makes the row-wise partition-of-unity identity exact and therefore conserves linear momentum by construction. The original formulation uses a standard nodal pressure interpolation, nodally averaged non-mortar normals, a biased one-pass mortar discretization, and penalty or augmented-Lagrange enforcement. It deliberately omits some force terms, producing a nonsymmetric tangent. Friction is outside this paper's scope.

## Closely related sources

- Puso and Laursen, **“A mortar segment-to-segment frictional contact method for large deformations”**, CMAME 193 (2004), 4891–4913, DOI `10.1016/j.cma.2004.06.001`. This is the direct frictional extension and supplies the objective slip history needed after the frictionless kernel is verified.
- Puso, Laursen, and Solberg, **“A segment-to-segment mortar contact method for quadratic elements and large deformations”**, CMAME 197 (2008), 555–566, DOI `10.1016/j.cma.2007.08.009`. This generalizes the geometric ideas to quadratic faces and alternative multiplier interpolations.
- Popp, Gitterle, Gee, and Wall, **“A dual mortar approach for 3D finite deformation contact with consistent linearization”**, IJNME 83 (2010), 1428–1465, DOI `10.1002/nme.2866`. This replaces regularized enforcement with dual multipliers, primal-dual active sets, static condensation, and a fully consistent 3D tangent.
- Popp and Wall, **“Dual mortar methods for computational contact mechanics—overview and recent developments”**, GAMM-Mitteilungen 37 (2014), 66–84, DOI `10.1002/gamm.201410004`. This is the best roadmap from the standard formulation toward dual mortar, higher order, friction, and robust nonlinear solution.
- Farah, Wall, and Popp, **“A mortar finite element approach for point, line, and surface contact”**, IJNME 114 (2018), 255–291, DOI `10.1002/nme.5743`. This is important for sharp corners, edges, and crosspoints, where a pure surface-contact multiplier field can overconstrain the discrete problem.
- Puso and Solberg, **“A dual pass mortar approach for unbiased constraints and self-contact”**, CMAME 367 (2020), 113092, DOI `10.1016/j.cma.2020.113092`. This is a later route to unbiased and self-contact-capable formulations.

## Lessons carried over from the repaired 2D implementation

The 2D failure showed that physical mortar support must not depend on an arbitrary owner selected for a shared vertex. The 3D implementation therefore keeps broad-phase history and local facet-pair overlap construction separate. Every proximate facet pair is clipped independently; projected node, edge, or vertex ownership is not used to decide whether the pair owns finite area. Crosspoint multiplier treatment will be explicit rather than hidden in projection tie-breaking.

## Scope decision

The first implementation remains intentionally narrow:

- frictionless contact;
- linear `TRI3` and bilinear `QUAD4` surface facets;
- standard mortar interpolation;
- projected polygon integration with 1-, 3-, and 7-point triangle rules;
- exact row-wise momentum consistency checks.

The nonlinear constraint law, bulk mechanics, consistent tangent, augmented Lagrangian, dual basis, friction, point/line contact, and self-contact are separate follow-up slices.
