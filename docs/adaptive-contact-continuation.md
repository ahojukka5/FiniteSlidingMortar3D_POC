# Adaptive contact continuation

The coupled Newton and augmented-Lagrange solvers operate at one prescribed load factor. This layer adds a transactional continuation policy around them without changing the residual, tangent, active-set, or multiplier-update formulations.

## Load-step policy

Starting from the last accepted factor `s_n`, a candidate is

```text
s_trial = min(s_target, s_n + Delta s).
```

A converged coupled augmented-Lagrange solve commits its displacement, multiplier states, interface penalties, and load factor together. An inexpensive accepted step may grow the next increment. Any failed inner equilibrium or exhausted penalty retry cuts the step back:

```text
Delta s <- beta_cut Delta s,    0 < beta_cut < 1.
```

The solve terminates explicitly if the reduced increment would fall below the configured minimum. No failed candidate leaks displacement or multiplier state into the next attempt.

## Penalty escalation

A candidate that reaches coupled equilibrium but exhausts the allowed augmentation count can still be under-resolved in penetration. When enabled, the controller retries that same load factor with

```text
kappa_new = min(kappa_max, gamma_kappa kappa),    gamma_kappa > 1.
```

The equilibrated displacement and multipliers from the under-resolved attempt are used only as the retry predictor. The larger penalty is committed only after the candidate satisfies the requested KKT tolerances. If the retry subsequently fails and the load is cut back, the problem, displacement, and multiplier states all roll back to the last accepted load state.

The implementation supports the production `MortarContactInterface` through its frozen `ContactPair.normal_penalty` field. It also supports verification interfaces exposing a direct `penalty` dataclass field.

## Recorded history

Every attempt records

- start and candidate load factors;
- attempted step size;
- accepted, cut-back, or penalty-increase action;
- inner termination reason;
- augmentation and Newton counts;
- contact-event restart count;
- equilibrium residual and maximum penetration; and
- penalties before and after the attempt.

This is enough to reproduce the continuation decisions and prove that penalty changes were not silently applied across failed load states.

## Warped nonmatching production regression

`tests/test_warped_nonmatching_adapter.py` exercises the production moving-overlap adapter on a warped `QUAD4` slave facet against two nonmatching `TRI3` master facets. The overlap is deliberately generic: both triangle intersections have six vertices and remain away from clipping-event bands. The regression verifies

- both independently clipped facet pairs are integrated;
- at least three slave rows retain support and active pressure;
- local interface forces remain self-equilibrated; and
- the fully analytical directional tangent agrees with an independent centered difference.

The committed controller benchmark is intentionally a deterministic policy regression. The next physics benchmark will combine this continuation layer with a full moving-overlap coupled boundary-value problem and adaptive prescribed-displacement control.
