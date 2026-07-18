---
id: decision-2
title: Error-type parity across engines is a non-goal
date: '2026-07-18 14:00'
status: accepted
---
## Context

`transform` (DataFusion) and `infer` (native) must return identical *values* on the
normal numeric path — non-negotiable, enforced by the differential harness. But the
two engines carry genuinely different failure information: e.g. integer
div/modulo-by-zero raises a clean `ValueError` from native vs a raw DataFusion
`Exception` ("Arrow error: Divide by zero error") from the batch path.

## Decision

**Matching the error type/message each engine raises is an explicit non-goal**
(2026-07-16). Only output *values* must match. Reconciling error hierarchies buys
nothing a user relies on.

## Consequences

- Div/mod-by-zero raising different error *types* on the two engines is accepted by
  design, not a gap. Locked in by a positive test
  (`test_div_by_zero_raises_on_both_engines_with_different_error_types`).
- The old "unify batch vs inference error semantics" item is descoped / won't-do.
- Supersedes decision-1 only on *errors*; value parity (decision-1) still binds.
