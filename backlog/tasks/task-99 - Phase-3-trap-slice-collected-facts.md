---
id: TASK-99
title: >-
  Phase-3 trap slice: collected facts
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 91000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The m-8 phase-3 work order, grown by the 2026-08-13 fleet: (a) INT32/16/8
overflow traps at the measured boundaries with DuckDB's message bodies;
(b) DOUBLE->narrow CAST truncation semantics (currently the lane never
truncates — silent wrong value as an intermediate); (c) TRY_CAST(DOUBLE
AS narrow) must check the RAW double half-open, not the rounded i64;
(d) TRY_CAST(VARCHAR AS int) accepts fractional/exponent strings on
DuckDB; (e) DuckDB's optimizer pushes fitting constants through TRY_CAST
(our NULL becomes their FALSE downstream). When traps land, DELETE the
eval_i32_literal/eval_i128_literal AST evaluators (subsumed) and the
narrow-emit interim refusals, and re-run a 20k campaign expecting the
DIVERGE_TRAP classes to close.

**2026-08-17 (TASK-118): (a) has LANDED.** INT32/16/8 overflow now traps at
the point of production, on every consumer, with DuckDB's comparison
simplification reproduced so the trap stays invisible exactly where DuckDB's
is. What remains here is (b) DOUBLE->narrow CAST truncation, (c) TRY_CAST
checking the RAW double, (d) TRY_CAST(VARCHAR AS int) accepting
fractional/exponent strings, and (e) the constant-through-TRY_CAST push. The
narrow-emit interim refusals are deliberately NOT deleted yet -- with the
production-site trap in front of them they now only fire where nothing
produced the value, so they are a cheap backstop rather than the mechanism.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the campaign's INT32/16/8 DIVERGE_TRAP classes are gone
- [ ] #2 the AST-side constant evaluators and interim emit refusals are deleted in the same PR
- [ ] #3 TRY_CAST double/string semantics match DuckDB, live-oracle pinned
<!-- AC:END -->
