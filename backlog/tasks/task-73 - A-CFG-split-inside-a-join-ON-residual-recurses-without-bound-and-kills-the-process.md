---
id: TASK-73
title: >-
  A CFG split inside a join ON residual recurses without bound and kills the process
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - lowering
  - crash
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 66000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`DuckDBInferFn(...)` never returns and never raises - the process dies with
`STATUS_STACK_OVERFLOW` (0xC00000FD), or on some shapes spins allocating
(measured 227 MB and climbing).

```text
SELECT n, r.bud AS b FROM __THIS__ AS t JOIN r
  ON t.k = r.id AND n + COALESCE(r.bud, 0) > 50     -> 0xC00000FD
same with LEFT JOIN                                  -> 0xC00000FD
same without the COALESCE (no split)                 -> builds, correct
```

`emit_probe` seeds `blocks[cur].probes[j]` only in the `eval` block it creates
for the residual. Any CFG split inside that residual - a CASE arm, a
COALESCE/NULLIF over a nullable column, a guarded CAST - starts a fresh `PB`
whose `probes` map is empty, so the next `SKind::StaticCol { join: j }` misses
the cache and re-enters `emit_probe(j)`, which re-emits the residual, which
contains the split, which misses again.

Same on both backends: this is prepare-time lowering, before backend selection.

**This is a correction to TASK-68.** Its resolution note claims a scalar join
losing its probe cache across a split is "correct, and free when the split is a
branch (only one arm runs)". That is false when the split is inside the join's
OWN residual, and it was asserted without being tested.

**Reproduced by hand**, not just relayed.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 COALESCE / CASE / NULLIF / guarded CAST inside a join ON residual
      build and match DuckDB, inner and LEFT
- [ ] #2 A build-time input can never kill the process - worst case is a named
      refusal
- [ ] #3 TASK-68's resolution note is corrected, not left contradicting this
- [ ] #4 A recursion guard (depth cap or a re-entry flag on `emit_probe`) so
      the failure mode of any FUTURE cache hole is an error, not a stack death
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-68 fixed exactly this hole for the MANY-join cache by re-seeding on every
block transition (`enter_block`). The scalar path needs the same treatment, and
the residual case additionally needs re-entry protection because the miss is
recursive rather than merely wasteful.

The test for this must run in a SUBPROCESS - a stack overflow is not catchable
and would take the test session down.
<!-- SECTION:NOTES:END -->
