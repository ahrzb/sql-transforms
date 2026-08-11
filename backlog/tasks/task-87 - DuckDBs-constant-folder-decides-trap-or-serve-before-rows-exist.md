---
id: TASK-87
title: >-
  DuckDB's constant folder decides trap-or-serve before rows exist; the engine decides per row
status: Done
assignee: []
created_date: '2026-08-11 16:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/src/specializer/fold.rs
type: bug
ordinal: 80000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One root, four measured faces (fuzz round 2, 2026-08-11 — every claim below
re-measured directly after two initial inferences proved WRONG):

**A. A trapping constant serves here, errors there — even over zero rows.**

```text
SELECT nullif(CAST('one' AS DOUBLE), -1.5e0) FROM __THIS__ WHERE FALSE
  duckdb  Conversion Error (its folder evaluates constants at PLAN time)
  ours    0 rows (row-driven: nothing evaluated)          [seed 37231]
```

**B. Literal integer arithmetic folds in wide range analysis there.**

```text
SELECT 9223372036854775807 * -50            duckdb: ERRORS   ours: traps  (agree)
SELECT 9223372036854775807 > (9223372036854775807 * -50)
  duckdb  TRUE (folds the comparison via wide arithmetic, no overflow)
  ours    ValueError: Overflow in multiplication of INT64  [seed 38091]
```

**C. A dead-range BETWEEN is eliminated there — in WHERE only.**

```text
SELECT s FROM t WHERE (CAST(s AS BIGINT) BETWEEN 22 AND 10)   -- s = 'one'
  duckdb  [] (filter folds lo>hi to FALSE, the cast never runs)
  ours    ValueError: could not cast VARCHAR to BIGINT       [seed 42002]
MEASURED: in a PROJECTION DuckDB evaluates and traps — both engines agree
there; the elimination is filter-side only.
```

**D. A constant CASE producing NULL is invisible to the TASK-85 fold.**

```text
SELECT (CASE WHEN TRUE THEN NULL WHEN FALSE THEN -2.5e0 END) * sqrt(-83.025e0)
  duckdb  NULL rows (CASE folds to NULL, x*NULL folds, sqrt eliminated)
  ours    ValueError: cannot take square root                [seed 35476]
TASK-85's strict-op check sees literal `SKind::NullOf` only; a FOLDED
constant CASE that lands on NULL slips past it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A constant expression whose fold TRAPS refuses at build by name
      (DuckDB errors on every execution of it, row count irrelevant)
- [ ] #2 An all-literal integer comparison folds through i128 arithmetic to
      DuckDB's answer; an i64-overflowing literal subtree reaching any
      non-comparison context refuses by name
- [ ] #3 `e BETWEEN lo AND hi` with constant `lo > hi` folds to FALSE in the
      WHERE path (eliminating `e`), and stays EAGER in projections — the
      measured DuckDB split, pinned on both sides
- [ ] #4 A constant CASE folds before the strict-op NULL check runs, so
      TASK-85's elision covers folded NULLs too
- [ ] #5 Differential pins for all four faces, both backends
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All four are the same missing move: run (more of) DuckDB's constant folding
at build, and treat a fold that traps as a named refusal instead of a
deferred runtime question. Certification seed 8359 (nullif/ln, DuckDB
serving) did not survive re-measurement in isolation — re-verify it after
these land and drop or ticket what remains.
<!-- SECTION:NOTES:END -->
