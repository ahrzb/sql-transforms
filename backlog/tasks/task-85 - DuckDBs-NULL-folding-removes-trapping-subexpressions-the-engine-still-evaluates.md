---
id: TASK-85
title: >-
  DuckDB's NULL-folding removes trapping subexpressions the engine still evaluates
status: To Do
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/src/specializer/plan.rs
type: bug
ordinal: 78000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```text
SELECT (c2 / (ln(c2) + (c2 - NULL))) AS o0 FROM __THIS__   -- c2 negative
  duckdb  rows of NULL   (folds x + NULL -> NULL, ln never runs)
  ours    ValueError: cannot take logarithm of a negative number
```

DuckDB's optimizer constant-folds NULL through strict (NULL-propagating)
operators, ELIMINATING the sibling subexpression — so a trapping call under a
known-NULL operand simply never executes. The engine evaluates eagerly and
traps on rows the oracle serves.

TASK-75 fixed this shape for WHERE's AND/OR (branchless -> guarded); this is
the strict-operator sibling in projections. Other spellings from the same 20k
campaign (2026-08-11): `Overflow in addition/multiplication/subtraction of
INT64` under a NULL operand (seeds 4516, 2261, 8115), `string builder result
exceeds` under a comparison whose other side is NULL (seed 13217), ln (seed
3199). ~20 findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A strict operator with a statically-NULL operand folds to NULL at
      build time, on both backends, so its siblings never evaluate — matching
      DuckDB
- [ ] #2 The `may_trap` analysis (plan::may_trap, the TASK-74/75 single
      definition) stays the one place that answers "can this trap"
- [ ] #3 Pins for ln, i64 overflow and the giant-string builder under a NULL
      operand
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The safe direction: fold `op(.., NULL, ..) -> NULL` for STRICT ops in the
frontend's constant folder (it already knows a typed NULL); that is semantics
-preserving and removes the trap exactly where DuckDB removes it. The subtle
part is runtime NULLs: `x - y` where y is NULL only at runtime — DuckDB also
skips evaluating siblings there (lazy per-row NULL short-circuit?). Measure
first: probe whether DuckDB's elimination is optimizer-time only (literal
NULL) or row-wise. The fuzzer's campaign cases are all literal-NULL, so start
there; if row-wise NULLs also elide traps on DuckDB, that is the flag-lane
guard machinery from TASK-75 extended to strict operators.
<!-- SECTION:NOTES:END -->
