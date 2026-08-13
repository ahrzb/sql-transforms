---
id: TASK-101
title: >-
  Pure-by-default UDF bind fold (decided: do what DuckDB does)
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
updated_date: '2026-08-13 12:00'
labels:
  - m-8
dependencies: []
type: task
ordinal: 93000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DECIDED 2026-08-13 (AmirHossein: "do what duckdb does", bug-for-bug).
The original story ("DuckDB constant-folds all-constant-arg UDFs at
plan time") was wrong; the measured mechanism (probe battery
2026-08-13, pinned xfail-strict in test_integer_widths.py):

- DuckDB executes a python UDF at BIND only where the binder
  constant-folds an enclosing expression — field access `.f1`
  (struct_extract) folds its operand; `%` does not. Seed 601418
  diverged because of `(udf1(1, NULL, ...)).f1`, not the `%`.
- The fold is gated on `side_effects` (False = pure = DuckDB's
  default), NOT on null_handling. `side_effects=True` suppresses it.
- The fold honors special null handling: the udf RUNS with NULL args
  and its real result is used (a udf returning 99 for NULL input
  folds to 99/BIGINT — measured, never assumed NULL). None -> SQLNULL
  -> INTEGER; non-NULL -> constant of the declared type.
- A udf that raises during the fold fails the query at bind.

We mirror: protocol objects gain `side_effects: bool = False`
(same name and default as DuckDB); build may execute a pure udf when
its args are constants under a fold context, None routes through the
SQLNULL channel (null_of int32). Spec:
docs/superpowers/specs/2026-08-13-bind-fold-alignment-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the TASK-101 xfail-strict pin in test_integer_widths.py flips to passing
- [ ] #2 a pure udf returning non-NULL for NULL args folds to that value (contract pin)
- [ ] #3 side_effects=True udf is never executed at build; schema matches DuckDB's same flag
- [ ] #4 a raising pure udf under a fold context is a BUILD error (error-phase parity)
- [ ] #5 fold contexts match DuckDB's, probed (field access measured; probe ||-operand, cmp, arith before coding)
- [ ] #6 20k campaign: the seed-601418 class is gone, no new classes
<!-- AC:END -->
