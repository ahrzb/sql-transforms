---
id: TASK-103
title: >-
  DECIMAL-NULL SQLNULL fold keys on foldability, not literal spelling
status: To Do
assignee: []
created_date: '2026-08-13 14:00'
labels:
  - m-8
dependencies: []
type: bug
ordinal: 95000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR #130's certification campaign (seed 20275804), verified
live: `(- (CASE WHEN FALSE THEN 1.25 END))` and
`(1.25 + (CASE WHEN FALSE THEN 1.25 END))` are SQLNULL/INTEGER on
DuckDB; ours types double. Phase 2's DECIMAL (+|-|*|%) bare-NULL arm
keys on the literal SPELLING (`ast_decimal_literal` + bare NULL);
DuckDB's actual rule is operand FOLDABILITY — the same binder fold
behind TASK-102, now expressible with `plan::bind_foldable`. Unary
minus over a DECIMAL foldable-NULL collapses too (no unary arm exists
at all). DOUBLE-typed foldable NULLs do NOT collapse (measured,
catalogue control row). Xfail-strict pin:
test_integer_widths.py::test_decimal_arith_over_foldable_null_collapses.
Pre-existing on master (not a #130 regression — the arm only touches
StringConcat); surfaced by the fresh campaign seed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the xfail pin flips: DECIMAL arith over a bind-foldable NULL operand is SQLNULL/int32
- [ ] #2 unary minus included; DOUBLE-typed foldable NULL stays double (control row keeps passing)
- [ ] #3 20k campaign: the class is gone, no new classes
<!-- AC:END -->
