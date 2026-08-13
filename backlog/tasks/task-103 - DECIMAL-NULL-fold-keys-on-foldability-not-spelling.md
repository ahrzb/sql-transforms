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

WIDENED by the TASK-101 adversarial review (2026-08-13): the same
root — our bind-time constant evaluator is strictly weaker than
DuckDB's — also leaves `upper(us9(1,2)) || s` and
`(udf9(abs(-3), NULL)).f1` at our declared types where DuckDB folds
the whole constant subtree to SQLNULL/int32 (xfail pins:
test_bind_fold_composition_gaps). One lever fixes all three
families: a stronger bind-fold that evaluates total ops and pure
externs over constants, gated on DuckDB's binder-foldability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the xfail pin flips: DECIMAL arith over a bind-foldable NULL operand is SQLNULL/int32
- [ ] #2 unary minus included; DOUBLE-typed foldable NULL stays double (control row keeps passing)
- [ ] #3 the composition xfail pins flip (upper-wrapped || operand; unfinished-constant udf args)
- [ ] #4 20k campaign: the class is gone, no new classes
<!-- AC:END -->
