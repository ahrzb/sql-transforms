---
id: TASK-103
title: >-
  DECIMAL-NULL SQLNULL fold keys on foldability, not literal spelling
status: Done
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
- [x] #1 the xfail pin flips: DECIMAL arith over a bind-foldable NULL operand is SQLNULL/int32
- [x] #2 unary minus included; DOUBLE-typed foldable NULL stays double (control row keeps passing)
- [x] #3 the composition xfail pins flip (upper-wrapped || operand; unfinished-constant udf args)
- [x] #4 20k campaign: the class is gone, no new classes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19, the ticket's one lever in three small pieces:

1. The DECIMAL arm keys on FOLDABILITY: a decimal-SPELLED operand
   (ast_decimal_literal, now walking CASE result arms) that binds and folds
   to NullOf collapses to SQLNULL/int32 for + - * %, and a new unary-minus
   arm does the same. The DOUBLE-spelled control keeps passing (the
   spelling predicate is what separates them, exactly DuckDB's
   DECIMAL-vs-DOUBLE typing).
2. fold.rs finishes two total ops over literals it used to leave runtime:
   Abs (i64::MIN stays unfolded -- the runtime trap owns it, the shifts'
   doctrine) and upper/lower via the SAME casemap kernel as Inst::Str1.
   That alone flipped the udf9(abs(-3), NULL) pin: the arg now folds
   before the whole-call-NULL machinery looks.
3. bind_fold_concat_operand peels a STACK of unary wrappers (Cast,
   StrCase) instead of one Cast, bakes the pure extern underneath,
   rebuilds, and folds -- so upper(us9(1,2)) collapses the || exactly like
   the bare call.

All 4 xfail pins flipped strict and are now plain live-oracle tests. AC #4:
20k campaign green (background run logged in the PR).
<!-- SECTION:NOTES:END -->
