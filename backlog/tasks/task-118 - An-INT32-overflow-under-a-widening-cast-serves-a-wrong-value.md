---
id: TASK-118
title: >-
  An INT32 overflow under a widening cast serves a wrong value
status: Done
assignee: []
created_date: '2026-08-16 02:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 103000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT CAST((i + 1) AS BIGINT) AS o FROM __THIS__   -- i is int32, value 2147483647
-- DuckDB: Out of Range Error, overflow in addition of INT32
-- ours:   2147483648
```

Not a missed trap - a **wrong value**, served with no refusal anywhere. `i + 1`
is INT32 arithmetic on DuckDB and overflows; we compute the narrow lane in i64
(the deliberate erase strategy) and the widening cast then consumes the i64
result before any range check sees it.

TASK-84's block in test_open_divergences.py names `CAST(k AS INTEGER) * 2` as
its residual. That case is caught today. This one is not, and the difference
matters: the caught case refuses, this one answers.

The erase strategy is sound - i64 compute plus a range trap is bit-identical to
real int32 for every operator, because the width is observable ONLY as the trap
threshold and the output schema. Both must hold. Here the trap threshold is
never consulted because the value leaves through a wider type.

Found 2026-08-16 by an adversarial classification pass over the divergence
file; reproduced by hand before filing. Pinned xfail-strict as
`test_int32_overflow_under_a_widening_cast_does_not_serve_a_wrong_value`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 an INT32-typed intermediate that overflows traps regardless of what
      consumes it - widening cast, comparison, function argument
- [x] #2 the range check happens on the INT32 result, not on the lane, so a
      later widening cannot skip it
- [x] #3 int8 and int16 intermediates are covered by the same rule, not just
      int32
- [x] #4 in-range arithmetic under a widening cast still serves and still
      matches DuckDB - the guard must not cost the common path a refusal
- [x] #5 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The check moved from the OUTPUT BOUNDARY to the point of PRODUCTION: `emit`
in lower.rs wraps every expression whose type is I8/I16/I32 in a range trap
(CFG split + Term::Trap, the same shape the VARCHAR cast uses). The opt-out
is an ALLOWLIST -- Col/StaticCol (range-checked on the way in), Lit (refused
at build), NullOf, JoinHit, Case (forwards an already-checked arm) -- so an
operator added later is checked by default. That is AC #1/#2/#3 in one rule.

AC #4 turned out to be the hard half, and not for the reason the ticket
expected. DuckDB's optimizer SIMPLIFIES `x +- c <cmp> k` to `x <cmp> k-+c`, so
the arithmetic never runs there: `(i + 1) > 5` over INT32_MAX SERVES on
DuckDB while `(i + 1)` alone traps. A trap on the result alone would have
made us trap where DuckDB serves -- the one outcome the contract has no room
for -- so the rewrite is reproduced in `cmp` (frontend.rs). It is exact
arithmetic, not an approximation.

Measured before implementing, and each line is now a pinned row:

  (i + 1) > 5            serves   all six predicates, both operand orders
  (1 - i) > 5            serves   the subject swaps sides, the pred flips
  (k + 1) > 5            serves   BIGINT too, not only narrow widths
  (i + 2) > -2147483648  TRAPS    shifted constant leaves the width
  (i + 1) = 2147483648   TRAPS    k does not fit INTEGER, so the comparison
                                  is at BIGINT and the addition is left alone
  (i * 2) > 5            TRAPS    multiplication is not rewritten
  (i + j) > 5            TRAPS    nor a non-constant operand
  (i + 1) IN (5, 6)      TRAPS    nor through IN -- but BETWEEN does rewrite

The IN exclusion needed a Cell (`in_inlist`), because IN is OUR desugar into
the same `=` chain and DuckDB does not shift through it.

Two rows of `test_a_genuinely_trapping_one_sided_on_residual_is_still_refused`
turned out to be wrong about their own premise: `9223372036854775807 + n > 1`
cannot overflow on DuckDB, because the same rewrite removes the addition.
They now build and serve, with the ELSE-arm variants left as the real
trapping cases.

Phase 3 is not finished by this: TASK-99 still owns DOUBLE->narrow CAST
truncation, TRY_CAST's raw-double range check, and the string-cast semantics.
The interim narrow-emit refusals stay in place as a backstop rather than
being deleted, since they now fire only where nothing produced the value.
<!-- SECTION:NOTES:END -->
