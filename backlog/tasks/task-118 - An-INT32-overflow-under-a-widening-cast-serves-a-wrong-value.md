---
id: TASK-118
title: >-
  An INT32 overflow under a widening cast serves a wrong value
status: To Do
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

TASK-84's block in test_known_divergences.py names `CAST(k AS INTEGER) * 2` as
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
- [ ] #1 an INT32-typed intermediate that overflows traps regardless of what
      consumes it - widening cast, comparison, function argument
- [ ] #2 the range check happens on the INT32 result, not on the lane, so a
      later widening cannot skip it
- [ ] #3 int8 and int16 intermediates are covered by the same rule, not just
      int32
- [ ] #4 in-range arithmetic under a widening cast still serves and still
      matches DuckDB - the guard must not cost the common path a refusal
- [ ] #5 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->
