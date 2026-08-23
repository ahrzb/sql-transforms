---
id: TASK-122
title: >-
  Narrow-width modulo by -1 at INT_MIN does not overflow
status: Done
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 107000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT (i % -1) AS o FROM __THIS__      -- i is INTEGER, value INT32_MIN
-- DuckDB: Out of Range Error: Overflow in division of -2147483648 / -1
-- ours:   0
```

A wrong VALUE, served with no refusal. Measured 2026-08-17 across the widths,
and the bug is narrow `%` only — everything else already agrees:

| expression | i/t = width MIN | DuckDB | ours |
|---|---|---|---|
| `i % -1` | int32 | overflow | **0** |
| `t % -1` | int8 | overflow | **0** |
| `i // -1` | int32 | overflow | traps (range) |
| `t // -1` | int8 | overflow | traps (range) |
| `b % -1` | int64 | overflow | traps |
| `b // -1` | int64 | overflow | traps |
| `i % -2` | int32 | 0 | 0 |

The asymmetry has a cause. `MIN / -1` is `2147483648`, outside int32, so
TASK-118's range trap on the RESULT catches the division. `MIN % -1` is
mathematically 0, which is inside int32, so nothing fires — but DuckDB
computes the modulo through the same checked division and overflows anyway.
i64 is unaffected because the i64 lane performs the real operation and its own
`irem` guard fires.

So this is the one narrow-width overflow that the result-range check
structurally cannot see: the overflow is in the OPERATION, not in the value it
produces. The i64 lane needs the same INT_MIN-by-minus-one guard applied at
the narrow width.

Found by seed 2668 of the 2026-08-17 campaign (`c1 % ord('')`, where
`ord('')` is -1 — the fuzzer got there before a human would have).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 narrow `%` by -1 at the width's MIN traps, at int8, int16 and int32
- [x] #2 the guard is on the OPERATION, not the result range, since the result
      is in range — TASK-118's check cannot be extended to cover this
- [x] #3 `%` by any other divisor, and any dividend other than MIN, still
      serves and still matches
- [x] #4 int64 is untouched — it already traps through the lane's own guard
- [x] #5 the campaign's seed 2668 class is gone at 4000 seeds
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed 2026-08-19, exactly along the ticket's analysis: the guard is on the
OPERATION. In the Arith arm of `emit_kind`, a narrow-typed `%` gets a
runtime check `dividend == width_MIN && divisor == -1` -> trap, sharing the
block machinery TASK-118's range trap already had (factored to `trap_if`).
Masked payloads are defaults, so a NULL row structurally cannot fire it.
The message is DuckDB's own: `Out of Range Error: Overflow in division of
{MIN} / -1`.

The fold had the same blind spot one layer up: `checked_rem` in i64 folds
`MIN % -1` to 0, hiding the constant spelling from the runtime guard. The
fold now declines exactly that shape (the fold's contract is "None = the
runtime would trap"), so `(-128)::TINYINT % (-1)::TINYINT` traps
identically to the column form.

Swept 120 cases (4 widths x {MIN, MIN+1, -1, 0, 7, NULL} x {-2, -1, 1, 3,
NULL}) against optimizer-off DuckDB: 0 mismatches, including the
mixed-width control `TINYINT % SMALLINT` where -128/-1 = 128 FITS the
result width and serves 0 on both engines. i64 untouched (its lane guard
already fires). AC #5: the 4000-seed campaign no longer contains seed
2668's class.

Pins flipped strict and moved to `test_integer_widths.py`, with the
constant-spelling and mixed-width controls added.
<!-- SECTION:NOTES:END -->
