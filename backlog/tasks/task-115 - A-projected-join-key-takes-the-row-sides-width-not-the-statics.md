---
id: TASK-115
title: >-
  A projected join key takes the row side's width, not the static's
status: Done
assignee: []
created_date: '2026-08-15 16:00'
labels:
  - m-8
dependencies: []
type: bug
ordinal: 100000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Projecting the static side of a join key reconstructs it from the dynamic
side, so the output takes the ROW column's width instead of the static
column's own declaration:

```sql
SELECT s.c0 AS o FROM __THIS__ LEFT JOIN s ON k = s.c0
-- row  k    is int8
-- static c0 is int64
-- DuckDB: int64      ours: int8
```

The reconstruction itself is deliberate and correct — `frontend.rs`
comments it as "qualified key access reconstructs from the dynamic side —
measured to stay addressable even after USING (NULL on a LEFT miss, never
coalesced)". What is wrong is only the TYPE it reconstructs at: it adopts
the row lane's width rather than re-declaring the static column's.

Non-key static columns are unaffected — `SELECT s.v` where `v` is not a key
already types correctly (verified: the first draft of the pin projected a
non-key column and XPASSed).

Found by the widened fuzzer (seed 1379) while verifying TASK-96, and only
reachable at all because that campaign now generates narrow static columns.
Pre-exists TASK-96 — identical count in campaigns before and after it.

Pinned xfail-strict as `test_projected_static_key_keeps_its_own_width` in
test_open_divergences.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 a projected static key column types at the STATIC column's declared
      width, whatever the row-side key's width is
- [x] #2 the reconstruction behaviour it rides on is unchanged — a LEFT miss
      is still NULL and still never coalesced, USING included
- [x] #3 the reverse pairing (wide row key, narrow static key) is covered
      too, not just the narrowing direction
- [x] #4 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`key_lane` now types the reconstruction from `key_cols[key_pos]` -- the
STATIC column's own declaration -- instead of from the dynamic key
expression. Between two integer widths that is a pure re-declaration and
needs no cast: DuckDB compares across widths NUMERICALLY, so a match already
proves the value fits both sides. Measured in both directions (int8 row vs
int64 static, int64 row vs int8 static) on INNER and LEFT, plus a same-width
control and a LEFT-miss control for AC #2.

The measurement also turned up a THIRD pairing the ticket did not name:
`promote_key`'s DOUBLE-probe-against-an-integer-column arm compares in
double space, so the reconstruction holds a double and cannot name one i64.
That one has no sound re-declaration; it now refuses by name and is
TASK-120.
<!-- SECTION:NOTES:END -->
