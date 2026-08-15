---
id: TASK-115
title: >-
  A projected join key takes the row side's width, not the static's
status: To Do
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
test_known_divergences.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a projected static key column types at the STATIC column's declared
      width, whatever the row-side key's width is
- [ ] #2 the reconstruction behaviour it rides on is unchanged — a LEFT miss
      is still NULL and still never coalesced, USING included
- [ ] #3 the reverse pairing (wide row key, narrow static key) is covered
      too, not just the narrowing direction
- [ ] #4 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->
