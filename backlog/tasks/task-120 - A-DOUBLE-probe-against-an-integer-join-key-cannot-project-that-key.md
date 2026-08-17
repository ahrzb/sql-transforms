---
id: TASK-120
title: >-
  A DOUBLE probe against an integer join key cannot project that key
status: To Do
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 105000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT s.c0 AS o FROM __THIS__ JOIN s ON k = s.c0
-- row  k    is double
-- static c0 is int64
-- DuckDB: int64        ours (before TASK-115): double, value 5.0
-- ours (now):          refuses by name
```

A projected static key is reconstructed from the DYNAMIC side, which is
sound when the two sides share a lane. `promote_key`'s last arm does not:
a DOUBLE probe against an integer column compares in DOUBLE space and the
build side is converted while the probe table is built, so the
reconstruction holds a double. A double does not name one i64 — every i64
above 2^53 shares its double with neighbours, so two build rows can collide
on one probe key and the "reconstructed" value is not any particular one of
them.

TASK-115 fixed the two integer-width pairings by re-declaring the static
column's own width, which is exact there (DuckDB compares across widths
numerically, so a match already proves the value fits both). This pairing
has no such re-declaration, so TASK-115 made it REFUSE rather than serve a
double under an integer column's name. Only the PROJECTION of the key
refuses; the join itself and every other column are unaffected.

Found 2026-08-17 while fixing TASK-115 — the probe that measured both
integer directions measured this one too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a projected integer key column joined against a DOUBLE probe serves
      the STATIC column's real value at its declared type, not a
      reconstruction
- [ ] #2 the fix reads the real column (materialise the key as a probe VALUE
      lane when it is projected) rather than converting the double back —
      the conversion is the unsound part
- [ ] #3 multiplicity is measured first: two int64 build rows that collide on
      one double are one bucket for us; check what DuckDB emits for both
      shape='map' and shape='many' before choosing the fix
- [ ] #4 `test_a_double_probe_against_an_integer_key_refuses_to_project_it`
      in test_integer_widths.py is rewritten to assert the served answer
<!-- AC:END -->
