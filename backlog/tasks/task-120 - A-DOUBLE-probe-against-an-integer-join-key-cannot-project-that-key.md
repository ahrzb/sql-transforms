---
id: TASK-120
title: >-
  A DOUBLE probe against an integer join key cannot project that key
status: Done
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
- [x] #1 a projected integer key column joined against a DOUBLE probe serves
      the STATIC column's real value at its declared type, not a
      reconstruction
- [x] #2 the fix reads the real column (materialise the key as a probe VALUE
      lane when it is projected) rather than converting the double back —
      the conversion is the unsound part
- [x] #3 multiplicity is measured first: two int64 build rows that collide on
      one double are one bucket for us; check what DuckDB emits for both
      shape='map' and shape='many' before choosing the fix
- [x] #4 `test_a_double_probe_against_an_integer_key_refuses_to_project_it`
      in test_integer_widths.py is rewritten to assert the served answer
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec (docs/superpowers/specs/2026-08-25-task-120-design.md).
A lossy join key (a DOUBLE probe against any integer static key width)
rides as a shadow VALUE lane; a projected key reads that lane -- the
static column's real values at its declared type -- never a
reconstruction from the probe. Four resolution sites guard the shadow
lane (join-head ambiguity counting, unqualified, qualified, and star
expansion, which had to go key-first or a USING key would unmerge).

Multiplicity, measured first per the ticket: two i64 build rows
colliding on one double are TWO output rows under the fan-out loop, and
our many shape already emitted the right multiset -- the divergence was
only ever the projection.

Scope note: promote_key's widening covers int8/int16/int32 as well,
because those widths refused the WHOLE join ("cannot join f64 with i8")
before this branch -- the ticket named only the i64 pairing because
TASK-115 only measured it. A purpose-built 1500-case differential went
from 28 agree / 1340 refusals to 1288 agree / 4 NaN-sort-artifact cells
against the optimizer-off oracle. The shared fuzzer cannot reach this
class (gen.py only equi-joins same-typed columns) -- widening the
generator is noted in the spec as out of scope.
<!-- SECTION:NOTES:END -->
