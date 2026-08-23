---
id: TASK-126
title: >-
  A joined relation's column-list alias is silently dropped
status: Done
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 111000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AS x(p, q)` renames a relation's columns positionally. The driving-table arm
(`frontend.rs:646-683`) consumes the column list properly. The JOIN arm
(`frontend.rs:730`) and the comma-relation arm (`frontend.rs:925`) take
`alias.name.value` and discard `alias.columns`.

Measured 2026-08-18, static `s(a BIGINT, b BIGINT)` = `(1, 99)`, row
`__THIS__(k)` = 1:

```sql
SELECT x.a AS o FROM __THIS__ JOIN s AS x(p, q, r) ON x.a = __THIS__.k
-- oracle: Binder Error: table "s" has 2 columns available but 3 columns specified
-- ours:   [{'o': 1}]

SELECT x.b AS o FROM __THIS__ JOIN s AS x(p, q) ON x.b = __THIS__.k
-- oracle: Binder Error: Table "x" does not have a column named "b"
-- ours:   []

SELECT x.p AS o FROM __THIS__ JOIN s AS x(p, q) ON x.p = __THIS__.k
-- oracle: [(1,)]
-- ours:   bind error: column 'p' does not exist in 'x'
```

So all three directions are wrong: we serve past an arity error, we serve a
query whose column reference DuckDB rejects, and we refuse the rename that
actually works. The comma form behaves identically.

This is the same class `972fd72` closed, and it survived that audit, because
`plain_table()` does destructure `TableFactor::Table` exhaustively in one
place -- and then two of its three CALL SITES partially consume the `alias`
it hands back. The doctrine has to reach the consumer, not only the
destructure.

Found by an independent review of the divergence-sweep branch, 2026-08-18,
and reproduced by hand before ticketing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 a column-list alias on a JOINed relation renames positionally, so
      `x.p` resolves and the original name stops resolving
- [x] #2 the same for the comma-relation form
- [x] #3 more names than the relation has columns refuses, as DuckDB does
- [x] #4 (not needed -- the rename itself landed) refusing the whole construct is an acceptable first cut if the
      rename is more work than it looks -- what is NOT acceptable is
      dropping the clause and answering
- [x] #5 a grep-level check that no other consumer of `plain_table()`'s
      return value partially consumes it, since the destructure being
      exhaustive is what made this invisible
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed 2026-08-19. `apply_column_alias` (frontend.rs) renames positionally
over the DECLARED columns -- TASK-125's `star` list, which is exactly the
structure a positional rename needs: `Real(ci)` renames `cols[ci]`, a name
landing on an `Opaque` entry (struct / non-vocabulary column) refuses by
name, more names than declared columns is DuckDB's own arity error text.
Both arms call it where the catalog table attaches, BEFORE `bind_on`, so
`ON x.p = ...`, USING, stars and EXCLUDE all see the new names. The renamed
table rides the existing `Cow` as Owned; no rename stays Borrowed.

Corners measured against the oracle beyond the pins (all matching): partial
prefix rename, second-column reference, `x.*` and `EXCLUDE` over renamed
names, USING through a renamed key, old-name shadowing. One DELIBERATE COST
kept: `AS x(p, p)` -- DuckDB serves the first match, we refuse as ambiguous
(severity 4, pinned in the wave5 file with the measurement date).

A column-list alias on a SELF-join now refuses by name ("column-list alias
on a self-join") instead of being dropped -- serving it there is unpinned
and nobody asked.

AC #5 audit: three `plain_table` callers remain -- the driving arm (full
consumer since wave 5), and the two join arms, both now routing through
`apply_column_alias`. No partial consumer left.

Pins flipped strict and moved to `test_duckdb_wave5_structural.py`, next to
the driving-arm alias tests they mirror.
<!-- SECTION:NOTES:END -->
