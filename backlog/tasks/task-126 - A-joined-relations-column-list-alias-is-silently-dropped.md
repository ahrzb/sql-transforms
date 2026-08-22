---
id: TASK-126
title: >-
  A joined relation's column-list alias is silently dropped
status: To Do
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
- [ ] #1 a column-list alias on a JOINed relation renames positionally, so
      `x.p` resolves and the original name stops resolving
- [ ] #2 the same for the comma-relation form
- [ ] #3 more names than the relation has columns refuses, as DuckDB does
- [ ] #4 refusing the whole construct is an acceptable first cut if the
      rename is more work than it looks -- what is NOT acceptable is
      dropping the clause and answering
- [ ] #5 a grep-level check that no other consumer of `plain_table()`'s
      return value partially consumes it, since the destructure being
      exhaustive is what made this invisible
<!-- AC:END -->
