---
id: TASK-125
title: >-
  A static relation's star emits a column set DuckDB never produces
status: To Do
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 110000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`frontend.rs:1979-2010` expands a static relation's star by iterating
`sj.table.cols` directly. The ROW-table star at `frontend.rs:1952` interleaves
`StarLane::Opaque` and refuses when a column cannot be served. The static path
has no such interleave, so it answers with whatever lanes happen to exist.

Two spellings, measured 2026-08-18 against optimizer-off DuckDB, static
`s(id BIGINT, w STRUCT(mean DOUBLE, sd DOUBLE), z BIGINT)`, row `__THIS__(k)`:

```sql
SELECT s.* FROM __THIS__ JOIN s ON s.id = __THIS__.k
-- oracle: cols ['id','w','z']            [(1, {'mean': 1.5, 'sd': 0.25}, 7)]
-- ours:   {'id': 1, 'w.mean': 1.5, 'w.sd': 0.25, 'z': 7}
```

The struct leaves TASK-116 flattened into `cols` (`duckdb/mod.rs:1349`) reach
the star as if they were top-level columns, so the result carries two phantom
column names and is missing `w`. Same for bare `SELECT *`.

The opaque spelling is worse, because it is silent:

```sql
-- static s(id BIGINT, ts TIMESTAMP, z BIGINT)
SELECT s.* FROM __THIS__ JOIN s ON s.id = __THIS__.k
-- oracle: cols ['id','ts','z']   [(1, datetime(1970,1,1), 7)]
-- ours:   {'id': 1, 'z': 7}
```

A column vanishes from the output with no refusal anywhere. That is the exact
failure `972fd72` ("apply the no-dropped-clause doctrine to the relation")
set out to end, one level below where its audit stopped. The TIMESTAMP case
predates this branch; the struct case is new with TASK-116, and
`duckdb/mod.rs:1347` currently asserts the opposite in a comment -- "`s.w` as
a whole value, and `s.*`, are still unserved". `s.w` does refuse (measured).
`s.*` does not.

TASK-116's acceptance criteria never mention star expansion, which is why
nothing caught it.

Found by an independent review of the divergence-sweep branch, 2026-08-18,
and reproduced by hand before ticketing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the static star interleaves `StarLane::Opaque` exactly as the row
      star does, so an unservable column REFUSES by name instead of being
      dropped or expanded into leaves
- [ ] #2 a struct column under `s.*` or `*` refuses, naming the struct --
      it must never appear as `w.mean` / `w.sd` phantom columns
- [ ] #3 an opaque column (TIMESTAMP and friends) under a star refuses,
      naming the column; no output row is ever short a column silently
- [ ] #4 a static relation of only servable scalar columns still expands,
      in declaration order, unchanged
- [ ] #5 the `duckdb/mod.rs:1347` comment is corrected -- it currently
      states the bug's absence
- [ ] #6 the fuzz grammar emits `s.*` and `*` over a static relation
      carrying a struct and an opaque column, so the class is reachable
<!-- AC:END -->
