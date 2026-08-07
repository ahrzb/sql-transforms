---
id: TASK-68
title: >-
  CASE or COALESCE over a joined column under shape='many' fails to build
status: To Do
assignee: []
created_date: '2026-08-08 01:20'
labels:
  - bug
  - lowering
dependencies: []
documentation: []
type: bug
ordinal: 61000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under `shape='many'`, any CFG-splitting expression over a **joined column**
refuses to build:

```text
SELECT COALESCE(d.v, 'z')                    AS v FROM t JOIN d ON t.pid = d.id
SELECT CASE WHEN d.v = 'a' THEN 'A' ELSE d.v END AS v FROM t JOIN d ON t.pid = d.id
SELECT NULLIF(d.v, 'a')                      AS v FROM t JOIN d ON t.pid = d.id
  -> internal specializer bug: lowered program failed verification:
     b6[1]: @0 is a multimap: use probe.range

SELECT d.v AS v FROM t JOIN d ON t.pid = d.id        -- control: OK
```

Reproduced 2026-08-08 on all three constructs, LEFT and inner join alike.

**Root cause.** `reseed_many` (`lower.rs:1476`) repopulates the current
block's probe cache from the live stack's trailing lanes, and
`store_out_row` (`:1494-1497`) calls it **once before each output
expression**. Its own doc comment names the hazard — "emission may split
blocks, and each new block starts with an empty cache" — but a split
*inside* one expression is exactly the case it does not cover. `case`
(`lower.rs:1733+`) calls `create_block`, whose `PB::new` starts with an empty
`probes` map, and never reseeds. So a joined column read inside a CASE arm
misses the cache and falls through to the scalar `Inst::Probe`, which
`verify.rs:613` rejects for a multimap.

**Distinct from TASK-66**, though found while fixing it: different site
(`verify.rs:613` vs `:294`), different message, no `tree_predict` involved,
and it needs no model set at all. TASK-66 stranded VALUES across a split;
this strands the per-block probe CACHE.

Build-time and loud, so not a wrong answer — but `shape='many'` is the only
shape under which join multiplicity builds, and a `COALESCE` over a joined
column is ordinary SQL.

Pinned by `test_duckdb_stageb_many.py::test_split_over_a_joined_column_under_many`
(xfail-strict, three constructs), so it cannot silently start or stop failing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `COALESCE`, `CASE` and `NULLIF` over a joined column build and match
      the DuckDB oracle under `shape='many'`, inner and LEFT
- [ ] #2 The existing xfail-strict pin is removed, not weakened
- [ ] #3 A case with the split in the JOIN CONDITION, and one with two output
      columns where only the second splits, are covered
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The reseed is at the wrong granularity: per output expression, when the cache
dies per block. Options are to reseed on every block transition under a
many-join (probably inside whatever calls `rebind_live`, so it cannot be
forgotten arm by arm), or to make the many-join lanes ordinary live-stack
residents so the cache is not a separate mechanism to keep in sync.

Worth checking whether the scalar-join probe cache has the same shape of hole
and merely fails silently — it would re-emit a probe per block rather than
tripping verify, which is a performance bug rather than a refusal, and so
would not have surfaced.
<!-- SECTION:NOTES:END -->
