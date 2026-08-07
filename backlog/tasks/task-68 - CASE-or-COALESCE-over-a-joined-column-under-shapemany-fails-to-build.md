---
id: TASK-68
title: >-
  CASE or COALESCE over a joined column under shape='many' fails to build
status: Done
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
- [x] #1 `COALESCE`, `CASE` and `NULLIF` over a joined column build and match
      the DuckDB oracle under `shape='many'`, inner and LEFT
- [x] #2 The existing xfail-strict pin is removed, not weakened
- [x] #3 A case with the split in the JOIN CONDITION, and one with two output
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

## Resolution (2026-08-08)

Reseeding moved from per-output-expression to per-BLOCK-TRANSITION, which is
the granularity the cache actually has.

- `FB.many: Option<ManySeed>` holds the active many-join while one expression
  is emitted under it.
- `enter_block(live, params)` replaces all nine bare `Self::rebind_live(...)`
  call sites: rebind, then restore what the new block cannot inherit. One
  chokepoint, so a future arm that splits the CFG cannot forget.
- `emit_many(e, live, j, hit, nd)` replaces the four `reseed_many` + `emit`
  pairs, and clears `many` afterwards.

**The part that is easy to get wrong:** `ManySeed` records `base`, the live
stack POSITION of the join's value lanes, not `nd` alone. The old
`live[live.len() - nd..]` is only the right slice at an expression boundary —
once an enclosing multi-operand expression has pushed operands on top, the
trailing lanes are somebody else's, and seeding from them wires the cache to
the wrong registers and answers with the wrong column. Mutation-checked:
restoring `live.len() - nd` fails exactly one test,
`test_split_inside_a_multi_operand_expression`
(`d.v || (CASE WHEN d.id > 1 THEN 'x' ELSE d.v END)`), which was written for
that reason. Nothing else catches it.

Second mutation, dropping the reseed from `enter_block`: 10 of 13 fail.

13 oracle-checked tests now cover: three splitting constructs, LEFT join
(a separate seed site — `hit=false`, different lane set), a split in a later
output column only, splits in WHERE on both the matched and null-extended
paths, the joined column read twice across a split, nested splits, and the
multi-operand case above.

**Not changed:** a SCALAR join loses its probe cache across a split too, but
`emit_probe` simply re-emits `Inst::Probe` in the new block — correct, and
free when the split is a branch (only one arm runs). It costs a redundant
probe only when a column before the split and one inside it both read the same
join. Small, pre-existing, and a different fix; deliberately left alone.
<!-- SECTION:NOTES:END -->
