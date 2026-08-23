---
id: TASK-124
title: >-
  Boolean short-circuit is modelled per-statement, not per-context
status: Done
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 109000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four measured divergences, one cause. `FB::in_filter` (`lower.rs:391`, read at
`lower.rs:2029`) is a STATEMENT-level boolean: true for the whole predicate of
a filter-shaped query, false everywhere else. DuckDB's laziness is not a
property of the statement. It is a property of the CONTEXT a boolean
subexpression sits in, and that context recurses.

Where DuckDB is lazy:

* a filter's conjunction tree, recursively THROUGH nested AND/OR -- not only
  at the top level
* a `CASE WHEN` condition, including in a PROJECTION, which `in_filter`
  reports as false
* in that context an AND drops on a NULL left operand without evaluating the
  right, not only on a deciding FALSE

Where DuckDB is eager, and we are wrongly lazy:

* the operand of `NOT`, of `IS NULL`, of a comparison, of a function argument.
  These are VALUE context even when the whole thing sits under a `WHERE`.

Measured 2026-08-18, DuckDB 1.5.5, table `t(b BOOLEAN, s VARCHAR)` with rows
`(NULL,'abc')`, `(true,'1.5')`, `(false,'abc')`, and `T` = `CAST(s AS DOUBLE) > 1`:

| query | duckdb off | duckdb on | ours |
|---|---|---|---|
| `WHERE (b AND T) OR TRUE` | 3 rows | 3 rows | **TRAP** |
| `SELECT CASE WHEN (b AND T) THEN 1 ELSE 2 END` | `[2,1,2]` | `[2,1,2]` | **TRAP** |
| `WHERE CASE WHEN (b AND T) THEN TRUE ELSE TRUE END` | 3 rows | 3 rows | **TRAP** |
| `WHERE NOT (b AND T)` (row `(false,'abc')` only) | TRAP | TRAP | **1 row** |
| `WHERE (b AND T) IS NULL` (row `(false,'abc')` only) | TRAP | TRAP | **`[]`** |
| `WHERE b AND T` | `[('1.5',)]` | `[('1.5',)]` | `[('1.5',)]` |

The last row is the case commit `dfb3a99` measured and fixed, and it is still right.
The model simply does not extend one level down.

Rows 1-3 are the intolerable direction: we trap at runtime where DuckDB
serves, and here even optimizer-ON DuckDB serves, so the oracle choice is not
a defence. Rows 4-5 are the mirror: we answer a query DuckDB refuses to
answer.

The same cause reaches the many-join filter. `flatten_and` runs only on the
scalar path (`lower.rs:230`); `lower.rs:1627` and `lower.rs:1685` set
`in_filter` around `emit_many` but lower the predicate as ONE expression, and
`kleene_shortcut` drops only on a deciding operand, never on a NULL one:

```sql
SELECT s0.v FROM __THIS__ JOIN s0 ON c0 = s0.k WHERE (b AND CAST(s AS DOUBLE) > 1)
-- rows (1, NULL, 'abc'), (1, true, '1.5');  s0: k=1 twice, v in (10, 20)
-- duckdb off: [(20,), (10,)]   duckdb on: [(10,), (20,)]   ours: TRAP
-- the same predicate without the join serves correctly
```

Same for LEFT JOIN.

Found by a Fable 5 review of the divergence-sweep branch, 2026-08-18, and
reproduced by hand before ticketing. The 4000-seed campaign that gated that
branch reported zero traps-where-DuckDB-serves while all of this was reachable
one nesting level down -- see AC #6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 selection context is a per-node property, not a statement flag:
      it propagates recursively through AND/OR trees rooted at a filter
      predicate or at any `CASE WHEN` condition, projections included
- [x] #2 in selection context an AND drops on a NULL left operand, not only
      on a deciding FALSE -- the spine rule, generalized off the top level
- [x] #3 the operand of NOT, IS NULL, a comparison, or a function argument
      reverts to eager VALUE semantics, even under a WHERE
- [x] #4 the many-join filter sites (`lower.rs:1627`, `lower.rs:1685`) get
      the same treatment as the scalar path
- [x] #5 all six rows of the table above match optimizer-off DuckDB, and the
      last row -- `dfb3a99`'s case -- still passes
- [x] #6 the fuzz grammar generates nullable-bool conjuncts under OR and
      under CASE WHEN, so this class is reachable, and a 4000-seed campaign
      is clean AFTER that change -- the current grammar cannot see it
- [x] #7 `FB::in_filter` and the write-only `Binder::in_filter`
      (`frontend.rs:570`, `572`, `719`, `1465`) are both gone
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19, per the spec
(docs/superpowers/specs/2026-08-19-selection-context-design.md), which was
measured FIRST and corrected three of this ticket's own guesses -- most
importantly: OR is not blanket-lazy, it is AND's exact dual (TRUE left skips
the right; NULL/FALSE left evaluates it), and my first OR probe was a bad
discriminator (coalesce(false, TRUE) is false) that TASK-75's old pin
caught during implementation. Taken CASE arms are VALUE context even under
WHERE; selection context enters ONLY at the WHERE root and CASE conditions.

`emit_truth` (lower.rs): a bare-i1 TRUE-ness emission mode -- And and Or as
dual skip-branches, everything else exits to `emit` + truthy. Consumers: the
scalar WHERE (replacing the flatten_and spine, whose per-conjunct drop was
the top-level special case), both many-join WHERE sites (emit_truth_many,
the same wrapper shape as emit_many), and every CASE condition inside
`case()`. Deleted: FB::in_filter, flatten_and, kleene_shortcut, and the
may_trap gate on kleene -- value context is now ALWAYS eager.

One frontend hole the pins caught: fold.rs let a constant-TRUE-condition
CASE's arm (and the all-arms-dropped default) escape the CASE bare, moving
it from value context into whatever context the CASE occupied. Non-constant
escapes now keep a one-arm CASE as their value-context wrapper.

All 10 pins flipped strict (7 original + 3 from the spec's measurements) and
moved to known_divergences/test_short_circuit.py as a 21-row live-oracle
matrix plus the many-join pair. Full suite green in RELEASE and DEBUG builds
(2920 passed, 9 xfailed). AC #6: the grammar's new selctx arm emits the
nullable-bool-conjunct class (~12/2000 seeds); the 4000-seed campaign after
is clean in the forbidden direction -- the class now shows up as
DIVERGE_OPT (we match the oracle, optimizer-ON rewrites the overflow away),
the documented standing cost. The campaign's rng shift also re-rolled seed
380 into a PRE-EXISTING star-naming divergence (DuckDB dedupes colliding
join-star names to c0_1; we emit duplicates) -- pinned and ticketed as
TASK-130, not fixed here.
<!-- SECTION:NOTES:END -->
