---
id: TASK-69
title: >-
  QUALIFY and FETCH FIRST are parsed then silently discarded
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - frontend
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/test_dropped_clauses.py
type: bug
ordinal: 62000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The frontend validates `query.with`, `order_by`, `limit_clause`,
`select.distinct`, `group_by` and `having`, and refuses `LIMIT` by name with
"unsupported: LIMIT/OFFSET". It never looks at `select.qualify`, and does not
recognise the `FETCH FIRST n ROWS ONLY` spelling of LIMIT. Both are dropped and
every input row is emitted.

```text
QUALIFY row_number() OVER (PARTITION BY k ORDER BY ts DESC) = 1
  engine [(1,1),(1,2),(2,5)]   duckdb [(1,2),(2,5)]
FETCH FIRST 1 ROWS ONLY
  engine [(1,1),(1,2),(2,5)]   duckdb [(1,1)]
LIMIT 1
  engine REFUSED               duckdb [(1,1)]
```

Worse than a wrong row count: `one_row_blocker` sees no Filter node, so
`shape='map'` (the exactly-one-row-out-per-row-in PROOF) also builds and
certifies a query whose whole purpose is to drop rows. The QUALIFY spelling
above is the standard dedupe-to-latest-per-key idiom in serving SQL.

**Reproduced by hand**, not just relayed.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/known_divergences/test_dropped_clauses.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 QUALIFY either builds and matches DuckDB, or is refused by name
- [x] #2 FETCH FIRST is treated exactly as LIMIT is
- [x] #3 Neither can be certified by `shape='map'` while dropping rows
- [x] #4 A test sweeps the sqlparser `Select`/`Query` fields for any OTHER
      clause that is parsed and ignored - this is a class, not two instances
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The refusal already exists; these two spellings simply are not in it. Grep the
frontend for every `query.`/`select.` field sqlparser exposes and check each is
either handled or refused - the same audit would have caught both.

## Resolution (2026-08-08): refuse, and fix it as a CLASS

Chose refusal over implementation (AC #1 allows either): the contract is
match-DuckDB-or-refuse, and silently ignoring a clause is the third mode that
is not supposed to exist.

`refuse_unhandled_query` and `refuse_unhandled_select` in `frontend.rs`
destructure their sqlparser AST node **exhaustively - no `..` pattern**. A
clause added to sqlparser now breaks the BUILD instead of the answers. That
mechanism paid for itself immediately: it caught `Select::flavor` (FROM-first
syntax), which my hand audit of the field list had missed.

The audit found this was never two bugs. Silently ignored before the fix:
`fetch`, `locks`, `for_clause`, `settings`, `format_clause`, `pipe_operators`,
`top`, `qualify`, `prewhere`, `exclude`, `into`, `lateral_views`, `connect_by`,
`cluster_by`, `distribute_by`, `sort_by`, `named_window`, `optimizer_hints`,
`select_modifiers`, `value_table_mode`, `flavor`.

`SELECT TOP n` was a third row-limiter dropped exactly like QUALIFY and FETCH
FIRST, and nobody had reported it.
<!-- SECTION:NOTES:END -->
