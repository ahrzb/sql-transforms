---
id: TASK-128
title: >-
  A row limit without ORDER BY freezes an answer DuckDB itself cannot repeat
status: Done
assignee: []
created_date: '2026-08-19 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 113000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the widened 2000-seed campaign after TASK-125 (seed 1784), reported
as `backend-values: cranelift != interpreter`. It is not a backend bug. It is
build-vs-build:

```sql
SELECT c1 AS g, avg(c1) AS v FROM S0 GROUP BY c1 FETCH FIRST 1 ROWS ONLY
```

The main binder refuses `FETCH FIRST`, but a static-tables-only query falls
through to the constant emitter (`eval_static_only`, duckdb/mod.rs), which
hands the whole SQL to DuckDB once at build and freezes the result. Measured
2026-08-19: the same query over the same four rows returns FOUR distinct
answers across twelve fresh DuckDB connections. Whichever one the build-time
run happens to get is frozen into the artifact, so two builds of the same fn
disagree with each other.

The doctrine already exists -- it is the argument that moved the oracle to
optimizer-off DuckDB: an answer that is not a function of the query (plus the
statics) is not a target. A row limit with no ORDER BY has no defined answer
to match, so the contract-legal move is to REFUSE it by name (AmirHossein's
call, 2026-08-19).

Where: the constant-emitter fallback still holds the sqlparser AST it fell
through from. When the query PARSED, inspect it before calling
`eval_static_only`: a row-limit clause (`LIMIT`, `OFFSET`, `FETCH`, `TOP`)
with no `ORDER BY` refuses hard instead of falling through. A query sqlparser
cannot parse cannot be inspected -- residual hole, believed tiny (every
row-limit spelling sqlparser knows does parse).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 a static-tables-only query with a row limit (`LIMIT`/`OFFSET`/
      `FETCH`/`TOP`) and no `ORDER BY` refuses at build, naming the clause
      and the reason (result depends on scan order, not the query)
- [x] #2 MEASURED 2026-08-19, ORDER BY alone is NOT the cure: a tie fed
      from a GROUP BY returned 2 distinct answers over 20 fresh connections
      (`GROUP BY g ORDER BY avg(v) LIMIT 1`, tied averages), while a unique
      sort key was stable at 1/20. Base-table ties looked stable (1/20) but
      that is accidental scan-order stability, not a guarantee. So the rule
      is NOT "has an ORDER BY" -- pending AmirHossein's pick between
      (a) refuse row limits on the constant path entirely, or (b) allow only
      when ORDER BY covers the full GROUP BY key (the provable-total case)
- [x] #3 probe the disease WITHOUT a limit: does the ROW ORDER of a
      multi-row constant `GROUP BY` result vary across fresh connections /
      builds? If yes, that is a separate hole in every unordered multi-row
      constant result -- ticket it, do not widen this fix silently
- [x] #4 seed 1784 classifies clean (REFUSED) in the campaign, and the
      fuzzer's hostile-clause arm counts the refusal as the expected outcome
- [x] #5 known-limitations.md gains the rule under the constant-emitter
      section, with the twelve-connection measurement as its ground
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19, decision (a): EVERY row limit on the constant path refuses,
ORDER BY or not.

`row_limit_clause` (frontend.rs) re-parses the SQL the fallback already
holds and walks the statement -- top-level `limit_clause` and `fetch`,
`SELECT TOP`, CTEs, derived tables, and both sides of a set operation -- so
a limit one nesting level down cannot slip through the way the star and the
alias did in their tickets. An unparseable query cannot be inspected and
falls through to DuckDB exactly as before (the residual hole the ticket
named; every limit spelling sqlparser knows does parse).

The gate sits in duckdb/mod.rs immediately before `eval_static_only`, so
nothing else changes: the constant path still serves aggregation, ORDER BY
and DuckDB dialect. Error: `row limit ({clause}) on a static-tables-only
query -- which rows survive depends on scan order, not the query`.

AC #2 was already measured on the ticket (ORDER BY does not fix ties fed
from a GROUP BY: 2 answers / 20 runs), which is what made (a) the pick.
AC #3's probe found the disease WITHOUT a limit -- raw DuckDB: 12 distinct
row orders / 12 connections on an unordered 200-group GROUP BY; our arrow
materialization looked stable over 6 builds, which is luck, not contract --
ticketed as TASK-129 with three options for AmirHossein, not widened here.
AC #4: seed 1784 now classifies clean (0 findings). AC #5: the rule and the
twelve-connection measurement are in known-limitations.md section 2.

Tests: 7 refusal spellings + an untouched-path control in
test_arrow_schema_api.py, written red-first.
<!-- SECTION:NOTES:END -->
