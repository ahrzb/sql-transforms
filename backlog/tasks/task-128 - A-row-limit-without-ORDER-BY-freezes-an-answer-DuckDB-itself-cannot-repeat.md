---
id: TASK-128
title: >-
  A row limit without ORDER BY freezes an answer DuckDB itself cannot repeat
status: To Do
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
- [ ] #1 a static-tables-only query with a row limit (`LIMIT`/`OFFSET`/
      `FETCH`/`TOP`) and no `ORDER BY` refuses at build, naming the clause
      and the reason (result depends on scan order, not the query)
- [ ] #2 MEASURED 2026-08-19, ORDER BY alone is NOT the cure: a tie fed
      from a GROUP BY returned 2 distinct answers over 20 fresh connections
      (`GROUP BY g ORDER BY avg(v) LIMIT 1`, tied averages), while a unique
      sort key was stable at 1/20. Base-table ties looked stable (1/20) but
      that is accidental scan-order stability, not a guarantee. So the rule
      is NOT "has an ORDER BY" -- pending AmirHossein's pick between
      (a) refuse row limits on the constant path entirely, or (b) allow only
      when ORDER BY covers the full GROUP BY key (the provable-total case)
- [ ] #3 probe the disease WITHOUT a limit: does the ROW ORDER of a
      multi-row constant `GROUP BY` result vary across fresh connections /
      builds? If yes, that is a separate hole in every unordered multi-row
      constant result -- ticket it, do not widen this fix silently
- [ ] #4 seed 1784 classifies clean (REFUSED) in the campaign, and the
      fuzzer's hostile-clause arm counts the refusal as the expected outcome
- [ ] #5 known-limitations.md gains the rule under the constant-emitter
      section, with the twelve-connection measurement as its ground
<!-- AC:END -->
