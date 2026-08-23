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
- [ ] #2 with `ORDER BY` it still serves -- and MEASURE the ties case first:
      `ORDER BY v LIMIT 1` over duplicate `v` values, many fresh
      connections. If DuckDB is unstable there too, tighten the rule to
      refuse unless the order is provably total (e.g. ORDER BY the full
      GROUP BY key), and record the measurement either way
- [ ] #3 probe the disease WITHOUT a limit: does the ROW ORDER of a
      multi-row constant `GROUP BY` result vary across fresh connections /
      builds? If yes, that is a separate hole in every unordered multi-row
      constant result -- ticket it, do not widen this fix silently
- [ ] #4 seed 1784 classifies clean (REFUSED) in the campaign, and the
      fuzzer's hostile-clause arm counts the refusal as the expected outcome
- [ ] #5 known-limitations.md gains the rule under the constant-emitter
      section, with the twelve-connection measurement as its ground
<!-- AC:END -->
