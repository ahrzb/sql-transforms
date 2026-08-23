---
id: TASK-129
title: >-
  An unordered multi-row constant result freezes an accidental row order
status: To Do
assignee: []
created_date: '2026-08-19 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 114000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-128's AC #3 probe, measured 2026-08-19. The row-limit disease exists
WITHOUT a limit: the row ORDER of an unordered multi-row `GROUP BY` result is
not a function of the query either.

```
SELECT g, sum(v) FROM s GROUP BY g      -- 200 groups
raw DuckDB, rows INSERTed one by one:   12 distinct row orders / 12 fresh connections
                                        (int keys and varchar keys both)
our constant path (arrow materialize):  1 distinct row order / 6 builds
```

Raw DuckDB is fully unstable. Our constant path LOOKED stable across six
builds -- but that is an artifact of how `eval_static_only` materializes
(one bulk `CREATE TABLE AS SELECT` from a registered arrow table, one
contiguous scan), not a guarantee anyone has made. A DuckDB version bump, a
parallelism change, or a bigger static could flip it, and then two builds of
the same fn disagree on row order -- the exact TASK-128 failure, one clause
earlier.

Why the campaign never sees it: the fuzzer's generated statics are 0-6 rows,
few groups, where hash order has little room to vary. The class is
under-tested, not absent.

Options, deliberately NOT decided here (AC #3 said ticket, do not widen
TASK-128's fix silently):

  (a) sort the frozen rows at build (by all columns, NULLS consistent) --
      deterministic artifact, but its order then deliberately differs from
      any single DuckDB run's
  (b) require ORDER BY on a multi-row constant result, refuse without --
      mirrors TASK-128's shape; ORDER BY ties would need the same scrutiny
  (c) declare constant row order unspecified in the contract doc and pin
      only multiset equality -- cheapest, weakest

Needs AmirHossein's pick before any code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 AmirHossein picks (a), (b) or (c); the choice and its ground land
      in known-limitations.md
- [ ] #2 whichever pick: two builds of the same fn over the same statics
      produce identical `infer_rows([])` output, or the contract doc says
      in so many words that row order is unspecified
- [ ] #3 the fuzzer generates a static with enough groups (>= 50) for hash
      order to actually vary, so this class is reachable
- [ ] #4 the ties question from TASK-128 AC #2 inherits: if (b), an ORDER
      BY that does not totally order the result gets the same treatment
<!-- AC:END -->
