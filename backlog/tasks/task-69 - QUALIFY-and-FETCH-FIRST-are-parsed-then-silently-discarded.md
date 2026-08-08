---
id: TASK-69
title: >-
  QUALIFY and FETCH FIRST are parsed then silently discarded
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - frontend
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
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
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 QUALIFY either builds and matches DuckDB, or is refused by name
- [ ] #2 FETCH FIRST is treated exactly as LIMIT is
- [ ] #3 Neither can be certified by `shape='map'` while dropping rows
- [ ] #4 A test sweeps the sqlparser `Select`/`Query` fields for any OTHER
      clause that is parsed and ignored - this is a class, not two instances
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The refusal already exists; these two spellings simply are not in it. Grep the
frontend for every `query.`/`select.` field sqlparser exposes and check each is
either handled or refused - the same audit would have caught both.
<!-- SECTION:NOTES:END -->
