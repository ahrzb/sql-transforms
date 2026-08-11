---
id: TASK-82
title: >-
  Builtin argument binding accepts what DuckDB's overloads refuse (lpad with a BIGINT count)
status: To Do
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/src/specializer/frontend.rs
type: bug
ordinal: 75000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```text
SELECT lpad('', c0, CAST(c0 AS VARCHAR)) AS s FROM __THIS__   -- c0 BIGINT
  duckdb  Binder Error: No function matches lpad(STRING_LITERAL, BIGINT, VARCHAR)
  ours    builds and serves
```

DuckDB's function binder does NOT implicitly downcast BIGINT to the INTEGER
parameter of `lpad` (or `rpad`); our binder types the count as i64 and is
happy. One side builds, the other refuses — the engine serves SQL that the
oracle rejects, which the contract forbids in that direction too.

The 20k fuzz campaign 2026-08-11: 174 `No function matches` findings, 169 of
them lpad (seed 38 is the shrunk repro above; seed 1224 another). The four
singletons (levenshtein / strpos / ltrim / lower, one each) look like the
same laxity reached through NULL-typed arguments and should be triaged with
this ticket. Reproduced by hand 2026-08-11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every builtin whose DuckDB signature takes INTEGER (not BIGINT)
      either refuses a BIGINT argument by name or is proven to bind on DuckDB
      too — decided per builtin against the oracle, not assumed
- [ ] #2 The four singleton findings are each reproduced and either folded
      into this fix or ticketed on their own
- [ ] #3 An executable pin per affected builtin (the known-divergences style)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The frontend's SIGS for pad/repeat-family count parameters are the suspects.
Note DuckDB *does* downcast for some functions and not others — the oracle
decides, per name. The fuzzer's builtin-catalogue wildcard will keep finding
these; after the fix, re-run seeds 38 and 1224 and grep the campaign output
for `No function matches`.
<!-- SECTION:NOTES:END -->
