---
id: TASK-72
title: >-
  infer_arrow emits large_string where DuckDB emits string
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - boundary
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 65000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`arrow::emit` hard-codes `pa.large_string()` for every Str output lane;
DuckDB's own `.arrow()` for the identical query returns `pa.string()` (32-bit
offsets). Values agree exactly, schemas do not.

Not cosmetic: `pa.concat_tables([duckdb_out, confit_out])` raises
`ArrowInvalid`, and so does any pinned-schema writer (Parquet, Flight, a
Delta/Iceberg sink).

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The output `pa.Table` schema equals `duckdb.execute(SQL).arrow()`'s
      for string columns, or the divergence is a stated and tested part of the
      contract
- [ ] #2 Covers scalar Str columns and the wide/struct lane path
- [ ] #3 `pa.concat_tables([duck, ours])` succeeds
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two call sites in `arrow.rs` (`emit`, scalar and wide lanes). If large_string
is kept deliberately, it needs a docs line and a schema-pinning test so callers
know to cast before stacking against DuckDB output.
<!-- SECTION:NOTES:END -->
