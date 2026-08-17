---
id: TASK-72
title: >-
  infer_arrow emits large_string where DuckDB emits string
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - boundary
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/test_arrow_boundary.py
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
`packages/confit/tests/known_divergences/test_arrow_boundary.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The output `pa.Table` schema equals `duckdb.execute(SQL).arrow()`'s
      for string columns, or the divergence is a stated and tested part of the
      contract
- [x] #2 Covers scalar Str columns and the wide/struct lane path
- [x] #3 `pa.concat_tables([duck, ours])` succeeds
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two call sites in `arrow.rs` (`emit`, scalar and wide lanes). If large_string
is kept deliberately, it needs a docs line and a schema-pinning test so callers
know to cast before stacking against DuckDB output.
<!-- SECTION:NOTES:END -->

## Resolution (2026-08-08): match DuckDB, 32-bit offsets

`pa.string()` on both the scalar lane and the wide/struct lane. As the notes
warned, this is not a type swap: the scalar lane builds its own offset buffer,
which becomes `Vec<i32>`.

The 2 GiB-per-batch ceiling that comes with 32-bit offsets is real and is
refused by name ("string column exceeds 2 GiB in one batch — split the
batch") rather than wrapped. DuckDB splits such a result across record
batches; we emit a single chunk, so the ceiling is ours to state.

`ingest` already accepted both `string` and `large_string`, so our own output
still feeds back in as input — tested.

### A SECOND schema divergence turned up and is NOT part of this ticket

Widening the scenario sweep from "the string column" to "the whole output
schema" caught the `titanic` scenario: `multi_cabin` is `CASE WHEN .. THEN 1
ELSE 0 END`, DuckDB types a bare integer literal `INTEGER`, and so its arrow
output is `int32` where ours is `int64`. Same consequence — `concat_tables`
raises — but a different type and a different cause: it is the arrow-visible
face of the documented "narrow integer widths don't exist" limitation.

Pinned xfail-strict in `test_arrow_boundary.py` and noted in
`docs/known-limitations.md`. It needs its own ticket. The scenario sweep
allows exactly this one widening and nothing else, so it cannot quietly grow.
