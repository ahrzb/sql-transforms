---
id: TASK-60
title: >-
  Pyarrow input/output: infer_arrow(pa.Table) -> pa.Table — the columnar
  boundary
status: Done
assignee: []
created_date: '2026-07-28 03:14'
updated_date: '2026-08-08 03:38'
labels:
  - specializer
  - columnar
  - api
dependencies: []
type: feature
ordinal: 54000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roadmap step 1 of the user-approved columnar plan (after stage B, before the columnar core): an additive fast lane that skips per-value Python objects on both boundaries. Proposal + measured motivation: packages/confit/docs/proposals/2026-07-28-columnar-path.md (boundary ~262ns/row + ~37ns/out-col emit is pure PyObject work; DuckDB-on-arrow comparison table included).

Scope (v1, per the proposal's copy-first recommendation): fn.infer_arrow(batch) accepts a single-chunk pa.Table or RecordBatch; columns match the model by NAME with strict types (int64/float64/utf8|large_utf8/bool + validity bitmaps); ingest walks arrow buffers via the pyarrow buffer API (address+size, bit-unpacking bool/validity) into ColData — no arrow-rs dependency; output builds pa.Array.from_buffers per column from rust-built buffers (validity bitmap, data, utf8 offsets) — one allocation per column, zero per-value boxing. Named rejections: multi-chunk input (combine_chunks() is the caller's line), missing columns, unsupported dtypes (cast first), struct/opaque row models (use infer), static-only constant queries. All shapes work; non-map shapes just return fewer/more rows (re-alignment helpers deferred until asked). Bench: extend the ingest bench with an infer_arrow row vs spec_dict and the vectorized-numpy twin question stays for the columnar-core report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 infer_arrow serves arrow-in/arrow-out for all-scalar models with values byte-identical to infer() (differential test: infer_rows == infer_arrow converted, every serving scenario)
- [x] #2 Validity round-trips: NULLs in, NULLs out, incl. LEFT-join null-extensions under shape='many'
- [x] #3 Named rejections: multi-chunk, missing column, wrong dtype, struct/opaque model, constant path
- [x] #4 Measured: infer_arrow vs spec_dict vs python_dict on the serving scenarios (numbers in the PR)
- [x] #5 Full gates green on release build; PR opened
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged as PR #47; the status flag was never flipped. Closed as bookkeeping,
against the tree as it stands:

- `packages/confit/src/duckdb/arrow.rs` is the boundary, and
  `packages/confit/src/duckdb/mod.rs:1340` the binding. Ingest walks pyarrow's
  raw buffers, output builds one array per column — no `arrow-rs` dependency,
  as scoped.
- `tests/test_infer_arrow.py` (**passes**) carries each AC by name:
  `test_differential_basic_and_nulls`, `test_differential_every_serving_scenario`,
  `test_differential_shape_many_left_join` (the LEFT-join null-extension),
  `test_named_rejections`, `test_sliced_and_recordbatch_inputs`.
- The five rejections are all in the tree: missing column (`arrow.rs:82`),
  multi-chunk (`:93`), struct/opaque model (`:108`), wrong dtype (`:126`),
  constant path (`mod.rs:1351`).
- Numbers are written up in `packages/confit/docs/reports/performance-report.md` §4 rather than
  only in the PR: at n ≥ 1024 the Arrow lane beats the row path on every
  scenario (house_prices 5.31 ms → 2.80 ms per call), with the honest caveat
  that at n = 64 pyarrow's fixed ~150 µs per call exceeds the saving, so this
  is a large-batch lane and not a replacement.
<!-- SECTION:NOTES:END -->
