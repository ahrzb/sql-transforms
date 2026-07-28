---
id: TASK-60
title: >-
  Pyarrow input/output: infer_arrow(pa.Table) -> pa.Table — the columnar
  boundary
status: In Progress
assignee: []
created_date: '2026-07-28 03:14'
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
Roadmap step 1 of the user-approved columnar plan (after stage B, before the columnar core): an additive fast lane that skips per-value Python objects on both boundaries. Proposal + measured motivation: docs/proposals/2026-07-28-columnar-path.md (boundary ~262ns/row + ~37ns/out-col emit is pure PyObject work; DuckDB-on-arrow comparison table included).

Scope (v1, per the proposal's copy-first recommendation): fn.infer_arrow(batch) accepts a single-chunk pa.Table or RecordBatch; columns match the model by NAME with strict types (int64/float64/utf8|large_utf8/bool + validity bitmaps); ingest walks arrow buffers via the pyarrow buffer API (address+size, bit-unpacking bool/validity) into ColData — no arrow-rs dependency; output builds pa.Array.from_buffers per column from rust-built buffers (validity bitmap, data, utf8 offsets) — one allocation per column, zero per-value boxing. Named rejections: multi-chunk input (combine_chunks() is the caller's line), missing columns, unsupported dtypes (cast first), struct/opaque row models (use infer), static-only constant queries. All shapes work; non-map shapes just return fewer/more rows (re-alignment helpers deferred until asked). Bench: extend the ingest bench with an infer_arrow row vs spec_dict and the vectorized-numpy twin question stays for the columnar-core report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 infer_arrow serves arrow-in/arrow-out for all-scalar models with values byte-identical to infer() (differential test: infer_rows == infer_arrow converted, every serving scenario)
- [ ] #2 Validity round-trips: NULLs in, NULLs out, incl. LEFT-join null-extensions under shape='many'
- [ ] #3 Named rejections: multi-chunk, missing column, wrong dtype, struct/opaque model, constant path
- [ ] #4 Measured: infer_arrow vs spec_dict vs python_dict on the serving scenarios (numbers in the PR)
- [ ] #5 Full gates green on release build; PR opened
<!-- AC:END -->
