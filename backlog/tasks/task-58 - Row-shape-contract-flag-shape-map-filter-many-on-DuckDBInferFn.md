---
id: TASK-58
title: 'Row-shape contract flag: shape="map" | "filter" | "many" on DuckDBInferFn'
status: Done
assignee: []
created_date: '2026-07-28 01:50'
updated_date: '2026-08-08 03:38'
labels:
  - specializer
  - api
dependencies: []
type: feature
ordinal: 52000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-requested (2026-07-28 morning) safety fence ahead of stage-B: serving paths need a BUILD-TIME guarantee that a query is a true projection (exactly one output row per input row, out[i] <-> in[i]).

API: DuckDBInferFn(..., shape="filter") default = today's 0..1 behavior, unchanged. shape="map" = static proof of exactly-1: rejects WHERE (can drop), INNER JOIN (key miss drops), the static-only constant path (output unrelated to input rows), and every stage-B form; allows scalar exprs + LEFT JOIN (unique keys are already enforced). shape="many" = reserved now (named rejection pointing at stage-B), becomes the ONLY way multiplicity constructs build once stage-B lands.

The proof is static (query shape), not a runtime row-count assertion. Rejection messages name the blocking construct ("shape='map': WHERE clause can drop rows").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 shape param accepted with 'filter' default; invalid values are named ValueErrors
- [x] #2 shape='map' statically rejects WHERE, INNER JOIN, and constant-path queries with messages naming the construct; serves scalar projections and LEFT JOINs
- [x] #3 shape='many' is a named reserved rejection until stage-B
- [x] #4 Default behavior byte-identical to today (full gate green, corpus 529 unchanged)
- [x] #5 Tests cover accept/reject matrix; docs note in known-limitations SS2
- [x] #6 PR opened
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped some time before 2026-08-08; the status flag was never flipped. Closed
as bookkeeping, against what is in the tree today rather than a memory of the
work:

- `shape` is a three-arm enum in `packages/confit/src/duckdb/mod.rs:965`, with
  `None | Some("filter")` as the default and anything else a named
  `PyValueError` — "shape must be 'map', 'filter', or 'many', got '{other}'".
- The exactly-one-row proof is static, in `src/specializer/mod.rs:172`;
  `shape='map'` refuses by naming the blocker (`mod.rs:1159`), with the
  static-only constant path called out separately (`mod.rs:1135`).
- `tests/test_shape_contract.py` and `tests/test_infer_arrow.py`: **9 passed**.
  `tests/test_corpus_replay.py`: **1 passed** (22s).
- The docs note lives at `docs/known-limitations.md:39`.

Two ACs read stale and are ticked on intent, not letter. **#3** — stage-B has
since landed (TASK-59), so `'many'` *serves* rather than rejects; what survives
is the part that mattered, that multiplicity is an explicit opt-in arm and
never reachable by accident. **#4** — the corpus replay asserts an empty FAILED
set rather than pinning a count, so "529 unchanged" has no literal referent
today; green means zero FAILED.
<!-- SECTION:NOTES:END -->
