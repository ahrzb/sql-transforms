---
id: TASK-58
title: 'Row-shape contract flag: shape="map" | "filter" | "many" on DuckDBInferFn'
status: In Progress
assignee: []
created_date: '2026-07-28 01:50'
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
- [ ] #1 shape param accepted with 'filter' default; invalid values are named ValueErrors
- [ ] #2 shape='map' statically rejects WHERE, INNER JOIN, and constant-path queries with messages naming the construct; serves scalar projections and LEFT JOINs
- [ ] #3 shape='many' is a named reserved rejection until stage-B
- [ ] #4 Default behavior byte-identical to today (full gate green, corpus 529 unchanged)
- [ ] #5 Tests cover accept/reject matrix; docs note in known-limitations SS2
- [ ] #6 PR opened
<!-- AC:END -->
