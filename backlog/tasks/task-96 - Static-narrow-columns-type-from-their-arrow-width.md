---
id: TASK-96
title: >-
  Static narrow columns type from their arrow width
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 88000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An int32 static column still types I64 (Base::Int collapse in the catalog
path), so its join payload emits int64 where DuckDB emits int32 —
unpatrolled because gen never makes narrow statics. Map arrow dtype -> Ty
directly for the catalog; payloads stay i64-lane. Add a narrow-static gen
production. Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a static int8/int16/int32 column's references type at that width; join payloads emit it
- [ ] #2 the fuzzer generates narrow static columns and the campaign stays clean
<!-- AC:END -->

## Notes

2026-08-13 scope update: the ROW-side half shipped in PR #144 (arrow schema API) — pa.Schema row columns bind at their declared width, probed against DuckDB through joins/CASE///%/overflow with zero divergence. Remaining scope is the STATIC-side collapse only (schema.rs arrow_type_to_base -> Base::Int and duckdb/mod.rs base_to_ty -> Ty::I64), plus the key_lane f64 probe-key question from the RFC.
