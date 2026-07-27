---
id: TASK-55
title: >-
  Specializer small-tails sweep — NULL-value statics + schema-qualified
  relations
status: Done
assignee: []
created_date: '2026-07-27 22:10'
updated_date: '2026-07-27 22:23'
labels: []
milestone: m-7
dependencies:
  - TASK-53
priority: medium
type: feature
ordinal: 49000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-wave-B census tails that are conservative rejects rather than semantic gaps:

1. NULL values in static join tables (9 cases): the original static-map design rejected any NULL in a value column. Real fitted-encoding tables have NULLs. Design: split each NULLABLE value column into (validity i1, payload) pairs at materialization — zero IR changes; the frontend's static_lane combines validity into the null lane (AND with the LEFT-miss flag). NULL KEYS keep the existing drop-the-row rule (never equi-match).

2. Schema-qualified relations (5 cases): SELECT test.tbl.col FROM test.tbl / FROM s1.t1, s2.t1. The engine is schema-less; accept a single qualifier when the table part matches the registered bare name (driving, joined, and comma-joined statics — registered names may themselves be qualified), and bind 3-part column refs (schema.table.col). This AMENDS the wave-5 main.-only pin: DuckDB's schema-existence errors are unknowable to a schema-less registry; document as a contract choice in known-limitations.md + twin test updates in the same commit.

Deferred with named rejects: COLUMNS(* REPLACE ...) and fn(COLUMNS(*)) expression forms, try_trim_null (a corpus-local macro), UBIGINT key payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Static tables with NULL VALUE columns serve: NULL flows through join lanes (INNER and LEFT, incl. residuals); NULL keys keep dropping; both backends
- [x] #2 Schema-qualified driving/joined/comma-joined relations + 3-part column refs bind by suffix match; known-limitations.md §5 + twin suite updated in the same commit
- [x] #3 Corpus replay: zero FAILs, match count reported (expect ~+12 to ~517)
- [x] #4 Gate green both backends, clippy clean
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Small-tails sweep shipped: corpus 505 -> 511 of 678, zero FAILs. (1) NULL values in static join tables serve — declared-nullable value columns flatten to (validity i1, payload) pairs in the map layout with zero IR changes; the probe's typed miss-defaults make validity=false free on LEFT misses, and StaticCol ANDs validity into the null lane on both backends (INNER/LEFT/residual all oracle-checked); NULL keys keep the drop rule; the materializer maps NULL -> (false, typed default) with the non-nullable guard kept as a safety net. (2) Schema qualifiers are registry-noise: s1.t1 resolves by table-part suffix match for driving/joined/comma-joined relations, 3-part schema.table.col refs bind; amends the wave-5 main.-only rule, documented as a §5 contract choice (DuckDB's schema-existence errors are unknowable to a schema-less registry); ambiguity still errors. known-limitations.md + twin suite updated in the same commit (NULL-value reject row flipped to served). 3 of the 9 NULL-static corpus cases uncovered second blockers (dup keys / self-joins — stage-B constituency grows again). Gates: 155 Rust + 599 py green, interp identical minus backend-identity guards, clippy clean on touched files. PR #42 (stacked on #41).
<!-- SECTION:FINAL_SUMMARY:END -->
