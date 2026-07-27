---
id: TASK-55
title: >-
  Specializer small-tails sweep — NULL-value statics + schema-qualified
  relations
status: In Progress
assignee: []
created_date: '2026-07-27 22:10'
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
- [ ] #1 Static tables with NULL VALUE columns serve: NULL flows through join lanes (INNER and LEFT, incl. residuals); NULL keys keep dropping; both backends
- [ ] #2 Schema-qualified driving/joined/comma-joined relations + 3-part column refs bind by suffix match; known-limitations.md §5 + twin suite updated in the same commit
- [ ] #3 Corpus replay: zero FAILs, match count reported (expect ~+12 to ~517)
- [ ] #4 Gate green both backends, clippy clean
<!-- AC:END -->
