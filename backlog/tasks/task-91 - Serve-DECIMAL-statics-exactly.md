---
id: TASK-91
title: >-
  Serve DECIMAL statics exactly (m-8 Dec lane, first slice)
status: To Do
assignee: []
created_date: '2026-08-13 01:45'
labels:
  - m-8
  - parity
dependencies: []
documentation:
  - docs/superpowers/specs/2026-08-11-duckdb-type-lattice-design.md
type: feature
ordinal: 83000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fitted params tables carry `sum(BIGINT)` as decimal128(38,0) — the m-8
phase-0 measurement's own finding, so this is the epic's first REAL decimal
demand path. Today the engine refuses an inexact decimal static by name
(interim, decided 2026-08-13: better a named no than the silent off-by-one
it replaced, and no half-implementation of serving in the meantime).

This task lands exact serving whole: decimal static payloads stored as
scaled i128 through ingest and the join path, typed `Dec(p,s)` in the
frontend, emitted as decimal128(p,s) at the boundary. Pure store-and-serve —
no decimal arithmetic; that stays with the later Dec phases.

The forward edge is pinned as an xfail-strict test
(`test_an_inexact_decimal_static_serves_exactly`): it asserts exact serving
and flips loudly the moment this task lands, forcing the interim refusal
and its wording out in the same PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A decimal128 static column serves bit-exactly (2^53+1 comes back as
      itself), schema decimal128(p,s) matching DuckDB
- [ ] #2 The interim refusal and its message are REMOVED in the same PR
- [ ] #3 The xfail pin flips to a green parity test in the same PR
<!-- AC:END -->
