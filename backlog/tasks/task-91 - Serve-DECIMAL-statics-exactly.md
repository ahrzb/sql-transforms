---
id: TASK-91
title: >-
  Serve DECIMAL statics exactly (m-8 Dec lane, first slice)
status: Done
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
- [x] #1 A decimal128 static column serves bit-exactly (2^53+1 comes back as
      itself), schema decimal128(p,s) matching DuckDB
- [x] #2 The interim refusal and its message are REMOVED in the same PR
- [x] #3 The xfail pin flips to a green parity test in the same PR
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec (docs/superpowers/specs/2026-08-25-task-91-design.md).
Decimal static payloads are scaled i128 lanes, typed Dec(p,s) in the
frontend, emitted decimal128(p,s) at the boundary over all four DuckDB
storage tiers -- pure store-and-serve, no decimal arithmetic (everything
else over a Dec value refuses by name, trading today's silent f64 wrong
answers for named refusals). Join keys need no Dec key variant: the key
lane stays the probe expression's type and the static side converts at
materialize time, exact-integer-or-drop for an integer probe and
DuckDB's own div/mod TryCastDecimalToFloatingPoint algorithm for a
double probe (mutation-proven against the naive divide). The cranelift
i128 capability probe (select + block params) was written first and
passed, so codegen keeps decimal programs.

Downstream-visible flip, mandated by the bit-exact criterion: an
f64-exact decimal static that used to come back as a Python float /
double now comes back as decimal.Decimal / decimal128(p,s), matching
DuckDB. The 2^63+1 fitted-params sum(BIGINT) case is pinned.

Gate: full root suite release AND debug (3056 passed, 5 xfailed -- the
decimal pin is deleted, flipped green), cargo 5 known pre-existing
failures only, 4200-seed campaign with gen.py now emitting decimal
statics (~18% of seeds): zero decimal-static findings, both
decimals-tagged survivors trace to the documented literal narrowing.
<!-- SECTION:NOTES:END -->
