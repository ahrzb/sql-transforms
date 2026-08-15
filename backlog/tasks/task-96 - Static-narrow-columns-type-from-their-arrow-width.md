---
id: TASK-96
title: >-
  Static narrow columns type from their arrow width
status: Done
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
- [x] #1 a static int8/int16/int32 column's references type at that width; join payloads emit it
- [x] #2 the fuzzer generates narrow static columns and the campaign stays clean
<!-- AC:END -->

## Notes

2026-08-13 scope update: the ROW-side half shipped in PR #144 (arrow schema API) — pa.Schema row columns bind at their declared width, probed against DuckDB through joins/CASE///%/overflow with zero divergence. Remaining scope is the STATIC-side collapse only (schema.rs arrow_type_to_base -> Base::Int and duckdb/mod.rs base_to_ty -> Ty::I64), plus the key_lane f64 probe-key question from the RFC.

2026-08-15 done. The catalogue now reads the SAME walker the row path uses
(`schema::arrow_static_schema`), so a static column types at its declared
arrow width. `base_to_ty` and the whole `Base` prefix-matching arrow parser
(`from_arrow_table`, `arrow_schema_to_ordered_fields`,
`arrow_field_to_field_type`, `arrow_pytype_to_base`, `arrow_type_to_base`)
were its only callers and are deleted — 119 lines, and with them the second
source of truth that made this defect possible.

The two paths still accept DIFFERENT type sets and that is now explicit
(`schema::Policy`) rather than accidental: statics additionally take
unsigned ints into the i64 lane, and float32/decimal128 into the f64 lane.
Narrowing statics to the row policy would break builds that work today, so
it stays until decided on purpose. Caught by two tests (an exact decimal
static, a UBIGINT key payload) when the first attempt unified the policies
as well as the parsers.

NOT in this ticket, found while verifying it: projecting a static column
through a LEFT JOIN types the OUTPUT from the join key's width rather than
the column's own. `SELECT s0.c0 FROM __THIS__ LEFT JOIN s0 ON (c2 = s0.c0)`
with `s0.c0` int64 and `c2` int8 emits int8; DuckDB emits int64. Pre-exists
this change (same count before and after, fuzz seed 1379) and is a distinct
defect — key promotion, not catalogue width.
