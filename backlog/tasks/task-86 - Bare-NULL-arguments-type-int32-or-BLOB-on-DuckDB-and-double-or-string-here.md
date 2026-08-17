---
id: TASK-86
title: >-
  Bare NULL arguments type int32 or BLOB on DuckDB and double or string here
status: Done
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - boundary
  - parity
  - fuzz
dependencies: []
documentation:
  - docs/known-limitations.md
type: bug
ordinal: 79000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The engine refuses SOME bare NULLs ("bare NULL literal without a ..."), but
where a function parameter gives it a type to adopt, it adopts OURS — while
DuckDB types the bare NULL first (INTEGER) and lets IT drive the signature:

```text
SELECT nullif(NULL, 84.754e0) AS o0 FROM __THIS__
  duckdb  o0: int32   (nullif's output takes the FIRST argument's type)
  ours    o0: double

SELECT repeat(NULL, 3) AS o0 FROM __THIS__
  duckdb  o0: binary  (BLOB -- repeat(BLOB, n) wins overload for NULL)
  ours    o0: string
```

Values agree (all NULL); schemas do not, so `pa.concat_tables` against the
oracle raises — the TASK-72/79 consequence through a different door.

Found by the fuzzer 2026-08-11: 11 `DIVERGE_VALUE schema` findings (seeds
1184, 8363, 9443, 10845). Both shapes reproduced by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A bare NULL argument either types the way DuckDB types it (schema
      -equal output) or refuses by name — extending the existing bare-NULL
      refusal to these positions is an acceptable answer
- [ ] #2 The choice is recorded in docs/known-limitations.md next to the
      existing bare-NULL rule
- [ ] #3 Pins for the nullif and repeat shapes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refusal is the cheap honest answer and matches the engine's existing bare-
NULL stance; matching DuckDB's typing drags in int32 (TASK-79's width work)
and BLOB (a fifth type). Recommend: refuse bare NULL wherever its adopted
type would differ from DuckDB's inference — which in practice means refusing
bare NULL as a direct argument of any function unless cast (`CAST(NULL AS
DOUBLE)` already works and is the documented spelling).

Closed by the 2026-08-13 grooming pass, in two waves: 7f5acdb refuses a
divergently-typed bare NULL argument by name (recorded in known-limitations
by aab2995), then e1c62ae upgraded nullif to real DuckDB-matching int32
typing once the TASK-79 width work landed - schema-parity-pinned in
test_integer_widths.py::test_output_width_matches_duckdb. repeat's BLOB face
stays a documented refusal (known-limitations line on the residual). Pins
for both shapes per AC #3 live in test_literal_typing.py.
<!-- SECTION:NOTES:END -->
