---
id: TASK-79
title: >-
  infer_arrow emits int64 where DuckDB emits int32 for an integer-literal expression
status: Done
assignee: []
created_date: '2026-08-08 04:00'
labels:
  - bug
  - boundary
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/
type: bug
ordinal: 72000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DuckDB types a bare integer literal `INTEGER`, so an expression whose type is
driven by one comes back `int32` in its Arrow output. The engine has no narrow
integer widths, so ours is `int64`. Values agree exactly; the schemas do not.

```text
SELECT CASE WHEN k > 1 THEN 1 ELSE 0 END AS c FROM __THIS__
  duckdb  c: int32
  ours    c: int64
  pa.concat_tables([duck_out, ours]) -> ArrowInvalid
```

Same consequence as TASK-72 (a pinned-schema writer and `concat_tables` both
reject it) but a different type and a different cause, which is why it was not
folded into that fix.

This is the Arrow-visible face of the documented "narrow integer widths don't
exist" limitation in `packages/confit/docs/known-limitations.md`, which until now was only ever
discussed as an arithmetic concern.

**Not hypothetical SQL.** It hits the `titanic` serving scenario, whose
`multi_cabin` column is exactly `CASE WHEN .. THEN 1 ELSE 0 END`.

Found 2026-08-08 while fixing TASK-72, by widening that ticket's scenario
sweep from "the string column" to the whole output schema. Reproduced by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The output `pa.Table` schema equals `duckdb.execute(SQL).arrow()`'s
      for integer columns, or the divergence is a stated and tested part of the
      contract with a documented cast-before-stacking recipe
- [ ] #2 `pa.concat_tables([duck, ours])` succeeds for the `titanic` scenario
- [ ] #3 Whatever is decided, `test_output_schema_matches_duckdb_for_every_
      scenario` stops allowing the int32/int64 widening as a special case
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The engine computes in exactly four types and adding a narrow integer width to
the TYPE SYSTEM is out of proportion to this. The realistic options are output
-side only:

1. **Track a declared output width per column** and emit `int32` when DuckDB
   would, without ever computing in it — the values are already known to fit,
   since they agree. This is a lie-free narrowing at the boundary only.
2. **State the divergence** and give callers `table.cast(duck_schema)` as the
   documented recipe. Cheapest, and honest, but it means `infer_arrow` output
   is not drop-in interchangeable with DuckDB's.

Option 1 needs the frontend to carry DuckDB's inferred width alongside `Ty`,
which is a real piece of work; option 2 is a docs change plus a test.

Note the row path (`infer` / `infer_rows`) is unaffected: Python ints have no
width, so this is purely an Arrow-schema question.

Pinned xfail-strict as `test_infer_arrow_integer_width_matches_duckdb`, and
`test_output_schema_matches_duckdb_for_every_scenario` allows exactly this one
widening and nothing else, so it cannot quietly grow.

Closed by the 2026-08-13 grooming pass: option 1 landed as m-8 phase 2
(393c204 - real I8/I16/I32 frontend types, infer_arrow emits the narrow
width; struct lanes hardened in 9ea07df/3199c37). The xfail pin rang XPASS
and came off, and the `_WIDENED` bypass was deleted from the every-scenario
test in the same commit; the width catalogue lives in test_integer_widths.py.
The row-time overflow trap is m-8 phase 3, out of this ticket's scope.
<!-- SECTION:NOTES:END -->

## Notes

2026-08-13 hygiene: fixed by the phase-2 integer-width work. Probe on master 5e88d38: `SELECT 1 AS o` serves int32 via infer_arrow, matching DuckDB's INTEGER exactly.
