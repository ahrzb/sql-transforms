---
id: TASK-80
title: >-
  Negative zero loses its sign in constant folding and unary minus
status: To Do
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/fuzz/README-in-module-docstrings
type: bug
ordinal: 73000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two mechanisms, one observable: the sign bit of `-0.0` is gone by the time a
value leaves the engine, on BOTH backends.

```text
SELECT -0.0e0 AS o0 FROM __THIS__          -- constant fold
  duckdb  -0.0        ours  0.0

SELECT (- x) AS v FROM __THIS__  -- x = 0.0 at runtime, no folding involved
  duckdb  -0.0        ours  0.0
```

The runtime case means unary minus is compiled as `0 - x` (IEEE: `0.0 - 0.0 =
+0.0`) rather than a sign-bit flip (`fneg`). The folded case drops the sign in
the frontend before either backend runs.

Not cosmetic: the sign becomes any-magnitude wrong through division —

```text
SELECT (0.1e0 / 1.5e0) / (1.0e0 * -0.0e0) AS o0
  duckdb  -inf        ours  +inf
```

and is string-visible through `CAST(.. AS VARCHAR)` ('-0.0' vs '0.0').

Found by the differential fuzzer 2026-08-11: 113 of 963 findings in the 20k
campaign reduce to this one root cause (seeds 34, 1018, 3505, 4087, 4341,
8720, 9410 among them). Reproduced by hand on both backends; shrunk repro is
the one-liner above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `SELECT -0.0e0` serves bit-identical to DuckDB (sign bit included)
- [ ] #2 Runtime `(- x)` with `x = 0.0` yields `-0.0` on both backends
- [ ] #3 The derived cases (division to ±inf, VARCHAR cast) agree with DuckDB
- [ ] #4 A fuzz re-run over the campaign's diverging seeds shows the
      `DIVERGE_VALUE values` class empty of -0.0 cases
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two sites: the frontend's constant folder (fold of the `-` unary and of
literal `-0.0` itself) and the lowering of unary minus (needs an `fneg`, not
`sub(0, x)` — cranelift has `fneg`; the interpreter needs the same). Keep the
integer path as-is: `-0` has no sign bit in i64.
<!-- SECTION:NOTES:END -->
