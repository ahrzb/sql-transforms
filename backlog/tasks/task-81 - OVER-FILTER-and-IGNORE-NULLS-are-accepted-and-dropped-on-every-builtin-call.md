---
id: TASK-81
title: >-
  OVER, FILTER and IGNORE NULLS are accepted and silently dropped on every builtin call
status: Done
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - fuzz
dependencies: []
documentation:
  - packages/confit/src/specializer/frontend.rs
type: bug
ordinal: 74000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-69 class, still open on the builtin path: a call-node modifier that
DuckDB refuses is parsed, ignored, and the bare call served instead.

```text
SELECT abs(k) OVER () AS c FROM __THIS__
  duckdb  Catalog Error: abs is not an aggregate function
  ours    builds, serves abs(k)

SELECT rtrim(s) FILTER (WHERE TRUE) ..     duckdb: Invalid Input Error / ours: builds
SELECT ltrim(lower(s) IGNORE NULLS) ..     duckdb: Parser Error / ours: builds
```

`bind_udf_args` screens `filter` / `null_treatment` / `within_group` for
DECLARED udf calls (frontend.rs:3642), and the wide-call path screens `over`
too (frontend.rs:3863) — but the ordinary builtin path (`function()`) checks
none of them, and `f.over` is unchecked even on the udf scalar path.

The 20k fuzz campaign 2026-08-11 measured the blast radius: ~570 of 963
findings are this class, across 25+ builtins, udf calls and tree calls, in
three DuckDB error spellings (CatalogException for OVER, InvalidInput for
FILTER at execution, ParserException for IGNORE NULLS / OVER-on-constants).
Example seeds: 6570 (`rtrim(c2) OVER ()`), 4124 (IGNORE NULLS), 1415
(FILTER). Reproduced by hand 2026-08-11 before the fuzzer existed — probe in
chat — and rediscovered by it from pure random generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A scalar call carrying OVER / FILTER / IGNORE NULLS / RESPECT NULLS /
      WITHIN GROUP refuses at build with a named error, for builtins, udfs
      and tree calls alike
- [ ] #2 The refusal is a CLASS check (one screen every call path routes
      through), not a per-modifier patch — per the TASK-69 lesson
- [ ] #3 The fuzz smoke's planted-OVER assertion flips from DIVERGE_BUILD to
      REFUSED and stays green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The screen exists and is worded correctly in `bind_udf_args`; it needs to run
where `function()` dispatches builtins, and to add `f.over`. Consider
destructuring the `Function` node exhaustively (no `..`), the same move
`refuse_unhandled_select` made, so the next sqlparser field breaks the build
instead of the answers.

Closed by the 2026-08-13 grooming pass: fixed in 8943b43 - one
exhaustive-destructure screen at the top of `function()` refuses
OVER/FILTER/IGNORE-RESPECT NULLS/WITHIN GROUP on every call path (builtins,
udfs, tree calls), per the class-check AC. Pinned by
test_scalar_call_modifiers_are_refused_not_dropped plus controls, and the
fuzz smoke's planted-OVER assertion.
<!-- SECTION:NOTES:END -->
