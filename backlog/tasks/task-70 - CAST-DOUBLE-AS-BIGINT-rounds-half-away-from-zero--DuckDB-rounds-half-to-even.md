---
id: TASK-70
title: >-
  CAST DOUBLE AS BIGINT rounds half away from zero; DuckDB rounds half to even
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 63000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`lower::cast` emits `Inst::Ftoi { mode: RoundMode::Round }` under the comment
"ftoi.round matches DuckDB CAST rounding". It does not. Both backends implement
`RoundMode::Round` as Rust `f64::round()` (half AWAY from zero - `interp.rs`
even comments `// half away from zero`), while DuckDB's DOUBLE->BIGINT cast is
half-to-EVEN.

```text
f    = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
got  = [  -4,   -3,   -2,   -1,   1,   2,   3,   4,   5]
want = [  -4,   -2,   -2,    0,   0,   2,   2,   4,   4]   (duckdb 1.5.5)
```

Every exactly-representable half-integer differs by 1, on both backends.

The confusion is genuine and worth recording: the SQL `round()` FUNCTION *is*
half-away-from-zero and is correctly pinned that way in the wave-1 builtin pins.
The CAST is a different operation that happens to share the IR mode.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 CAST(DOUBLE AS BIGINT) matches DuckDB on every half-integer, both
      backends and the TRY_CAST range-guarded path
- [ ] #2 The `round()` BUILTIN keeps half-away-from-zero - its wave-1 pin must
      not regress
- [ ] #3 The misleading "ftoi.round matches DuckDB CAST rounding" comment goes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Needs a distinct IR rounding mode for the cast, or `RoundMode::Round` becomes
half-to-even and the `round()` builtin moves to its own mode. Cranelift's
`nearest` instruction is already half-to-even, so the JIT side may be a
one-opcode change; the interpreter needs `f64::round_ties_even()`.
<!-- SECTION:NOTES:END -->
