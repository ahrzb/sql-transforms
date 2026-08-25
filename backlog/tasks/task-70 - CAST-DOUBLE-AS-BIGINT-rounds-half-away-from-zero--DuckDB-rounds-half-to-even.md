---
id: TASK-70
title: >-
  CAST DOUBLE AS BIGINT rounds half away from zero; DuckDB rounds half to even
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/test_cast_semantics.py
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
`packages/confit/tests/known_divergences/test_cast_semantics.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 CAST(DOUBLE AS BIGINT) matches DuckDB on every half-integer, both
      backends and the TRY_CAST range-guarded path
- [x] #2 The `round()` BUILTIN keeps half-away-from-zero - its wave-1 pin must
      not regress
- [x] #3 The misleading "ftoi.round matches DuckDB CAST rounding" comment goes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Needs a distinct IR rounding mode for the cast, or `RoundMode::Round` becomes
half-to-even and the `round()` builtin moves to its own mode. Cranelift's
`nearest` instruction is already half-to-even, so the JIT side may be a
one-opcode change; the interpreter needs `f64::round_ties_even()`.
<!-- SECTION:NOTES:END -->

## Resolution (2026-08-08)

`RoundMode::Round` is now `RoundMode::Nearest`, half-to-EVEN on both backends
(`f64::round_ties_even()` in the interpreter, the same flag through `h_ftoi`
in the JIT). The IR text opcode is `ftoi.nearest`. No new mode was needed:
CAST and TRY_CAST were the only two emitters, and the `round()` builtin lowers
through `SKind::Round`, a different path entirely.

### The measurement that matters, because two existing pins got it wrong

```text
CAST(DOUBLE AS BIGINT)   half to even        -2.5 -> -2
CAST(DECIMAL AS BIGINT)  half away from zero -2.5 -> -3
round(DOUBLE)            half away from zero -2.5 -> -3.0
```

`specializer::tests::cast_matrix` and the CASTS fixture both asserted
half-away-from-zero for a DOUBLE cast and had to be corrected. Both were
written from a DuckDB query on a bare `-2.5` literal, which DuckDB types
`DECIMAL(2,1)` — so they measured the decimal cast and pinned its answer onto
the double one. **Measure a DOUBLE cast with a DOUBLE column or an explicit
`::DOUBLE`, never a bare literal.** Recorded in `packages/confit/docs/known-limitations.md`
next to the decimal-literals-are-f64 divergence, which is what makes the two
reachable from the same SQL.

The CASTS fixture's static moved 2.5 -> 3.5 as well: with half-to-even,
`nearest(2.5) == trunc(2.5)`, and the fixture would no longer have noticed the
two opcodes collapsing into one.

AC #1 covers TRY_CAST (a second `Ftoi` site, inside the range guard) and both
backends; AC #2 is held by the existing DuckDB-differential `round()` pins in
`test_duckdb_interpreter.py`, plus a same-query contrast in the new test that
fails if the cast ever adopts `round()`'s mode.
