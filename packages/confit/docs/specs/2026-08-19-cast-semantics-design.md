# Cast semantics: DOUBLE->narrow and VARCHAR->integer, measured (TASK-99/113)

## 1. DOUBLE -> narrow integers (items b+c)

Released DuckDB (v1.5.5, the pinned oracle) checks the RAW double against
`[MIN, MAX+1)` BEFORE rounding (`numeric_cast.hpp`,
`TryCastWithOverflowCheckFloat`), then `std::nearbyint`, then a
`static_cast` that is UB out of range. Consequences, measured and traced to
source by an Opus 5 agent (2026-08-19):

    CAST(127.5::DOUBLE AS TINYINT)  -> -128 on x86 (WRAP; aarch64 would
                                       saturate at INTEGER width)
    CAST(-128.4999 AS TINYINT)      -> Conversion Error (rounds to -128!)

Upstream FIXED it on main -- PR duckdb/duckdb#24393, merged 2026-08-03,
four days after v1.5.5 shipped: round first, check the rounded value. No
released DuckDB has the fix.

**Decision (AmirHossein, 2026-08-19): implement the FIXED semantics** --
round half-to-even, then check -- because the 1.5.5 behaviour differs by
CPU and so is not a function of the query (the same disqualifier that moved
the oracle off the optimizer). The engine already implemented exactly this
(verified 24/24 against upstream main's own conformance grid); the work was
pinning it: the agreed grid as passing tests, and the two half-unit slivers
as a KEPT divergence vs 1.5.5 that remeasures the live oracle and fails
loudly when the DuckDB pin advances (known_divergences/test_cast_semantics).

TRY_CAST applies the same window to the same rounded value, NULL instead of
trap. The f64->i64 guard's trap text now uses DuckDB's Conversion-Error
sentence instead of the engine-flavoured "in ftoi".

## 2. VARCHAR -> integer (item d, closes TASK-113)

One kernel, `kernels::duck_stoi`, mirrored from DuckDB's
`IntegerDecimalCastOperation` and measured (50-case grid x 3 widths x both
spellings, 228/228):

- trim ASCII whitespace; optional sign; then hex `0x`, binary `0b`, or
  decimal `digits[.frac][eE[+-]exp]`; `_` joins digits ('1_000' yes,
  '1__0'/'_1'/'1_'/'0x_1A' no); '.5', '5.', '1e+2' parse; 'e5', 'inf',
  '0o17' do not.
- rounding is HALF AWAY FROM ZERO decided on decimal DIGITS, never through
  a double: '1.4999999999999999' is 1, '155e-2' is 2. The double path
  rounds half-to-even -- DuckDB's two cast paths disagree on every exact
  half, and both are preserved.
- the WIDTH check applies to the rounded value inside the cast: CAST says
  "Conversion Error: Could not convert string to INT8" (DuckDB's spelling,
  sans the payload -- trap messages are static), TRY_CAST is NULL. The
  interpreter, cranelift's h_stoi, and the frontend's constant probe all
  call the ONE kernel, so fold and runtime cannot drift.

## 3. Retired with measurement

Item (e) -- "the optimizer pushes fitting constants through TRY_CAST" -- is
an optimizer pass. Optimizer off: `TRY_CAST(300 AS TINYINT) IS NULL` is
TRUE. Reproducing it would be an OPT_EMULATED bug now. Dead like TASK-117.

AC #2 is re-scoped rather than executed: see
packages/confit/docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md.
