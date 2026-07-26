# Wave-1 builtin & predicate pins — DuckDB 1.5.5, measured 2026-07-26

The implementation contract for TASK-47. Every claim below was MEASURED through
the vectorized path (table columns, not literals) by a six-family fleet; the
full pin tables (300+ rows incl. exact reprs, result types, and error heads)
are the JSON files in `pins-wave1/`. Where the constant-fold path diverges it
is noted. Nothing here is inferred from documentation.

## Decisions (implementation-shaping)

### Frontend desugars — NO new IR (proven exact against the pins)
- `x BETWEEN lo AND hi` → `(x >= lo) AND (x <= hi)`; `NOT BETWEEN` → `NOT(...)`.
  K3 (Kleene) AND over `duck_fcmp` reproduces every pinned row, including
  `(5, NULL, 4) → FALSE` (NULL AND FALSE) and NaN-above-inf ordering. No
  empty-range special case — and none may be added (short-circuiting the
  NULL/FALSE interplay diverges).
- `x IN (a, b, …)` → K3 OR-chain of equalities; `NOT IN` → `NOT(...)`.
  Truth table pinned: `1 IN (1,NULL)`=TRUE, `3 IN (1,NULL)`=NULL. Type
  unification is WHOLE-EXPRESSION (one common type across x and every
  element; a single DOUBLE drags the list to DOUBLE) — unify int→f64 like
  our cmp coercion; VARCHAR↔numeric mixing has exec-time cast semantics we
  do not model → clean-unsupported. `x IN ()` is a parse error upstream.
- `power(x,y)`, `x ^ y`, `x ** y` → one `pow` op. DuckDB `^` IS pow (not
  xor); unary minus binds TIGHTER (`-2^2 = +4.0`); `^`/`**` are
  LEFT-associative (`2^3^2 = 64`). Verify sqlparser reproduces this
  precedence/associativity — if not, fix at the frontend.
- `pi()` → fold to the f64 literal 0x400921FB54442D18.
- Aliases → one op each: instr = strpos = 2-arg "position" (haystack,
  needle); the SQL form `position(n IN s)` is needle-first; prefix =
  starts_with, suffix = ends_with; len = char_length = character_length =
  length; `^@` operator = starts_with.
- `least/greatest(n-ary)` → left fold of BINARY Least/Greatest IR ops
  (first-arg-wins on ties makes the left fold exact).

### New IR ops
Math unary (f64→f64, int inputs cast to f64 first; VARCHAR/BOOLEAN columns
are binder errors — no implicit cast):
- `Ln, Log2, Log10`: TRAP on x ≤ 0 — "cannot take logarithm of zero" /
  "cannot take logarithm of a negative number" (exact heads in pins);
  -0.0 hits the ZERO check first. NaN (either sign) passes THROUGH to
  libm-NaN (comparison-based guard, not is_finite). NULL pre-empts every
  domain check. Otherwise bit-exact platform libm.
- `Exp`: TOTAL — never errors. exp(1000)=inf, exp(-inf)=+0.0 (positive
  zero), underflow passes through denormals (exp(-745)=5e-324).
- `Sqrt`: TRAP on any negative incl. -inf ("cannot take square root of a
  negative number"); sqrt(-0.0) = -0.0 (not a trap — IEEE).
- `Cbrt`: TOTAL; cbrt(-8) = -2.0 exactly (NOT pow(x,1/3) which is NaN).
  CI-discovered addendum: DuckDB's own wheels disagree with each other on
  cbrt by one ulp across platforms (Windows wheel == Rust/ucrt bit-exact;
  Linux wheel's bundled std::cbrt returns e.g. 3.0000000000000004 for
  cbrt(27)). The engine stays deterministic (Rust cbrt); oracle parity for
  cbrt is pinned to <= 1 ulp, not repr-exact — the only such exception.
- `SinF64/CosF64/TanF64`: TRAP on ±inf ("Out of Range Error: input value
  inf is out of range for numeric function"); NaN passes through
  BIT-EXACTLY (payload+sign preserved — check is_nan() BEFORE calling
  libm); finite inputs bit-match platform libm incl. 1e300 (full argument
  reduction; 0/10000 fuzz mismatches). tan(pi()/2) is finite 1.633…e16.
- `FloorF64/CeilF64/TruncF64/RoundF64`: Rust `f64::floor/ceil/trunc/round`
  are bit-exact (round is half-AWAY-from-zero, incl. the
  0.49999999999999994 → 0.0 exactness — no +0.5 tricks). floor/ceil have
  NO integer overloads (int → lossy f64); single-arg trunc/round on
  integers are identity.
Math binary:
- `LogBase(b, x)`: EXACTLY log10(x)/log10(b) — NOT ln-ratio (refuted
  6989/20000) or log2-ratio. Base domain-checked BEFORE x; base==1 has a
  dedicated error (DuckDB's typo verbatim in pins); sign-of-zero leaks
  from the division (log(0.5,1.0) = -0.0) and must be preserved.
- `Pow(x, y)`: TOTAL, pure IEEE: pow(NaN,0)=1, pow(1,NaN)=1, pow(0,-1)=inf,
  negative^fractional=NaN, overflow=inf. Result always DOUBLE (int operands
  cast first — no integer pow, no overflow trap).
- `Least/Greatest` (binary): NULL-IGNORING (result NULL only if BOTH are);
  ties return the FIRST argument (pinned via -0.0 vs 0.0); NaN sorts above
  +inf under duck order (greatest(x, NaN) = NaN).
- `Round2/Trunc2 (x, n)` — the nasty one, own ops:
  - round(x, n<0): NaN/±inf → +0.0 (!); n ≤ -309 → 0.0 for every finite x.
  - trunc(x, n<0) non-finite fallback returns the INPUT (differs from round).
  - Integer round(int, n<0) WRAPS at i64 (round(i64::MAX,-2) =
    -9223372036854775700); |n| ≥ 19 collapses to 0. Never errors.
  - The f64 scale factor is DuckDB's own std::pow(10,n) which is NOT
    correctly rounded (1-ulp off strtod at n=23) — extract an
    oracle-pinned pow10 table (n ∈ [-323, 308]) with a generator script
    rather than calling any libm (same playbook as the casemap table).
  - round(DOUBLE, BIGINT-column) does not bind in DuckDB — digits must be
    INTEGER; constant integer digits (the corpus shape) bind fine.
String search (all TOTAL, all NULL-strict in every argument — validity =
AND of input validities, no three-valued special cases):
- `StrFind` (instr/strpos/position): 1-based CODEPOINT index (not bytes,
  not graphemes), 0 = not found, empty needle → 1 (matches even in '').
- `SContains`, `SStartsWith`, `SEndsWith`: empty needle → TRUE. Byte-wise
  `str` ops reproduce every pin (no normalization, no case folding).
- `SLenChars` (length: codepoints), `SLenBytes` (strlen: UTF-8 bytes).
  bit_length = 8×strlen (desugar if wanted; not in wave scope).

### Catalogue rejections (do NOT ship)
log1p/expm1 do not exist in DuckDB 1.5.5 (catalog error — scenarios use
ln(1+x) instead); `truncate` does not exist; contains(numeric,…) etc. have
no implicit numeric→VARCHAR casts (binder errors); `'2^-2'` is a bind-time
error in DuckDB (lexer eats `^-`) — mirror as clean unsupported if sqlparser
differs.

### Cross-cutting, engine-shaping
- Trap laziness is already correct by construction: WHERE filters and
  untaken CASE branches genuinely prevent evaluation in DuckDB (pinned),
  and our lowering evaluates the projection only after the filter branch;
  a domain-violating row that IS evaluated aborts the whole call — also
  matching DuckDB.
- Bare SQL float literals are DECIMAL (no signed zero; exact comparisons
  against BIGINT) — pins used `'…'::DOUBLE`/columns to dodge it; our
  frontend maps decimal literals to f64 (known, documented v0 divergence).
- Literal-vs-vectorized: zero divergences in string-search, trig, log,
  pow families; DECIMAL-literal round differs by design (we follow the
  DOUBLE column path, as with substr).
- Repo doc drift found during verification: interp.rs header still says
  fcmp is IEEE-with-NaN-false; it is duck_fcmp. Fix in this wave.
