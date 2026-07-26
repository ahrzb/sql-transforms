# Stretch 4: builtin catalogue — measured DuckDB 1.5.5 pins + implementation spec

All values below were MEASURED against duckdb-python 1.5.5 (workflow fan-out of 8
pin agents + local micro-pins, 2026-07-26). DuckDB is the oracle; when in doubt,
re-measure, never assume.

## Divergences found in already-landed code (fix in this stretch, with pin tests)

1. **`%` by zero**: `5 % 0` → NULL (NOT an error). `MIN % -1` → traps
   ("Out of Range Error: Overflow in division"). Our irem traps on both; keep the
   IR inst trapping (MIN % -1 stays a trap ✓) and guard the SQL lowering:
   `a % b` → `CASE WHEN b = 0 THEN NULL ELSE a irem b END` (skip the guard when b
   is a non-zero literal). Result becomes nullable when guarded.
2. **Float `%`**: currently rejected at bind. DuckDB: `5.0 % 2.0` = 1.0,
   `-5.5 % 2.5` = -0.5 (sign of dividend), `5.0 % 0.0` = NaN — exactly Rust's
   `%` on f64. Add `BinOp::Frem` (never traps), un-reject at bind. No guard.
3. **F64 comparison order** (interp.rs Cmp F64 + fold.rs cmp mirror are IEEE —
   wrong): DuckDB order = IEEE except NaN: `nan = nan` TRUE, `nan > 1` TRUE,
   `nan > inf` TRUE, `nan <= nan` TRUE, `nan <> nan` FALSE, `1 <= nan` TRUE;
   `-0.0 = 0.0` TRUE, `-0.0 < 0.0` FALSE. Implementation:
   both NaN → Equal; one NaN → NaN is Greater; else IEEE partial_cmp.

## New IR instructions (each lands in ir/mod + verify + print + parse + gen + interp)

- `supper` / `slower` (Str→Str): DuckDB uses SIMPLE (1:1) case mapping:
  `upper('ß')` = 'ẞ' U+1E9E (NOT 'SS'), `lower('İ' U+0130)` = plain 'i' (dot
  dropped), emoji pass through. Rust std is FULL mapping → v0: per char, use
  to_uppercase()/to_lowercase() iff it yields exactly 1 char, else keep the char
  unchanged. KNOWN divergence on ß (we keep 'ß', DuckDB 'ẞ') and İ→lower (we keep
  'İ', DuckDB 'i'): xfail-strict differential case + ticket note. ASCII exact.
- `strim.both|lead|trail s, chars` (Str,Str→Str): removes chars in the SET
  `chars` from the side(s). 1-arg SQL trim = chars " " (ONLY space 0x20 — tab/
  newline NOT trimmed). Empty set = no-op. NULL either arg → NULL (flag algebra
  at lowering, inst itself total).
- `ssubstr s, start, len` (Str,I64,I64→Str): codepoint-based (NOT grapheme —
  slices inside ZWJ emoji). Algorithm (1-based virtual window):
  if len < 0 → ""; if start < 0 → start = char_len + start + 1;
  window [start, start.saturating_add(len)); intersect with [1, char_len];
  missing SQL len → Lit(i64::MAX) (saturating add makes it "rest of string").
  Pins: substr('hello',0)='hello', (0,3)='he', (-2)='lo', (-6,3)='he',
  (-10,8)='hel', (1,0)='', (1,-1)='', (10)=''.
- `iabs` (I64→I64): traps on i64::MIN ("Out of Range"-style). `fabs` (F64→F64):
  Rust f64::abs — clears sign bit, abs(-0.0)=+0.0, abs(nan)=nan, abs(-inf)=inf.
- `fround` (F64→F64): Rust f64::round = half AWAY from zero ✓ (2.5→3, -2.5→-3),
  nan→nan, inf→inf, 1e300→1e300, -0.4→-0.0 (sign KEPT on double). round(int) is
  identity at the FRONTEND (no inst); round(x, digits) → clean unsupported
  (scale-then-round algorithm deferred).
- `frem` BinOp (F64,F64→F64): Rust `%`. Never traps.

## Frontend dispatch (new SKind variants + lowerings)

- `Expr::Trim` (sqlparser: trim_where BOTH/LEADING/TRAILING, trim_what,
  trim_characters) + `ltrim`/`rtrim`/2-arg functions → SKind::Trim{side}.
  All forms NULL-propagating on both args.
- `Expr::Substring` (SUBSTR and SUBSTRING both route here) → SKind::Substr.
  NULL-propagating on all three args.
- `Expr::Function` name dispatch (case-insensitive):
  - upper/lower → SKind::StrCase{upper} — arg must be Str (DuckDB: upper(123)
    is a Binder Error, NO numeric coercion).
  - abs → SKind::Abs; I64 or F64 only (abs('5')/abs(true) = binder errors).
    Result type = arg type.
  - round (1-arg) → identity for I64; SKind::Round for F64. 2-arg → unsupported.
  - concat(a,...) → NULL-SKIPPING: each nullable arg wraps in
    CASE WHEN x IS NULL THEN '' ELSE cast_to_varchar(x) END, chain SKind::Concat.
    Result never NULL. concat() zero args → Bind error. Non-string args render
    via VARCHAR cast (1→'1', 1.5→'1.5', true→'true').
  - coalesce(a,...) → nested CASE WHEN a IS NOT NULL THEN a ELSE rest END.
    Lazy per-row (pinned: untaken erroring arm does NOT fire). Args unify under
    existing promotion (I64+F64→F64); else Bind error. coalesce() → parser
    error in DuckDB; here Bind error fine.
  - nullif(a,b) → CASE WHEN a = b THEN NULL ELSE a END; comparison at PROMOTED
    type, result type = FIRST arg's type (never unified!). nullif(nan,nan) →
    NULL (needs divergence fix #3).
- `BinaryOperator::StringConcat` (`||`): ALWAYS string concat in DuckDB — even
  `1 || 2` = '12', `true || true` = 'truetrue'. NULL-PROPAGATING (unlike
  CONCAT). Cast non-Str operands to VARCHAR: bool → 'true'/'false' (measured).
- CONCAT vs ||: CONCAT skips NULLs (all-NULL → ''), || propagates.

## fold.rs

New kinds: fold children only (like Case). Update the f64 cmp mirror to the
DuckDB order (divergence #3) — fold and interp MUST stay bit-identical.

## Out of scope (clean unsupported), deliberate

round(x, digits); DECIMAL anything (literals stay F64 — known v0 ceiling);
upper/lower non-simple-map codepoints byte-exactness (xfail + ticket).

## Adversarial-fleet addendum (6 probe agents, ~1,400 probes, 2026-07-26)

Divergences FOUND and FIXED (each now pinned in Rust + differentially):

1. **NULL divisor `%`**: the `b = 0` CASE guard alone is NULL for b NULL —
   fell through to irem on the garbage zero payload. `b IS NULL OR b = 0`
   shields it (TRUE OR NULL = TRUE).
2. **Trap-under-false-flag class bug** (predates stretch 4): computed
   garbage payloads are unbounded (`(x + MAX) + MAX` with x NULL overflows
   its payload lane). FIX: `FB::masked` forces nullable payloads to the type
   default before every trapping instruction (integer arith, iabs, ssubstr
   positions, ftoi cast input).
3. **1-arg trim set**: exactly the Unicode Zs space separators (per-codepoint
   census) — NBSP/ideographic space etc. trim; tab/newline/ZWSP/BOM do not.
4. **substr**: DuckDB's constant-fold path and vectorized path DISAGREE on
   negative starts. We implement the VECTORIZED path (what columns, real
   queries, and the mined corpus use): negative start clamps to 1 after
   end-resolution (`rs = max(n+start+1, 1)`), start 0 stays virtual, a
   NEGATIVE length slices BACKWARDS `[rs+len, rs)`, offsets/lengths outside
   ±2^32 trap ("Out of Range"), the 2-arg form never length-traps (why
   `len` is `Option`, not a sentinel). Known residual: pure-literal
   negative-start substr goes through DuckDB's constant path and can differ.
5. **Float -> VARCHAR**: DuckDB writes an explicit exponent sign with at
   least two digits (`1e+300`, `1e-05`) and lowercase `nan` (DuckF64 in
   interp.rs).
6. **Oracle artifact, harness-fixed, no engine change**: duckdb-python
   pushes constant filters into REGISTERED-ARROW scans with IEEE NaN
   semantics, disagreeing with its own native-table order (and violating
   3VL for `x <= NaN`). duck_check now materializes native tables.
7. **Simple-case-map divergence extended**: `upper('ᾀ')` (ypogegrammeni
   titlecase U+1F88) joins ß/İ in the strict xfail.

Also landed with this pass (corpus-driven): implicit numeric->BOOLEAN in
conditional contexts (WHERE/AND/OR/NOT/CASE WHEN; nonzero -> true incl.
NaN, NULL -> NULL); `rowid` and DuckDB lateral aliases reject as clean
unsupported; corpus replay wired into pytest with the three-outcome
contract at 49 match / 629 clean-unsupported / 0 FAIL of 678.
