# Wave-3 builtin pins — DuckDB 1.5.5, measured 2026-07-26

The implementation contract for TASK-49. Every claim was MEASURED through the
vectorized path (table columns, not literals) by a six-family probe fleet; the
full pin tables (300+ probes with exact reprs, float bit patterns, result
types, and verbatim error heads) are the JSON files in `pins-wave3/`. Literal
(constant-fold) spot checks matched the column path in every family unless
noted. Nothing here is inferred from documentation.

## Similarity — levenshtein, editdist3, damerau_levenshtein, jaccard, hamming, mismatches

- UNIT is raw UTF-8 **bytes** for all six — never codepoints or graphemes.
  Witnesses: levenshtein('é','e')=2, jaccard('é','è')=1/3 (the two codepoints
  share lead byte 0xC3 — codepoint-set semantics would give 0.0),
  hamming('é','e') errors on byte-length mismatch while hamming('é','è')=1.
- Types: levenshtein/editdist3/damerau_levenshtein/hamming/mismatches →
  BIGINT; jaccard → DOUBLE. All NULL-strict every arg. All case-sensitive.
- `editdist3` == `levenshtein` (200-pair sweep, 0 diffs). Plain Levenshtein,
  empty strings fine: ('','')=0, ('','abc')=3.
- `damerau_levenshtein` is the **UNRESTRICTED** DL variant, transposition
  cost 1 — NOT restricted OSA. Witness: ('ca','abc')=2 (OSA gives 3);
  300-pair sweep 0 diffs vs an unrestricted-DL reference, 3 diffs vs OSA.
- `jaccard` = |A∩B|/|A∪B| over **single-byte sets**, duplicates ignored:
  ('ab','ba')=1.0, ('abc','abd')=0.5. Empty string on either side TRAPS:
  "Invalid Input Error: Jaccard Function: An argument too short!".
- `hamming` == `mismatches` (identical values AND error texts — the errors
  say "Mismatch Function" even for hamming). Byte-length mismatch traps
  "…Strings must be of equal length!"; ANY empty input (both empty too)
  traps "…Strings must be of length > 0!" — ('','') is an ERROR, not 0.
- Embedded NUL is an ordinary byte everywhere.

## String builders — repeat, lpad, rpad, replace, translate, concat_ws, concat

- `repeat(s,n)`: n≤0 → '' silently; NULL-strict; multi-byte safe. Huge n
  deliberately unpinned (OOM risk) — corpus uses small n.
- `lpad/rpad(s,l,pad)`: l counts **codepoints**. Truncation (l < length(s))
  keeps the FIRST l codepoints for BOTH lpad and rpad (rpad('abcdef',3,'x')
  = 'abc' — a naive rpad keeps the suffix; refuted). l≤0 → ''. pad cycles
  left-to-right cut to (l−length(s)) codepoints, never splitting a
  codepoint. pad='' traps "Invalid Input Error: Insufficient padding in
  LPAD." (resp. RPAD) ONLY when growth is needed (l > length(s)); otherwise
  the prefix returns without error — the trap is data-dependent. NULL-strict
  all three args.
- `replace(s,from,to)`: empty needle is a strict NO-OP; leftmost
  non-overlapping single pass, output not rescanned ('aaa','aa','b'→'ba');
  byte-sequence match; NULL-strict.
- `translate(s,from,to)`: per-**codepoint** map; from-chars beyond |to| are
  DELETED; duplicate in from → FIRST wins; to-extras ignored; to='' deletes
  every from-char; NULL-strict.
- `concat_ws(sep, args…)`: NULL args are SKIPPED with their separator;
  NULL sep → NULL; all args NULL → '' (NOT NULL). Zero value-args is a
  binder error. `concat(args…)` = same skip-NULL fold joined by '' — and is
  NOT sugar for `||`, which stays NULL-strict. DuckDB implicitly casts
  numeric/boolean VALUE args with its own float rendering ('1e+20',
  '-0.0'); the SEPARATOR never casts. v0 restricts both functions to
  VARCHAR args — non-VARCHAR args reject by name (we do not model DuckDB
  float-to-string rendering).
- `reverse` is **DESCOPED**: measured to operate on UAX-29 grapheme
  clusters (combining sequences attached, ZWJ emoji one unit, regional-
  indicator pairs swap as units — real segmentation, not an approximation).
  Full segmentation machinery for 3 corpus cases fails the cost test;
  rejects by name citing grapheme semantics. Pins retained for a future
  wave.

## Inspection + case aliases — unicode, ord, ascii, bit_length, ucase, lcase

- `unicode(s)`/`ord(s)`: FIRST codepoint as INTEGER; multi-char fine
  ('abc'→97); `unicode('')` = **-1**. `ord` == `unicode` (exhaustive
  1,112,063-codepoint sweep + 210 strings, 0 diffs).
- `ascii(s)` == unicode EXCEPT `ascii('')` = **0** — the sole divergence,
  and ascii does NOT restrict to ASCII (ascii('é')=233).
- `bit_length(s)` = 8 × strlen(s) exactly (bytes; BIGINT) — pure desugar.
  `octet_length` does not exist in DuckDB 1.5.5 (binder error).
- `ucase`/`lcase` == `upper`/`lower`: exhaustive all-codepoint sweep both
  directions, zero mismatches — pure aliases onto the existing casemap ops.

## VARCHAR subscripts — array_extract, list_extract, array_slice, list_slice, s[i], s[a:b]

- UNIT = **codepoints** (not graphemes: extract(2) of 'e'+U+0301 is the bare
  combining mark).
- `array_extract(s,i)` (== `list_extract` == `s[i]`, 220-row sweeps): 1-based;
  i=0 → ''; negative = from end (-1 last, resolution len+1+i with clamping);
  out-of-range EITHER direction → **''** (the LIST overload gives NULL — do
  not copy list semantics); NULL-strict; VARCHAR result.
- `array_slice(s,a,b)` (== `list_slice` == `s[a:b]`, 220-row sweeps):
  both-ends-INCLUSIVE, 1-based; negative from-end (-1 = last char); a≤0
  clamps to start; b>len clamps to end; fully out-of-range or a>b → '';
  NULL bound → NULL (**NULL is NOT an open bound** — open bounds exist only
  syntactically: s[:b] ≡ slice(1,b), s[a:] ≡ slice(a,-1), s[:] ≡
  slice(1,-1)).
- Step form REJECTS for every step value incl. 1: "Not implemented Error:
  Slice with steps has not been implemented for string types, …".
- `substr(s,2,3)`='ell' vs `array_slice(s,2,3)`='el' — (start,LENGTH) vs
  (begin,end) conventions; do not share lowering.

## Math tail — add, subtract, multiply, divide, mod, fmod, fdiv, nextafter

- `add/subtract/multiply/divide/mod` are EXACT aliases of `+ - * // %`:
  same values, types, and byte-identical error texts (overflow traps incl.
  INT64_MIN special cases) — pure frontend desugars onto existing ops.
  Unary add(x)=x, subtract(x)=-x also exist.
- `divide`(int,int) = truncating INT division (7,2 → 3); on DOUBLE, `//`
  and divide() are **plain division** (−7.5//2.0 = −3.75, NOT floor) with
  divisor==0 → NULL.
- `mod`/`%` = truncated C-fmod (dividend's sign) on ints AND doubles;
  INT x%0 → NULL (no trap); DOUBLE x%0.0 → **NaN 7ff8…** (a value, NOT
  NULL — the fleet's decision line over-generalized from the int probes;
  re-measured 2026-07-26, correction appended to pins-wave3/math_tail.json.
  NaN-sign addendum (CI-discovered): the %-by-zero NaN comes from LIBM
  fmod and its sign is PLATFORM-dependent — 7ff8 on Windows ucrt, fff8 on
  Linux glibc; both engines use the platform libm so they agree with the
  oracle per-platform, and the pin is bit AGREEMENT, not a constant.
  fmod-by-zero is hardware-generated (0·inf under SSE) and stays fff8
  everywhere); mod(−7.5,2.5) = **−0.0**.
- `fmod`/`fdiv` are the FLOOR-division pair, always DOUBLE: fdiv =
  floor(x/y) (±inf on zero divisor); fmod takes the **DIVISOR's** sign
  (fmod(−7.5,2.5) = +0.0 where mod gives −0.0) and is computed as
  x − floor(x/y)·y — so fmod(1.0, inf) = NaN (not 1.0 as C fmod);
  fmod(x,0) = NaN. Computed NaNs surface as fff8… (negative quiet NaN)
  where mod-by-zero NULLs and propagated NaNs stay 7ff8… — reproduce by
  computing naturally on x86, verified by bit-exact pin tests.
- `nextafter` = C nextafter bit-exact, TOTAL (no traps): x==y returns y
  (nextafter(0.0,−0.0) = −0.0); denormal/inf/NaN edges pinned by bits. Int
  args promote to DOUBLE. The (FLOAT,FLOAT) overload returns FLOAT (f32
  nextafterf) — out of v0 scope (f32 columns already reject at binding).
- VARCHAR never implicitly casts into any of these (binder error) — matches
  our no-implicit-cast rule.

## strip_accents — oracle-extracted table + Hangul compose

- NOT purely per-codepoint: per-cp map, THEN a canonical-compose pass whose
  only observable effect is **Hangul jamo composition** (L+V→LV, LV+T→LVT,
  including precomposed LV + T; formula 0xAC00+(L−0x1100)·588+(V−0x1161)·28
  +(T−0x11A7), verified over all 399 LV pairs + 200 LVT triples). Wrong-order
  jamo do not compose.
- The per-cp map is ORACLE-EXTRACTED (full non-surrogate sweep): 4460
  changed codepoints; 2450 map to '' (Mn/Mc/Me marks — Indic vowel signs
  are deleted, not just accents); every non-empty output is exactly ONE
  codepoint. Includes accent-free rewrites (U+212A→'K', U+2126→Ω,
  U+2000→U+2002). Compatibility decompositions NOT applied.
- DuckDB's tables lag Unicode 16 by 57 codepoints — the Rust table is
  generated from the oracle map (scripts/gen_strip_accents.py, casemap/pow10
  playbook), NEVER from a host Unicode library.
- Row-local algorithm (validated 518/518 strings): all-ASCII input →
  returned VERBATIM (embedded NULs preserved); otherwise truncate at the
  first NUL (context-dependent NUL quirk — even a map-unchanged non-ASCII
  char like an emoji triggers the truncating path), apply the map, compose
  Hangul. Idempotent; total; NULL-strict.

## Implementation addenda (found while building/testing)

- Overflow trap texts are now DuckDB's verbatim, operand values included:
  "Overflow in addition of INT64 (x + y)!" (sub/mul likewise), "Overflow
  in division of x / y" for BOTH `//` and `%` on i64::MIN op −1 (no
  trailing '!'), and "Overflow on abs(x)" (measured this wave). One
  shared `overflow_msg`/`abs_overflow_msg` feeds both backends.
- hamming's two traps check EQUAL LENGTH first: hamming('', 'a') raises
  the equal-length message; only ('','') reaches "length > 0".
- Documented laxity (same stance as wave-1 round digits): the engine
  binds lpad/rpad with an i64 length column where DuckDB wants INTEGER;
  constant integer lengths — the corpus shape — bind in both.
- concat/concat_ws literal -0.0 renders '0.0' in DuckDB (DECIMAL literal
  path); the pinned '-0.0' is the DOUBLE column path, which we follow.
- Corpus replay: f32 base tables classify clean-unsupported — widening
  to f64 is value-exact but every f32-GRID-sensitive op (nextafter ulp
  steps, FLOAT→VARCHAR rendering) computes on the wrong grid (2 nextafter
  cases were the wave's only FAILs before this rule; 3 sources, 5 cases).
- Unary add(x)=x / subtract(x)=-x exist in DuckDB but are unpinned —
  they reject by arity until measured.

## Catalogue rejections (do NOT ship)

- Aggregates (`sum`, `count`, `geomean`) reject as aggregate-by-name.
- `regexp_matches`/`regexp_extract`/`regexp_full_match`: RE2 semantics, own
  wave.
- `reverse`: grapheme-cluster semantics (above).
- Step slicing on VARCHAR, non-VARCHAR args to concat/concat_ws,
  column-count-changing star macros (`columns(...)`).
