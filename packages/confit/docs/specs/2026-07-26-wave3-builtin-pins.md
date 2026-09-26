# Builtin pins: similarity, string builders, inspection, VARCHAR subscripts, math tail, strip_accents

DuckDB 1.5.5. Every claim was measured through the vectorized path (table
columns, not literals); the full pin tables (300+ probes with exact reprs,
float bit patterns, result types, and verbatim error heads) are the JSON
files in `pins-wave3/`. Literal (constant-fold) spot checks matched the
column path in every family unless noted. Nothing here is inferred from
documentation.

## Similarity — levenshtein, editdist3, damerau_levenshtein, jaccard, hamming, mismatches

- UNIT is raw UTF-8 **bytes** for all six — never codepoints or graphemes.
  Witnesses: levenshtein('é','e') = 2, jaccard('é','è') = 1/3 (the two
  codepoints share lead byte 0xC3), hamming('é','e') errors on byte-length
  mismatch while hamming('é','è') = 1.
- Types: levenshtein/editdist3/damerau_levenshtein/hamming/mismatches →
  BIGINT; jaccard → DOUBLE. All NULL-strict in every arg. All
  case-sensitive.
- `editdist3` == `levenshtein` (200-pair sweep, 0 diffs). Empty strings
  fine: ('','') = 0, ('','abc') = 3.
- `damerau_levenshtein` is the **UNRESTRICTED** DL variant, transposition
  cost 1 — not restricted OSA. Witness: ('ca','abc') = 2 (OSA gives 3).
- `jaccard` = |A∩B|/|A∪B| over **single-byte sets**, duplicates ignored:
  ('ab','ba') = 1.0, ('abc','abd') = 0.5. An empty string on either side
  TRAPS: "Invalid Input Error: Jaccard Function: An argument too short!".
- `hamming` == `mismatches` (identical values AND error texts — the errors
  say "Mismatch Function" even for hamming). The EQUAL-LENGTH check runs
  first: a byte-length mismatch traps "…Strings must be of equal length!"
  (including hamming('', 'a')); only ('','') reaches "…Strings must be of
  length > 0!" — ('','') is an ERROR, not 0.
- Embedded NUL is an ordinary byte everywhere.

## String builders — repeat, lpad, rpad, replace, translate, concat_ws, concat

- `repeat(s,n)`: n ≤ 0 → '' silently; NULL-strict; multi-byte safe. Huge n
  is unpinned.
- `lpad/rpad(s,l,pad)`: l counts **codepoints**. Truncation (l <
  length(s)) keeps the FIRST l codepoints for BOTH lpad and rpad
  (rpad('abcdef',3,'x') = 'abc'). l ≤ 0 → ''. pad cycles left-to-right,
  cut to (l − length(s)) codepoints, never splitting a codepoint. pad = ''
  traps "Invalid Input Error: Insufficient padding in LPAD." (resp. RPAD)
  ONLY when growth is needed (l > length(s)) — the trap is data-dependent.
  NULL-strict in all three args. The count is INTEGER: a BIGINT count is a
  binder error in DuckDB and in confit.
- `replace(s,from,to)`: empty needle is a strict NO-OP; leftmost
  non-overlapping single pass, output not rescanned ('aaa','aa','b' →
  'ba'); byte-sequence match; NULL-strict.
- `translate(s,from,to)`: per-**codepoint** map; from-chars beyond |to|
  are DELETED; duplicate in from → FIRST wins; extra to-chars ignored;
  to = '' deletes every from-char; NULL-strict.
- `concat_ws(sep, args…)`: NULL args are SKIPPED with their separator;
  NULL sep → NULL; all args NULL → '' (not NULL). Zero value-args is a
  binder error. `concat(args…)` = the same skip-NULL fold joined by '' —
  not sugar for `||`, which is NULL-strict. Numeric/boolean VALUE args cast
  to VARCHAR with DuckDB's float rendering ('1e+20', '-0.0'); the
  SEPARATOR never casts (a non-VARCHAR separator is a binder error).
  A literal -0.0 renders '0.0' (DECIMAL literal path); '-0.0' is the DOUBLE
  column path, which confit follows.
- `reverse`: see `2026-07-28-waveA-structural-tails.md`.

## Inspection + case aliases — unicode, ord, ascii, bit_length, ucase, lcase

- `unicode(s)`/`ord(s)`: FIRST codepoint as INTEGER ('abc' → 97);
  `unicode('')` = **-1**. `ord` == `unicode` (exhaustive codepoint sweep +
  210 strings, 0 diffs).
- `ascii(s)` == unicode EXCEPT `ascii('')` = **0**; ascii does not
  restrict to ASCII (ascii('é') = 233).
- `bit_length(s)` = 8 × strlen(s) (BIGINT). `octet_length` does not exist
  in DuckDB 1.5.5 (binder error).
- `ucase`/`lcase` == `upper`/`lower` (exhaustive all-codepoint sweep, zero
  mismatches).

## VARCHAR subscripts — array_extract, list_extract, array_slice, list_slice, s[i], s[a:b]

- UNIT = **codepoints** (extract(2) of 'e'+U+0301 is the bare combining
  mark).
- `array_extract(s,i)` (== `list_extract` == `s[i]`): 1-based; i = 0 → '';
  negative = from end (-1 last); out-of-range in EITHER direction → **''**
  (the LIST overload gives NULL); NULL-strict; VARCHAR result. confit
  serves the VARCHAR overload only.
- `array_slice(s,a,b)` (== `list_slice` == `s[a:b]`): both ends INCLUSIVE,
  1-based; negative from end; a ≤ 0 clamps to start; b > len clamps to end;
  fully out-of-range or a > b → ''; NULL bound → NULL (**NULL is not an
  open bound** — `s[:b]` ≡ slice(1,b), `s[a:]` ≡ slice(a,-1), `s[:]` ≡
  slice(1,-1)).
- Step slicing fails for every step value incl. 1: "Not implemented Error:
  Slice with steps has not been implemented for string types, …". confit
  refuses it.
- `substr(s,2,3)` = 'ell' vs `array_slice(s,2,3)` = 'el' — (start, LENGTH)
  vs (begin, end).

Bounds/index details: `2026-07-26-wave5-structural-pins.md`.

## Math tail — add, subtract, multiply, divide, mod, fmod, fdiv, nextafter

- `add/subtract/multiply/divide/mod` are EXACT aliases of `+ - * // %`:
  same values, types, and byte-identical error texts. Unary add(x) = x and
  subtract(x) = -x exist in DuckDB but are unpinned; confit refuses the
  1-arg forms.
- Overflow texts (verbatim, operand values included): "Overflow in
  addition of INT64 (x + y)!" (subtraction/multiplication likewise),
  "Overflow in division of x / y" for BOTH `//` and `%` on i64::MIN op −1
  (no trailing '!'), "Overflow on abs(x)".
- `divide`(int,int) = truncating integer division (7,2 → 3); on DOUBLE,
  `//` and divide() are **plain division** (−7.5 // 2.0 = −3.75, not floor)
  with divisor == 0 → NULL.
- `mod`/`%` = truncated C-fmod (dividend's sign) on ints AND doubles; INT
  x % 0 → NULL; DOUBLE x % 0.0 → **NaN** (a value, not NULL);
  mod(−7.5, 2.5) = **−0.0**. The %-by-zero NaN comes from libm fmod and its
  sign is platform-dependent (7ff8 on Windows ucrt, fff8 on Linux glibc);
  both engines use the platform libm, so the pin is bit AGREEMENT, not a
  constant.
- `fmod`/`fdiv` are the FLOOR-division pair, always DOUBLE: fdiv =
  floor(x/y) (±inf on zero divisor); fmod takes the **DIVISOR's** sign
  (fmod(−7.5, 2.5) = +0.0 where mod gives −0.0) and is computed as
  x − floor(x/y)·y, so fmod(1.0, inf) = NaN (not 1.0 as C fmod);
  fmod(x, 0) = NaN. fmod-by-zero NaN is hardware-generated (0·inf under
  SSE) and is fff8 everywhere; computed NaNs surface as fff8… while
  propagated NaNs stay 7ff8….
- `nextafter` = C nextafter bit-exact, TOTAL: x == y returns y
  (nextafter(0.0, −0.0) = −0.0); denormal/inf/NaN edges pinned by bits.
  Int args promote to DOUBLE. The (FLOAT, FLOAT) overload returns FLOAT
  (f32 nextafterf).
- VARCHAR never implicitly casts into any of these (binder error).
- The corpus replay classifies f32 base tables as clean-unsupported:
  widening to f64 is value-exact, but f32-grid-sensitive ops (nextafter ulp
  steps, FLOAT → VARCHAR rendering) would compute on the wrong grid.

## strip_accents — oracle-extracted table + Hangul compose

- Not purely per-codepoint: a per-cp map, THEN a canonical-compose pass
  whose only observable effect is **Hangul jamo composition** (L+V → LV,
  LV+T → LVT, including precomposed LV + T; formula
  0xAC00+(L−0x1100)·588+(V−0x1161)·28+(T−0x11A7), verified over all 399 LV
  pairs + 200 LVT triples). Wrong-order jamo do not compose.
- The per-cp map is ORACLE-EXTRACTED (full non-surrogate sweep): 4460
  changed codepoints; 2450 map to '' (Mn/Mc/Me marks — Indic vowel signs
  are deleted, not just accents); every non-empty output is exactly ONE
  codepoint. Includes accent-free rewrites (U+212A → 'K', U+2126 → Ω,
  U+2000 → U+2002). Compatibility decompositions are NOT applied.
- DuckDB's tables lag Unicode 16 by 57 codepoints. confit's table
  (`exec/strip_accents.rs`) is generated from the oracle map by
  `scripts/gen_strip_accents.py`, never from a host Unicode library.
- Row-local algorithm (validated 518/518 strings): all-ASCII input →
  returned VERBATIM (embedded NULs preserved); otherwise truncate at the
  first NUL (even a map-unchanged non-ASCII char like an emoji triggers the
  truncating path), apply the map, compose Hangul. Idempotent; total;
  NULL-strict.

## Catalogue facts

- Aggregates (`sum`, `count`, `avg`, `min`, `max`, `geomean`, …): confit
  refuses them by name (no aggregation).
- `columns(...)` as a column-count-changing star macro: see
  `2026-07-27-waveB-regexp-pins.md`.
