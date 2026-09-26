# Structural and dialect pins: parsing, slices, subscripts, bitwise, GLOB, star forms, binder

DuckDB 1.5.5 is the oracle; every claim below is backed by an executed
query recorded verbatim in `pins-wave5/*.json`. Tests:
`tests/test_duckdb_wave5_structural.py`.

Where DuckDB's behavior is a quirk (SIMILAR TO star asymmetry, silent
RENAME ignore, `^` = power), confit reproduces the quirk or refuses — it
does not "fix" it.

## Parsing (sqlparser 0.62, `pins-wave5/sqlparser-spike.json`)

- Under DuckDbDialect these already parse: slices `s[a:b]`, subscripts
  `s[off]`/`s[-1]`, bitwise `<< >> & |`, `* REPLACE`, `* EXCLUDE (t.abc)`
  (qualified name preserved), lateral aliases, `t AS u(x, y)`, NATURAL
  JOIN, `s1.t1`, `COLUMNS('re')` (an ordinary Function), IN-list mixing.
- GenericDialect is a strict superset on the sampled forms — it adds `^@`
  (BinaryOp PGStartsWith), `* ILIKE`, `* RENAME`. confit's frontend parses
  with GenericDialect.
- Forms neither dialect parses, which confit rewrites at the token level
  before sqlparser:
  - `* LIKE / NOT LIKE / ILIKE / GLOB / SIMILAR TO 'pat'` star filters: the
    filter suffix is carried side-band into star expansion as a marker.
    sqlparser parses `* ILIKE` and EXCLUDE as mutually exclusive, so the
    marker also carries the EXCLUDE entries.
  - `expr GLOB pat` → `expr LIKE <glob marker>(pat)`; the binder binds the
    dedicated GLOB matcher. `NOT GLOB` is a parse error in DuckDB itself
    and stays one.
  - `k: expr` select-item colon alias misparses SILENTLY as Snowflake
    JsonAccess under every dialect; confit rewrites `ident :` at
    select-item start to `expr AS ident`, and the binder rejects any
    JsonAccess that survives.

## Slices `s[a:b]` (`pins-wave5/slices.json`)

1-based, both ends INCLUSIVE, unit = CODEPOINTS. Negative bound =
`len+1+bound` (-1 = last). Clamps: start < 1 → 1, end > len → len; resolved
start > end → `''` (never NULL; [4:2], [50:60], [-60:-50] all ''). NULL
string or ANY NULL bound → NULL — **NULL is not an open bound**; open bounds
are pure syntax: `[:b]` ≡ `[1:b]`, `[a:]` ≡ `[a:-1]`, `[:]` ≡ `[1:-1]`.
Result always VARCHAR; `''` → `''`. Bounds implicit-cast to INT64 (DOUBLE
rounds half away from zero: `'abcdef'[2.5:4]` = 'cd'; numeric VARCHAR and
BOOLEAN accepted; out-of-i64 literals = bind-time INT128→INT64 Conversion
Error). No runtime overflow: i64 extremes clamp. `s[a:b]` ≡
`array_slice(s,a,b)`. Step form `s[a:b:step]` fails "Not implemented
Error: Slice with steps..." for EVERY step incl. 1; confit refuses it at
prepare. BLOB slicing is byte-addressed; confit has no BLOB type.

## Subscripts `s[i]` (`pins-wave5/subscripts-extended.json`)

Never NULL for non-NULL inputs: `s[0]` and out-of-range positive AND
negative → `''`. Negative indexes from the end: pos = `len+i+1`
('hello'[-1] = 'o'). Codepoint-based, 1-based. Hard index window
**[-4294967296, 4294967295]** (asymmetric): outside raises "Out of Range
Error: Substring offset outside of supported range (> 4294967295)" /
"(< -4294967296)" — **execution-time, per row, after NULL propagation**
(NULL string skips the check; `''` does not; dead CASE branches never fire
it). A beyond-i64 literal is a bind-time Binder Error (array_extract
no-match). Dynamic indexes behave like literals; index arithmetic overflow
traps as i64 addition. Bind: sub-BIGINT ints ok; HUGEINT/UBIGINT/DOUBLE/
DECIMAL/BOOLEAN reject (Binder Error); a VARCHAR index casts
('hello'['2'] = 'e', non-numeric → Conversion Error).

## Bitwise ops (`pins-wave5/bitwise-int-ops.json`)

Per row on i64; NULL propagates FIRST (NULL masks would-be errors):

- `<<`: a < 0 → "Cannot left-shift negative number {a}" (even a<<0); else
  b < 0 → "Cannot left-shift by negative number {b}"; else a == 0 → 0 (the
  zero shortcut precedes the range check); else b ≥ 64 → "Left-shift value
  {b} is out of range"; else a ≥ 1<<(63-b) → "Overflow in left shift ({a}
  << {b})"; else a<<b. All texts "Out of Range Error: ...".
- `>>`: NEVER errors: (b < 0 || b ≥ 64) → 0 (even for negative a); else
  arithmetic sign-extending shift.
- `&`, `|`, `xor()`, `~`: plain two's complement, never error. **xor is
  function-only; `^` is POWER (DOUBLE).** `#` and `XOR` don't parse; `~`
  prefix binds looser than arithmetic (~1+1 = ~(1+1) = -3) but tighter
  than shifts.
- Precedence: `<< >> & |` are ONE flat left-associative tier below
  arithmetic, above comparisons (4|1&1 = 1).
- confit serves `<< >> & |` and `xor()`; it refuses `~` (prefix bitnot),
  `#`, and `^` because sqlparser's precedence for them differs from
  DuckDB's.
- Narrow widths: DuckDB checks overflow at the narrower width when BOTH
  operands are sub-BIGINT (127::TINYINT<<1 errors; i64 gives 254). confit
  computes in i64, which equals DuckDB whenever either operand is BIGINT.

## Text operators (`pins-wave5/text-operators.json`)

- `^@` ≡ `starts_with(s,p)` ≡ `prefix()`: byte-prefix compare,
  case-sensitive, empty prefix true for all non-NULL, NULL-strict,
  BOOLEAN.
- **GLOB is not expressible via LIKE**: `?` consumes exactly one BYTE
  (LIKE `_` is a codepoint); bracket classes match one byte from a byte
  set. confit's byte-level matcher (`kernels::duck_glob`): `*` = any byte
  run; `?` = one byte; `\c` = literal next byte OUTSIDE classes, dangling
  `\` → match nothing; classes: leading `!` negates (`^` is a LITERAL —
  '[^h]ello' matches 'hello'), `]` literal if first member, `-` literal if
  first else a range whose endpoint may be `]` (`[a-]` matches NOTHING —
  `]` is eaten as the endpoint, class unclosed); malformed patterns never
  error, they match nothing. Case-sensitive, NULL-strict, VARCHAR-only (no
  implicit casts). `glob()` the function is a TABLE function (lists
  files), not a scalar alias.

## Star forms (`pins-wave5/star-forms.json`)

- Name filters (`* LIKE/ILIKE/GLOB 'pat'`): bind-time filter of the
  expanded star by matching column NAMES (declared case; LIKE
  case-sensitive, ILIKE case-folds, GLOB byte matcher); survivors keep
  table order; zero matches = Binder Error (DuckDB's text embeds its
  internal COLUMNS(list_filter(...)) desugaring; confit's says "resulted in
  an empty set of columns"). The pattern must be a constant.
- Grammar order: `* [EXCLUDE] [REPLACE] [RENAME]`; a name filter may only
  combine with EXCLUDE and must FOLLOW it; REPLACE/RENAME + filter parse
  but fail at bind: "Replace/Rename list cannot be combined with a
  filtering operation".
- EXCLUDE: case-insensitive identifier match, qualified names allowed,
  unknown → "Column \"x\" in EXCLUDE list not found in FROM clause",
  duplicates → Parser Error, excluding everything → "SELECT list is empty
  after resolving * expressions!". On joins: unqualified EXCLUDE strips ALL
  copies; qualified strips one; **on a USING join, EXCLUDE (t1.id)
  UNMERGES the merged column** (it disappears from its merged position and
  reappears at t2's ordinal with t2's values). confit refuses EXCLUDE of a
  USING-merged column; all other qualified EXCLUDE forms serve.
- REPLACE `(expr AS col)`: keeps position and name, may change type; expr
  sees ALL original columns (including EXCLUDEd ones); unknown target
  errors; an ambiguous unqualified target on a join errors; the same column
  in EXCLUDE and REPLACE = Parser Error.
- RENAME `(a AS q)`: position preserved; **a nonexistent target is
  silently ignored** (unlike EXCLUDE/REPLACE); collisions silently produce
  duplicate output names; an ambiguous join name renames BOTH copies.
- `* SIMILAR TO` and `COLUMNS('re')`: `2026-07-27-waveB-regexp-pins.md`.

## Duplicate output names (`pins-wave5/dup-names-client-contract.json`)

DuckDB's binder renames duplicates at every subquery/CTE/CTAS boundary, and
`.df()`/`.fetchdf()` apply the IDENTICAL algorithm: **left-to-right scan
after star expansion; the first occurrence keeps its name; later ones get
`<own-original-case-name>_N`, smallest free N; the collision check is
case-insensitive and checks candidate names too** (id,ID → id,ID_1;
id,id,id_1 → id,id_1,id_1_1). Never reorders.

confit (`dedup_output_names` in `frontend.rs`) applies this rename to its
output columns — synthesized model, dict mode, and slot-fill. A supplied
output model is validated against the renamed names.

## Binder (`pins-wave5/binder-tail.json`)

- Lateral aliases: **a real column beats a select alias** in both SELECT
  and WHERE (`a+1 AS k, k*2` uses column k). Left-to-right chains work; a
  forward reference errors "cannot be referenced before it is defined";
  WHERE sees aliases only when no real column shares the name.
- `t AS u(x, y)`: a partial list is legal (prefix rename); too many names =
  Binder Error; old column names AND the original table name stop working
  as qualifiers.
- NATURAL JOIN: dedups like USING (join columns first, left spelling, both
  qualifiers addressable); case-differing names DO match; **no common
  columns = Binder Error** ("No columns found to join on in NATURAL
  JOIN... Use CROSS JOIN..."), not a cross product; NATURAL LEFT
  null-fills.
- Schema qualifiers: `main.tbl` resolves to bare `tbl`; every OTHER
  qualifier is a Catalog Error ("...does not exist because schema \"test\"
  does not exist.") — never a silent bare-name fallback.
- Mixed-type comparisons (=, BETWEEN, IN): VARCHAR vs INT casts the STRING
  to the int side numerically with half-away-from-zero rounding (2 = '1.5'
  TRUE, '  5  ' = 5 TRUE, '1e2' = 100 TRUE); non-numeric strings =
  Conversion Error (bind-time for top-level constant comparisons,
  EXECUTION-time inside an IN-list against a column — `a IN ('a', ...)` on
  an empty table succeeds). BOOLEAN vs INT casts the BOOL to int (TRUE = 2
  is FALSE). IN/BETWEEN with NULL stay 3-valued. confit converts numeric
  and boolean literals at bind and refuses BETWEEN/IN that mix non-numeric
  string literals with numbers.
- NULL <op> NULL result types: `+ - * %` → BIGINT, `/` → DOUBLE,
  comparisons → BOOLEAN, `-NULL` → BIGINT, `NULL || NULL` → SQLNULL, which
  materializes as INTEGER.
- `try_trim_null` does not exist in DuckDB.
