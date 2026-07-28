# Wave 5 — structural + dialect sweep: pins (TASK-52)

Measured 2026-07-26 against DuckDB 1.5.5 (the oracle) by an 8-agent fleet;
every claim below is backed by an executed query recorded verbatim in
`pins-wave5/*.json` (committed next to this file). Census baseline at master
13a3a7e: 395 match / 281 unsupported / 2 known-divergent, zero FAILs.

Discipline note: pins are engine==oracle contracts. Where DuckDB's behavior
is a quirk (SIMILAR-TO star asymmetry, silent RENAME ignore, `^` = power),
we reproduce the quirk — we do not "fix" it.

## Parser strategy (sqlparser 0.62 spike, `pins-wave5/sqlparser-spike.json`)

Measured per-form on the repo's pinned sqlparser 0.62.0:

- **Already parse under DuckDbDialect** (bind-only work): slices `s[a:b]` and
  subscripts `s[off]`, `s[-1]` (CompoundFieldAccess/Subscript), bitwise
  `<< >> & |` (PGBitwiseShiftLeft/Right, BitwiseAnd/Or), `* REPLACE`,
  `* EXCLUDE (t.abc)` (qualified, 2-part ObjectName preserved), lateral
  aliases, `t AS u(x, y)`, NATURAL JOIN, `s1.t1`, `COLUMNS('re')` (ordinary
  Function — intercept by name), IN-list mixing.
- **GenericDialect is a strict superset on the 20-form sample** — it adds
  `^@` (parses as BinaryOp PGStartsWith), `* ILIKE`, `* RENAME`; nothing
  parsed only under DuckDbDialect. The DataFusion oracle path
  (packages/sql-transform/src/datafusion/plan.rs) already uses GenericDialect.
  **Decision: switch the specializer frontend to GenericDialect**, verified
  by the full Rust suite + 678-case corpus replay + py gate as the
  regression net (tokenization differences outside the sample are the risk;
  the corpus is the detector). Fallback if the net catches regressions:
  stay on DuckDbDialect and drop ^@/* ILIKE/* RENAME to clean-unsupported.
- **Parses under NEITHER — needs token-level pre-rewrite before sqlparser**:
  - `* LIKE / NOT LIKE / SIMILAR TO 'pat'` star filters: strip the filter
    suffix from the token stream (grammar position pinned below), carry it
    side-band into expand_star().
  - `expr GLOB pat`: rewrite `GLOB` → `LIKE` and wrap the RHS primary as a
    marker call (`s GLOB p` → `s LIKE __glob_pat(p)`); the binder sees the
    marker and binds the dedicated GLOB matcher. `NOT GLOB` is a parse
    error in DuckDB itself — never rewritten, stays a parse error.
  - **`k: expr` colon alias misparses SILENTLY as Snowflake JsonAccess under
    every dialect — no parse error will ever fire.** Pre-rewrite
    `ident COLON` at select-item start → `expr AS ident`; additionally the
    binder must reject any JsonAccess that survives (today's binder already
    rejects it as unsupported — that guard stays as the backstop).

## Slices `s[a:b]` (`pins-wave5/slices.json`)

1-based, both-ends-INCLUSIVE, unit = CODEPOINTS. Negative bound =
`len+1+bound` (-1 = last). Clamps: start<1 → 1, end>len → len; resolved
start>end → `''` (never NULL; [4:2], [50:60], [-60:-50] all ''). NULL-strict:
NULL string or ANY NULL bound → NULL — **NULL is not an open bound**; open
bounds are pure syntax: `[:b]`≡`[1:b]`, `[a:]`≡`[a:-1]`, `[:]`≡`[1:-1]`.
Result always VARCHAR; `''` input → `''`. Bounds implicit-cast to INT64
(DOUBLE rounds half-away-from-zero: `'abcdef'[2.5:4]`='cd'; numeric VARCHAR
and BOOLEAN accepted; out-of-i64 literals = bind-time INT128→INT64
Conversion Error, verbatim in JSON). No runtime overflow: i64 extremes clamp.
`s[a:b]` ≡ `array_slice(s,a,b)` exactly. Step form `s[a:b:step]` traps
"Not implemented Error: Slice with steps..." for EVERY step incl. 1 —
reproduce at bind time verbatim. BLOB slicing is byte-addressed — out of
scope (engine has no BLOB type).

## Extended subscripts `s[i]` (`pins-wave5/subscripts-extended.json`)

Never NULL for non-NULL inputs: `s[0]`, out-of-range positive AND negative
→ `''`. Negative indexes from the end: pos = `len+i+1` ('hello'[-1]='o').
Codepoint-based, 1-based. Hard index window **[-4294967296, 4294967295]**
(asymmetric): outside raises verbatim "Out of Range Error: Substring offset
outside of supported range (> 4294967295)" / "(< -4294967296)" —
**execution-time, per-row, after NULL propagation** (NULL string skips the
check; `''` does not; dead CASE branches never fire it). Beyond-i64 literal
is a different bind-time Binder Error (array_extract no-match, verbatim in
JSON). Dynamic/expression indexes identical to literals; index arithmetic
overflow traps i64-add verbatim. Bind: sub-BIGINT ints ok; HUGEINT/UBIGINT/
DOUBLE/DECIMAL/BOOLEAN reject (Binder Error verbatim); VARCHAR index casts
('hello'['2']='e', non-numeric → Conversion Error).

## Bitwise ops (`pins-wave5/bitwise-int-ops.json`)

Per-row on i64, NULL-propagates FIRST (NULL masks would-be errors row-wise):

- `<<`: a<0 → "Cannot left-shift negative number {a}" (even a<<0);
  else b<0 → "Cannot left-shift by negative number {b}"; else a==0 → 0
  (zero shortcut precedes range check); else b>=64 → "Left-shift value {b}
  is out of range"; else a >= 1<<(63-b) → "Overflow in left shift
  ({a} << {b})"; else a<<b. All texts "Out of Range Error: ..." verbatim.
- `>>`: NEVER errors: (b<0 || b>=64) → 0 (even for negative a); else
  arithmetic sign-extending shift.
- `& | xor() ~`: plain two's-complement, never error. **xor is
  function-only; `^` is POWER (DOUBLE) in DuckDB — never map `^` to xor.**
  `#` and `XOR` don't parse; `~` prefix binds looser than arithmetic
  (~1+1 = ~(1+1) = -3) but tighter than shifts.
- Precedence: `<< >> & |` are ONE flat left-associative tier below
  arithmetic, above comparisons (4|1&1 = 1). sqlparser's PG precedence
  must be verified against this in tests.
- **Documented divergence (narrow widths)**: DuckDB checks overflow at the
  narrower operand width when BOTH operands are sub-BIGINT (127::TINYINT<<1
  errors; i64 gives 254). Engine computes in i64 == DuckDB whenever either
  operand is BIGINT. Row-model ints are always BIGINT, so this arises only
  via explicit narrow CASTs — those are already unsupported in the engine;
  keep them so.

## Text operators (`pins-wave5/text-operators.json`)

- `^@` ≡ `starts_with(s,p)` ≡ `prefix()`: byte-prefix compare,
  case-sensitive, empty prefix true for all non-NULL, NULL-strict, BOOLEAN.
  Map to the existing starts_with kernel.
- **GLOB is NOT expressible via the LIKE family**: `?` consumes exactly one
  BYTE (LIKE `_` is codepoint); bracket classes match one byte from a
  byte-set (multi-byte members never match as a unit). Dedicated byte-level
  matcher: `*` = any byte run; `?` = one byte; `\c` = literal next byte
  OUTSIDE classes, dangling `\` → match nothing; classes: leading `!`
  negates (`^` is a LITERAL — '[^h]ello' matches 'hello'), `]` literal if
  first member, `-` literal if first else range whose endpoint may be `]`
  (`[a-]` matches NOTHING — `]` eaten as endpoint, class unclosed);
  malformed patterns NEVER error, they match nothing. Case-sensitive,
  NULL-strict, VARCHAR-only (no implicit casts; binder errors verbatim).
  `glob()` the function is a TABLE function (lists files) — do not create a
  scalar alias. `NOT GLOB` is a parse error in DuckDB.

## Star forms (`pins-wave5/star-forms.json`)

- Name filters (`* LIKE/ILIKE/GLOB 'pat'`): bind-time filter of the
  expanded star by matching the pattern against column NAMES (declared
  case; LIKE case-sensitive, ILIKE case-folds, GLOB byte matcher);
  survivors keep table order; zero matches = Binder Error (verbatim texts in
  JSON embed DuckDB's internal COLUMNS(list_filter(...)) desugaring —
  reproduce as clean-unsupported-shaped errors, exact text). Pattern must
  be a constant ("Pattern applied to a star expression must be a constant").
- Grammar order FIXED: `* [EXCLUDE] [REPLACE] [RENAME]`; a name filter may
  only combine with EXCLUDE and must FOLLOW it; REPLACE/RENAME + filter
  parse but fail at bind: "Replace/Rename list cannot be combined with a
  filtering operation" (verbatim).
- EXCLUDE: case-insensitive identifier match, qualified names allowed,
  unknown → "Column \"x\" in EXCLUDE list not found in FROM clause",
  duplicates → Parser Error, excluding everything → "SELECT list is empty
  after resolving * expressions!". On joins: unqualified EXCLUDE strips ALL
  copies; qualified strips one; **on a USING join, EXCLUDE (t1.id) UNMERGES
  the merged column** (disappears from front, reappears at t2's ordinal
  with t2's values).
- REPLACE `(expr AS col)`: keeps position and name, may change type, expr
  sees ALL original columns (including EXCLUDEd ones); unknown target
  errors; ambiguous unqualified target on a join errors; same col in
  EXCLUDE+REPLACE = Parser Error.
- RENAME `(a AS q)`: position preserved; **nonexistent target silently
  ignored** (unlike EXCLUDE/REPLACE); collisions silently produce duplicate
  output names; ambiguous join name renames BOTH copies.
- Deferred to wave-B (regexp): `* SIMILAR TO` (unanchored RE2 search over
  names; NOT variant is NOT-full-match — not complements!) and
  `COLUMNS('re')` — both need a regex engine; classify clean-unsupported
  with the pinned zero-match/compile error shapes when reachable.

## Duplicate output names (`pins-wave5/dup-names-client-contract.json`)

DuckDB's own binder renames duplicates at every subquery/CTE/CTAS boundary,
and .df()/.fetchdf() apply the IDENTICAL algorithm — so this is DuckDB
semantics, not client convenience: **left-to-right scan after star
expansion; first occurrence keeps its name; later ones get
`<own-original-case-name>_N`, smallest free N; the collision check is
case-insensitive and checks candidate names too** (id,ID → id,ID_1;
id,id,id_1 → id,id_1,id_1_1). Never reorders.

Contract: apply this rename in the synthesized model, dict mode, and
slot-fill. For a supplied output_model, rename first, then validate fields
against renamed names (a pydantic model cannot hold two `id` fields).
The corpus compares positionally (`model_dump().values()`), so renaming
cannot create a wrong match; NOT renaming dict-collapses last-wins and
fails. Flips corpus cases 10 and 282 immediately; others in the bucket are
gated by self-join first.

## Binder tail (`pins-wave5/binder-tail.json`)

- Lateral aliases: **the REAL column beats the select alias** in both
  SELECT and WHERE (a+1 AS k, k*2 uses column k). Left-to-right chains
  work; forward reference = verbatim "cannot be referenced before it is
  defined"; aliases invisible inside aggregate arguments (n/a in v0);
  WHERE sees aliases only when no real column shares the name.
- `t AS u(x, y)`: partial list legal (prefix rename); too many names =
  verbatim Binder Error; old column names AND the original table name die
  as qualifiers (verbatim errors in JSON).
- NATURAL JOIN: dedup like USING (join cols first, left spelling, both
  quals addressable); case-differing names DO match; **no common columns =
  hard Binder Error** ("No columns found to join on in NATURAL JOIN...
  Use CROSS JOIN..."), not a cross product; NATURAL LEFT null-fills.
- Schema qualifiers: `main.tbl` resolves to bare `tbl`; every OTHER
  qualifier rejects with the verbatim Catalog Error shape ("...does not
  exist because schema \"test\" does not exist.") — never a silent
  bare-name fallback.
- Mixed-type comparisons (=, BETWEEN, IN): VARCHAR vs INT casts the STRING
  to the int side numerically with half-away-from-zero rounding
  (2='1.5' TRUE, '  5  '=5 TRUE, '1e2'=100 TRUE); non-numeric strings =
  Conversion Error (bind-time for literals, per-row for columns), verbatim
  INT32/INT64 texts; no widening past i64. BOOLEAN vs INT casts the BOOL to
  int (TRUE=2 is FALSE). IN/BETWEEN with NULL stay 3-valued.
- NULL <op> NULL result types: `+ - * %` → BIGINT, `/` → DOUBLE,
  comparisons → BOOLEAN, `-NULL` → BIGINT, `NULL || NULL` → SQLNULL which
  materializes as INTEGER (treat as i64 out-col).
- `try_trim_null` does not exist in DuckDB (stays clean-unsupported as an
  unknown function); `COLUMNS` is star-expansion only (wave-B).

## Implementation addendum (2026-07-27, post-landing corrections)

Deviations and corrections discovered while landing — same discipline as
the wave-3 addendum (the corpus replay is the arbiter):

- **Mixed BETWEEN/IN with non-numeric string literals stays
  clean-unsupported, not a bind error.** The binder-tail pin's bind-time
  "Conversion Error" applies to top-level constant comparisons; inside an
  IN-list against a COLUMN the conversion is EXECUTION-time — corpus case
  `a IN ('a', ...)` on an EMPTY table succeeds in DuckDB. Caught as a
  replay FAIL, fixed to conservative unsupported. Numeric/bool literals DO
  convert at bind (half-away rounding, bool -> 0/1).
- **sqlparser parses `* ILIKE` and EXCLUDE as mutually exclusive**, so the
  star-filter rewrite absorbs the EXCLUDE entries into its marker string
  and the binder re-applies them.
- Error-class simplifications (all corpus-clean, texts not verbatim):
  zero-match star filters use a short "empty set of columns" Bind text
  rather than DuckDB's internal COLUMNS(list_filter(...)) desugaring;
  `* SIMILAR TO` and REPLACE/RENAME-plus-filter classify at parse (DuckDB
  parses then bind-errors); EXCLUDE/REPLACE duplicate-entry checks surface
  at bind with matching wording.
- `* EXCLUDE (t.key)` on a USING join stays unsupported (DuckDB UNMERGES
  the column — not modeled); all other qualified EXCLUDE forms serve.
- `~` (prefix bitnot) and `#` stay unsupported alongside `^` (pow): their
  sqlparser precedence disagrees with DuckDB's measured one, and mapping
  them would silently compute the wrong tree. xor() covers the semantics.
- NULL || NULL types as VARCHAR here (DuckDB's SQLNULL materializes as
  INTEGER); value-NULL either way, positional corpus compare unaffected.

## Implementation stages (each lands with tests + corpus replay green)

1. Parser groundwork: GenericDialect switch (full regression net), colon-
   alias pre-rewrite, GLOB pre-rewrite, star-filter capture.
2. Slices + extended subscripts: shared codepoint kernel, both backends,
   runtime range traps verbatim.
3. Bitwise `<< >> & |` + `xor()` + `~`: exact error ladder, NULL-first.
4. `^@` → starts_with; dedicated GLOB byte matcher.
5. Star forms: name filters, qualified EXCLUDE, REPLACE, RENAME; duplicate-
   name rename algorithm across all output surfaces.
6. Binder tail: lateral aliases, u(x,y), NATURAL JOIN, main. qualifier,
   mixed-type comparison casts, NULL-op-NULL typing.
7. Census + bench parity + ticket close-out.
