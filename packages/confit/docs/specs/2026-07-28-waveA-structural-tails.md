# Wave A: structural tails — census, pins, and design (TASK-56)

Measured against DuckDB 1.5.5, 2026-07-28. Raw pins: `pins-waveA/*.json`
(one file per fleet agent, verbatim queries + raw results). Census of all
167 corpus non-matches: `pins-waveA/census-all-nonmatches.json`.

## Why this wave replaced "wave C lists/structs"

The case-level census dissolved the "~25 lists/structs cases" estimate:
the true nested pool is 6 servable STRUCT cases + 1 LIST case that is
double-blocked by a self-join (stage B), and ZERO corpus cases use
`regexp_extract_all`/`regexp_split_to_array`. The rest of the old bucket
was BIT casts (13, mostly narrow-width casts we deliberately reject),
TIMESTAMP columns (3), HUGEINT (2), and table functions. Approved re-cut
("A makes sense"): ~17 servable cases with NO new runtime types.

## 1. Lazy non-scalar rejection (2 cases) — no pins needed, our contract

Row-model columns whose types have no scalar lane become OPAQUE entries
(model position + name) instead of construction errors. Referencing one —
bare, qualified, via `*`/`COLUMNS` expansion that keeps it, or via a
column-list alias rename that reaches its position — raises the same
`unsupported: row column 'x' has a non-scalar type`. EXCLUDE, name
filters, and REPLACE remove them (REPLACE gives the position a real lane,
which DuckDB also serves). Star expansion interleaves opaque names back
into MODEL order so positions and EXCLUDE semantics stay exact.

## 2. Structs-as-lanes (6 cases) — pins: struct-star.json, struct-nested.json

**Design: struct row columns flatten to scalar LANES at build time.** A
`STRUCT(i INT, j INT)` column `a` contributes lanes for leaf paths a.i,
a.j; the binder knows the tree shape; IR/exec/lower are untouched. NULL
structs materialize as NULL in every leaf lane at ingest, which makes
DuckDB's NULL-propagation pins fall out for free.

Measured rules the binder must reproduce:
- `a.*` expands IN PLACE to bare field names in declaration order; NULL
  struct row → NULL per field; identical to `SELECT a.i, a.j`.
- EXCLUDE/REPLACE on struct-star match fields case-insensitively — EVEN
  double-quoted (`EXCLUDE("J")` removes j). REPLACE output name takes the
  alias's exact case; REPLACE exprs may reference other table columns.
  Unknown names: `Column "z" in EXCLUDE list not found in a`. Excluding
  every field is legal if other select items remain; an empty select list
  is `SELECT list is empty after resolving * expressions!`.
- **A table alias with the same name beats the struct column** (a.* on
  `FROM t AS a` is a TABLE star, silently). Our check order (tables
  first) reproduces this.
- Multi-part resolution is longest-qualifier-first WITH BACKTRACKING:
  try (schema.table).column, then (table|alias).column, then bare column;
  commit to the longest prefix whose COLUMN binds; remaining parts become
  field extractions. Backtracking happens on column-bind failure (r.s
  with table r, column r STRUCT(s) → column r, field s); a bad FIELD
  after a committed column is a hard error (`Cannot extract field ...
  because it is not a struct...`). All matching case-insensitive at every
  position, quoting included. Output name of a dot chain = last part as
  written.
- An alias hides table AND schema.table paths, which can silently change
  a reference's meaning (t.t under `AS z` = column.field, not
  table.column) — falls out of the same resolution order.
- Whole-struct VALUES stay unsupported (non-scalar output): `SELECT a`,
  prefixes like case-2 `t.t.t` (len ≤3 = the bare column). Named error.
- Descoped, named rejections: bracket field access `t['t']` (output
  naming = full expression text; no corpus cases), `(expr).field` (same),
  structs containing unmappable leaf types (leaf becomes opaque).
- Python surface: struct inputs are NESTED pydantic models (replay builds
  them from arrow `struct<...>` types); DuckDB materializes structs as
  dicts.

## 3. FROM-position colon alias (4 cases) — pins: from-colon-alias.json

`FROM b : a` ≡ `FROM a AS b` in EVERY probed behavior (shadowing,
duplicate-alias laxity, binder errors). Left = alias, single bare/quoted
identifier only; right = table ref (may be schema-qualified). Whitespace
around `:` irrelevant. Not combinable with postfix alias or chained; no
column lists (`u(q) : a` parse error; `u : a(q)` is a table-FUNCTION
call). **Implementation: token pre-rewrite `x : T` → `T AS x`** in FROM
context (after FROM/JOIN/comma), like the wave-5 select-item colon
rewrite. Right sides we don't serve (subqueries, table functions) fall
through to the existing clean rejections.

## 4. reverse() (3 cases) — pins: reverse-graphemes.json

**DuckDB reverse() has TWO paths** (the pin that justifies pins-first):
- All-ASCII string → BYTE reverse. This SPLITS CRLF (`'a\r\nb'` →
  `'b\n\ra'`), violating UAX-29 — a pure grapheme implementation would be
  wrong on ASCII inputs.
- Any non-ASCII char → UAX-29 EXTENDED grapheme cluster reverse,
  byte-preserving per cluster, zero normalization. Verified: RI pairs
  greedy-from-left (odd counts leave a trailing singleton), ZWJ
  emoji/keycap/VS sequences hold, ZWJ travels with the PRECEDING char,
  Hangul LVT jamo stay one cluster (not composed), CRLF is one cluster,
  marks after CR/LF/controls stand alone, U+0600 Prepend binds forward
  (proving extended, not legacy).

Rust: `if s.is_ascii() { reverse bytes } else {
UnicodeSegmentation::graphemes(s, true).rev().collect() }` — the
`unicode-segmentation` crate implements exactly UAX-29 extended clusters.
`reverse(NULL)` → NULL. Non-VARCHAR args are BINDER errors (no implicit
cast; sole overload `reverse(VARCHAR)`). Lifting the wave-3 descope:
remove the limitations row + flip its twin test in the same commit.

## 5. COLUMNS(* REPLACE ...) + paren-less star modifiers (3 cases) — pins: columns-replace.json

- `COLUMNS(* <modifiers>)` as a BARE select item ≡ `* <modifiers>`
  (names, order, values — measured, NATURAL JOIN order included).
  Implementation: serve the star-argument form inside COLUMNS by routing
  through the same star expansion. `COLUMNS('re' REPLACE ...)` is a
  DuckDB PARSER error — keep rejecting.
- Wrapped `f(COLUMNS(...))` stays unsupported (naming pin recorded:
  replaced columns render as `f(a := (a + 10))`, bare ones keep bare
  names — for a future wave).
- **Paren-less `* REPLACE e AS c` consumes exactly ONE item**: a
  following comma starts a NEW select item (duplicate names and all —
  measured `* REPLACE i+100 AS i, j+1 AS j` → 3 columns i,j,j).
  Implementation: token rewrite wrapping the single item in parens,
  stopping at top-level comma / FROM / star-filter keyword / EOF.
  sqlparser already handles paren-less EXCLUDE/RENAME singles.
- `* REPLACE ... LIKE 'p'` parses in DuckDB but binder-errors ("Replace
  list cannot be combined with a filtering operation") — we already
  produce exactly this error class for filter+replace.

## 6. NULL constant regex patterns (1 case) — pins: regex-null-pattern.json

- NULL pattern (bare NULL or `CAST(NULL AS STRING)` — any constant-folded
  NULL) is NEVER a bind error: matches/full_match/SIMILAR TO/`~`/`!~` →
  NULL BOOLEAN per row; replace/extract → NULL VARCHAR. Constant path ==
  row path. We already serve bare-NULL literals; the fix is recognizing
  FOLDED NULL constants in `regex_pattern`.
- NULL replacement string → NULL (same recognition in the rewrite arg).
- Re-verified: NULL OPTIONS raise `Regex options field must not be NULL`
  for matches/full_match/extract (even with zero rows) and only
  regexp_replace returns NULL — matches our wave-B contract.
- NOT lifted (stay named rejections, no corpus cases): NULL group index
  in regexp_extract (DuckDB: '' for non-NULL subject — behaves like a
  missing group, NOT null propagation; a wrong lift would be silent), and
  star-filter `* SIMILAR TO CAST(NULL AS STRING)` (DuckDB itself
  binder-errors "must be a constant").

## Corpus arithmetic

511 serving before the wave. Expected flips: struct 6, FROM colon 4,
reverse 3, lazy non-scalar 1 ([162]; [161] is aggregation), COLUMNS(*
REPLACE) 2, paren-less REPLACE 1, NULL regex pattern 1 → **target ≥ 528**
of 678, zero FAILs, everything else clean-unsupported.
