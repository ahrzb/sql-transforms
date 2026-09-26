# Structural pins: non-scalar columns, structs, FROM colon alias, reverse, COLUMNS(*), NULL regex patterns

DuckDB 1.5.5. Raw pins: `pins-waveA/*.json` (verbatim queries + raw
results). Tests: `tests/test_duckdb_wavea_structural.py`.

## Non-scalar row columns (confit contract)

Row-model columns whose types have no scalar lane are OPAQUE entries (model
position + name), not construction errors. Referencing one — bare,
qualified, via `*`/`COLUMNS` expansion that keeps it, or via a column-list
alias rename that reaches its position — raises `unsupported: row column
'x' has a non-scalar type`. EXCLUDE, name filters, and REPLACE remove them
(REPLACE gives the position a real lane, which DuckDB also serves). Star
expansion interleaves opaque names back into MODEL order, so positions and
EXCLUDE semantics stay exact.

## Structs (`struct-star.json`, `struct-nested.json`)

confit flattens struct row columns to scalar LANES at build time: a
`STRUCT(i INT, j INT)` column `a` contributes lanes for the leaf paths
a.i, a.j; the binder knows the tree shape. NULL structs materialize as NULL
in every leaf lane at ingest, which yields DuckDB's NULL propagation.

Measured rules (confit reproduces them):

- `a.*` expands IN PLACE to bare field names in declaration order; a NULL
  struct row → NULL per field; identical to `SELECT a.i, a.j`.
- EXCLUDE/REPLACE on a struct star match fields case-insensitively — EVEN
  double-quoted (`EXCLUDE("J")` removes j). The REPLACE output name takes
  the alias's exact case; REPLACE exprs may reference other table columns.
  Unknown names: `Column "z" in EXCLUDE list not found in a`. Excluding
  every field is legal if other select items remain; an empty select list
  is `SELECT list is empty after resolving * expressions!`.
- **A table alias with the same name beats the struct column** (`a.*` on
  `FROM t AS a` is a TABLE star, silently).
- Multi-part resolution is longest-qualifier-first WITH BACKTRACKING: try
  (schema.table).column, then (table|alias).column, then bare column;
  commit to the longest prefix whose COLUMN binds; remaining parts become
  field extractions. Backtracking happens on column-bind failure (r.s with
  table r and column r STRUCT(s) → column r, field s); a bad FIELD after a
  committed column is a hard error (`Cannot extract field ... because it is
  not a struct...`). Matching is case-insensitive at every position,
  quoting included. The output name of a dot chain is its last part as
  written.
- An alias hides table AND schema.table paths, which can change a
  reference's meaning (`t.t` under `AS z` = column.field, not
  table.column).
- confit refuses whole-struct VALUES (non-scalar output): `SELECT a`, and
  prefixes like `t.t.t` that resolve to the bare struct column. It also
  refuses bracket subscripts on non-VARCHAR values (`t['t']`), and a
  struct leaf of an unmappable type is opaque.
- Python surface: struct inputs are NESTED pydantic models (built from
  arrow `struct<...>` types); DuckDB materializes structs as dicts.

## FROM-position colon alias (`from-colon-alias.json`)

`FROM b : a` ≡ `FROM a AS b` in every probed behavior (shadowing,
duplicate-alias laxity, binder errors). Left = alias, a single bare/quoted
identifier; right = table ref (may be schema-qualified). Whitespace around
`:` is irrelevant. Not combinable with a postfix alias or chained; no
column lists (`u(q) : a` is a parse error; `u : a(q)` is a table-FUNCTION
call). confit rewrites `x : T` → `T AS x` at the token level in FROM
context (`rewrite_from_colon_aliases` in `rewrite.rs`); right sides it does
not serve (subqueries, table functions) reach the existing refusals.

## reverse() (`reverse-graphemes.json`)

DuckDB reverse() has TWO paths:

- All-ASCII string → BYTE reverse. This SPLITS CRLF (`'a\r\nb'` →
  `'b\n\ra'`), violating UAX-29.
- Any non-ASCII char → UAX-29 EXTENDED grapheme cluster reverse,
  byte-preserving per cluster, no normalization. RI pairs group greedily
  from the left (odd counts leave a trailing singleton); ZWJ emoji/keycap/
  VS sequences hold; ZWJ travels with the PRECEDING char; Hangul LVT jamo
  stay one cluster (not composed); CRLF is one cluster; marks after
  CR/LF/controls stand alone; U+0600 Prepend binds forward (extended, not
  legacy, clusters).

confit: `if s.is_ascii() { reverse bytes } else {
graphemes(s, true).rev() }` using the `unicode-segmentation` crate (UAX-29
extended clusters). `reverse(NULL)` → NULL. Non-VARCHAR args are binder
errors (sole overload `reverse(VARCHAR)`).

## COLUMNS(* ...) and paren-less star modifiers (`columns-replace.json`)

- `COLUMNS(* <modifiers>)` as a BARE select item ≡ `* <modifiers>` (names,
  order, values; NATURAL JOIN order included). confit routes it through
  the same star expansion. `COLUMNS('re' REPLACE ...)` is a DuckDB PARSER
  error.
- Wrapped `f(COLUMNS(...))`: DuckDB renders replaced columns as
  `f(a := (a + 10))` and keeps bare names for the others; confit refuses
  it.
- **Paren-less `* REPLACE e AS c` consumes exactly ONE item**: a following
  comma starts a NEW select item (`* REPLACE i+100 AS i, j+1 AS j` → 3
  columns i, j, j). confit wraps the single item in parens at the token
  level (`rewrite_parenless_replace`), stopping at a top-level comma /
  FROM / star-filter keyword / EOF. sqlparser handles paren-less
  EXCLUDE/RENAME singles.
- `* REPLACE ... LIKE 'p'` parses in DuckDB but binder-errors ("Replace
  list cannot be combined with a filtering operation"); confit raises the
  same bind error.

## NULL constant regex patterns (`regex-null-pattern.json`)

- A NULL pattern (bare NULL or `CAST(NULL AS STRING)` — any constant-folded
  NULL) is never a bind error: matches/full_match/SIMILAR TO/`~`/`!~` →
  NULL BOOLEAN per row; replace/extract → NULL VARCHAR. Constant path ==
  row path. confit recognizes folded NULL constants as NULL patterns.
- A NULL replacement string → NULL.
- NULL OPTIONS raise `Regex options field must not be NULL` for
  matches/full_match/extract (even with zero rows); only regexp_replace
  returns NULL.
- A NULL group index in regexp_extract gives '' for a non-NULL subject (it
  behaves like a missing group, not NULL propagation); confit matches.
- `* SIMILAR TO CAST(NULL AS STRING)` is a DuckDB binder error ("must be a
  constant").
