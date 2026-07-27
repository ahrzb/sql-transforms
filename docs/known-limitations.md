# Known limitations — deliberate, named, and loud

This is the user-facing contract of the SQL specializer (`DuckDBInferFn`):
what it refuses to serve, **why**, and what you see when you hit a limit.
Its executable twin is [`tests/test_known_limitations.py`](../tests/test_known_limitations.py) —
every limitation below is asserted there, so lifting one breaks a test and
forces this document to change with it.

**The contract.** For any SQL you hand it, the engine does exactly one of:

1. **Serve it bit-for-bit identical to DuckDB** (verified continuously
   against DuckDB's own test corpus: 505 of 678 statements as of wave B), or
2. **Refuse loudly at BUILD time** — `DuckDBInferFn(...)` raises a
   `ValueError` naming the construct. Nothing is ever silently wrong or
   silently dropped at inference time.

There is no third mode. Every limitation here is a *measured decision*
recorded in a pins spec (`docs/superpowers/specs/`), not an accident.

---

## 1. The specialization bargain (inherent to the engine model)

The specializer's speed comes from doing ALL general work at build time:
parse once, bind once, compile once, freeze the static tables into the
code. Anything that would require re-doing general work per row is
rejected, permanently by design:

| Limitation | You'll see | Why |
|---|---|---|
| Regex patterns must be constants (`regexp_matches(s, pattern_col)` rejects) | `unsupported: non-constant regex pattern (compiled at prepare in v0)` | Regexes compile at prepare; DuckDB compiles per row. Per-row compilation is the opposite of specialization. |
| Replacement strings / regex options / extract group indexes must be constants | `non-constant regexp_replace replacement` etc. | Same. |
| Static (join) tables must be provided at build time, with unique keys | `duplicate map key` | Joins are frozen hash maps baked into the function. Duplicate keys mean 1:N join multiplicity — a designed extension (stage B, TASK-50 notes) that is deliberately not built yet. NULL *values* serve since TASK-55 (they flow through as NULL); NULL *keys* drop the row, matching equi-join semantics. |
| The dynamic table cannot be joined to itself | `joining the dynamic table to itself` | The batch is the probe side; using it as a build side too needs stage-B machinery. |
| Exactly one row table drives the query | `the specializer takes exactly one row table`, `must be the dynamic table` | The serving contract is rows-in → rows-out for one entity stream. |

## 2. Out of scope for row-serving (by decision, not difficulty)

The engine serves **row-at-a-time feature transforms**. Whole-relation
constructs are out of scope because their output shape is not
one-row-in/one-row-out:

- Aggregation: `GROUP BY`, `HAVING`, `sum`/`count`/`avg`/... →
  `aggregate function ... (no aggregation in v0)`
- `ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT` — row-independent transforms
  don't reorder or deduplicate.
- CTEs (`WITH`), `UNION`/`INTERSECT`/`EXCEPT`, subqueries, multiple
  statements.
- Table functions in FROM (`range(...)`) — there is no base table.
- `rowid` pseudo-column — rows have no stable identity in a stream.
- `FULL OUTER JOIN` — emits rows that no input row produced.

## 3. Type-system boundaries

The engine computes in exactly four types: `i64`, `f64`, UTF-8 string,
bool. Measured consequences:

- **f32 base tables reject** (`engine is f64-only`).
- **Static-table key/value columns must fit BIGINT** — `UBIGINT`/`HUGEINT`
  payloads outside i64 reject with a named message.
- **Lists and structs reject** (`row column 'x' has a non-scalar type`) —
  this is the designated next capability wave ("wave C"), and it also gates
  `regexp_extract_all` / `regexp_split_to_array` / the STRUCT form of
  `regexp_extract` (`list-valued — non-scalar in v0`).
- **DECIMAL literals are f64** — a documented divergence: DuckDB types
  `1.5` as `DECIMAL(2,1)` and does decimal arithmetic; we map to f64.
  Values agree on every corpus case; exact-decimal accumulation semantics
  are not reproduced.
- Narrow integer widths don't exist: bitwise ops compute in i64, which
  matches DuckDB whenever either operand is BIGINT (always true for
  row-model ints). Explicit narrow CASTs are rejected rather than
  risking DuckDB's narrow-width overflow behavior.

## 4. Semantics descoped after measurement

Each of these was measured against DuckDB 1.5.5 first (pins in
`docs/superpowers/specs/`), and rejected because serving it would risk a
wrong answer or require semantics we can't reproduce exactly:

| Construct | Why it's descoped |
|---|---|
| `reverse()` | DuckDB reverses UAX-29 grapheme clusters (incl. regional-indicator pairing), not codepoints. A codepoint reverse would be silently wrong on emoji/accents. |
| `^` operator | It IS pow in DuckDB, but sqlparser's precedence differs from DuckDB's (`2*x^y` would parse as `(2*x)^y`). Mapping it computes the wrong tree silently. Use `pow()`. |
| prefix `~`, `#`, `NOT GLOB` | Same class: precedence/parse divergences that would silently mis-associate. `xor()` covers bit-xor; `NOT (x GLOB p)` works. |
| Regex reject list: `\B`, `\Q…\E`, `(?<name>…)`, duplicate group names, bounds > 1000, stacked quantifiers (`a*+`), `\u` escapes, negated Perl classes inside `[...]` | The RE2↔rust-regex differential battery (98 entries) proved these are the constructs where the engines disagree or DuckDB itself is broken (`\B` crashes DuckDB at runtime on non-ASCII). Everything else is byte-identical. |
| Fuzzer-found regex rejects (TASK-54): `\1`–`\9` backrefs outside classes, the full stacked-quantifier grammar (`{2}*`, `?*`, `a???` — one lazy `?` is the only legal follower), nested repetition products > 1000, whitespace inside `{m, n}` bounds, class set-op lookalikes (`--`/`&&`/`~~`), non-POSIX `[` inside classes, Perl-class range endpoints (`[a-\d]`), capturing `(x){0}`, anchor-only multi-anchor patterns (DuckDB is SELF-inconsistent on these — its row path disagrees with its own constant fold) | The standing differential fuzzer (`tests/test_duckdb_regexp_fuzz.py`, in the normal gate) found these 12 classes in its first 3k-case deep run — each one a silent-wrong-answer risk in rust-regex — then re-swept to ZERO divergences over 40k cases across 8 seeds. Pins: `pins-waveB/fuzzer-task54.json`. |
| `SIMILAR TO ... ESCAPE` | Not implemented in DuckDB itself. |
| `* EXCLUDE (t.key)` on a USING join | DuckDB UNMERGES the coalesced column (it reappears at the right table's position) — measured, not modeled. Unqualified EXCLUDE works. |
| `BETWEEN`/`IN` mixing non-numeric string literals with numbers | DuckDB converts at EXECUTION time (an empty input succeeds!); a bind-time conversion was measured to be over-eager. Numeric literals convert fine. |
| `COLUMNS(...)` inside expressions, lambda/list forms | Only bare `COLUMNS('re')` / `COLUMNS(*)` as select items are served. |
| Bare `NULL` with no typing context, `CASE` where every branch is NULL, `COALESCE`/`NULLIF` of only NULLs | DuckDB's SQLNULL type has no i64/f64/str/bool home. `NULL <op> NULL` IS served with the measured result types. |

## 5. Deliberate contract choices (behavior differs from raw DuckDB surface)

These are served, but with a consciously chosen surface — know them:

- **Duplicate output column names are renamed**, using DuckDB's own
  boundary-rename algorithm (`id, id, id_1` → `id, id_1, id_1_1`;
  case-insensitive collision check). Raw DuckDB keeps duplicates at the
  top level, but a pydantic model or dict cannot — and this rename is
  bit-identical to what DuckDB itself does at every subquery/CTE/CTAS
  boundary and in `.df()`. Verified against `.df()` in tests.
- **`NULL || NULL` types as VARCHAR** in the output model (value is NULL
  either way; DuckDB's SQLNULL materializes as INTEGER).
- **Error TEXTS are approximate where noted.** Runtime traps
  (overflow, shifts, substring range) reproduce DuckDB's message bodies
  verbatim; some bind-time rejections (star-filter zero-match, regex
  compile errors) use our own wording with the same error class. The
  corpus only ever compares successful results, so texts never affect
  parity.
- **Two known oracle divergences** (excluded from the corpus by name in
  `tests/test_corpus_replay.py::_KNOWN_DIVERGENT_SOURCES`): DuckDB
  behaviors that depend on column STATISTICS (e.g. ILIKE's NUL handling
  selects a different kernel depending on *sibling rows*). A row-at-a-time
  engine cannot reproduce statistics-dependent semantics even in
  principle; the engine is NUL-transparent (the ASCII-kernel behavior).
- **`%`-by-zero NaN bit pattern is platform-libm** — pinned as
  engine==oracle bit agreement per platform, not a constant.
- **Schema qualifiers are registry-noise** (TASK-55): the engine's table
  registry is schema-less, so `s1.t1` (and 3-part `s1.t1.col` refs)
  resolve when the table part matches a registered bare name. DuckDB's
  schema-existence errors (`schema "x" does not exist`) are not
  reproduced — a schema-less registry cannot know which schemas would
  exist. Ambiguous matches still error.

## 6. How to read a rejection

Every rejection is a `ValueError` at `DuckDBInferFn(...)` construction
whose message starts with a classification:

- `unsupported: ...` — real SQL, deliberately not served (this document).
- `parse error: ...` — the dialect surface ends here.
- `bind error: ...` — the query is wrong against YOUR schema (typo,
  type mismatch), not a limitation.

If a message you hit isn't in this document or the tests, that's a bug in
our bookkeeping — file it.

## 7. How this document stays honest

Three mechanisms, all in the normal test gate:

1. **The corpus replay** (678 statements mined from DuckDB's test suite):
   every statement must match bit-for-bit, reject cleanly, or be a named
   divergence — a wrong answer anywhere fails the gate.
2. **The executable twin** (`tests/test_known_limitations.py`): every
   limitation in this document is asserted; lifting one breaks a test.
3. **The standing differential fuzzer** (`tests/test_duckdb_regexp_fuzz.py`):
   randomized DuckDB-vs-engine sweeps of the regex surface on every run
   (seed/size overridable for deep runs) — new divergences fail with the
   reproducing seed and SQL, and their fix lands as a reject-list entry
   plus a row in this document.
