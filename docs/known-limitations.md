# Known limitations — deliberate, named, and loud

This is the user-facing contract of Confit (`DuckDBInferFn`):
what it refuses to serve, **why**, and what you see when you hit a limit.
Its executable twin is [`packages/confit/tests/test_known_limitations.py`](../tests/test_known_limitations.py) —
every limitation below is asserted there, so lifting one breaks a test and
forces this document to change with it.

**The contract.** For any SQL you hand it, the engine does exactly one of:

1. **Serve it bit-for-bit identical to DuckDB** (verified continuously
   against DuckDB's own test corpus: 550 of 678 statements as of stage B), or
2. **Refuse loudly at BUILD time** — `DuckDBInferFn(...)` raises a
   `ValueError` naming the construct. Nothing is ever silently wrong or
   silently dropped at inference time.

There is no third mode. Every limitation here is a *measured decision*
recorded in a pins spec (`docs/superpowers/specs/`), not an accident.

**Which DuckDB** (decided 2026-08-17, and it has a user-visible cost — see
§5). "Identical to DuckDB" means DuckDB with its query optimizer off:

```sql
PRAGMA disable_optimizer;
```

That is not a smaller DuckDB. The binder is untouched, so types, constant
folding and bind-time errors are the same; execution-level laziness — an
untaken `CASE` arm, `AND`/`OR` short-circuit in a filter — is the same. What
it removes is the 33 plan-rewrite passes.

The reason is that the optimizer-on reading is not a function of the query.
`statistics_propagation` decides from a column's stored null statistic, so
DuckDB answers the *same query over the same rows* differently depending on
the table's insert history — measured: a table built as `[-128, NULL]` and
then having the NULL deleted answers differently from one built as `[-128]`,
with identical contents. Confit compiles once against a *schema* and serves
many batches; it never sees a table, let alone its history. A target you
cannot compute from the query is not a target.

**What "identical" means for ROW ORDER** (TASK-129, 2026-08-19). Values are
bit-for-bit. A *sequence* is only promised where one is defined: on the row
path by the serving contract (output rows follow input rows -- `map` exactly,
`filter` as a subsequence, `many` as per-input-row blocks in input order,
join order within a block being the documented multiset), and on a
static-tables-only result by a total `ORDER BY`. Anywhere else SQL defines no
order and neither do we -- DuckDB itself returns the same unordered
`GROUP BY` in twelve different row orders over twelve connections (measured).
The campaign fuzzer compares per this rule: sequence-strict self-legs on the
row path (a reversed batch must reverse), sortedness-plus-multiset under
`ORDER BY` (ties are free), multiset otherwise.

**The break is CLOSED** (TASK-124, fixed 2026-08-19). Boolean short-circuit
is now decided per CONTEXT, as DuckDB decides it: selection context (the
WHERE root and every `CASE WHEN` condition, projections included) makes
`AND` lazy left-to-right and `OR` its exact dual, and exits to eager value
context at `NOT`/`IS NULL`/comparisons/function arguments and CASE arms. The
model, its measurements and the design are in
`docs/superpowers/specs/2026-08-19-selection-context-design.md`; the full
measured matrix runs against the live oracle in
`known_divergences/test_short_circuit.py`.

---

## 1. The specialization bargain (inherent to the engine model)

Confit's speed comes from doing ALL general work at build time:
parse once, bind once, compile once, freeze the static tables into the
code. Anything that would require re-doing general work per row is
rejected, permanently by design:

| Limitation | You'll see | Why |
|---|---|---|
| Regex patterns must be constants (`regexp_matches(s, pattern_col)` rejects) | `unsupported: non-constant regex pattern (compiled at prepare in v0)` | Regexes compile at prepare; DuckDB compiles per row. Per-row compilation is the opposite of specialization. |
| Replacement strings / regex options / extract group indexes must be constants | `non-constant regexp_replace replacement` etc. | Same. |
| Static (join) tables must be provided at build time; under the DEFAULT shapes their keys must be unique | `duplicate map key` | Joins are frozen hash maps baked into the function. Duplicate keys mean 1:N join multiplicity, which SERVES under the opt-in `shape='many'` (TASK-59) — along with cross joins, inequality `ON` predicates, and constant `ON` clauses — with multiset parity vs DuckDB (its join output order is a measured hash-join accident; the engine emits probe order outer, build insertion order inner). Under `filter`/`map` the 1:1 contract is unchanged. NULL *values* serve since TASK-55; NULL *keys* never match. |
| Self-joins require `shape='many'` (and the ON form) | `joining the dynamic table to itself` | Under `'many'` the batch itself becomes the build side (assembled per call, the ON as a per-pair residual) — comma/cross and `ON` self-joins serve with multiset parity. `USING`/`NATURAL` self-joins stay a named rejection (follow-up); the default shapes keep the original error. |
| Exactly one row table drives the query | `the specializer takes exactly one row table`, `must be the dynamic table` | The serving contract is rows-in → rows-out for one entity stream. |

## 2. Out of scope for row-serving (by decision, not difficulty)

**The row-shape contract** (TASK-58): `DuckDBInferFn(..., shape=...)`
declares how many output rows each input row may produce, checked at
build time. `"filter"` (the default) is the engine's native 0..1;
`"map"` statically PROVES exactly-one (`out[i] ↔ in[i]`, the strict
serving guarantee) by rejecting anything that can drop a row — a WHERE
clause, an INNER join (key misses drop), a static-tables-only constant
query; `"many"` (0..N) is the multiplicity opt-in (stage B): duplicate-key
joins, cross joins, and inequality/constant `ON` joins build ONLY under
it (one join per query for now, named restriction) — multiplicity can
never sneak into a serving path by default. Comma and `ON` self-joins
serve under `'many'` too; `USING`/`NATURAL` self-joins are a named
follow-up rejection.

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

The exception is a **static-tables-only query** (nothing dynamic remains):
it is evaluated once at build by DuckDB itself and frozen, so aggregation,
`ORDER BY` and DuckDB dialect beyond sqlparser all serve there. One carve-out
(TASK-128, decided 2026-08-19): a **row limit refuses** — `LIMIT`, `OFFSET`,
`FETCH`, `TOP`, anywhere in the statement, `ORDER BY` or not. Which rows
survive a limit is not a function of the query: measured, the same
`GROUP BY … FETCH FIRST 1 ROWS ONLY` over the same four rows answered
**four different ways across twelve fresh connections**, and `ORDER BY` does
not fix ties (a tie fed from a `GROUP BY` flipped in 20 runs). Freezing
whichever answer the build-time run happened to get would make two builds of
the same function disagree with each other. You'll see:
`row limit (LIMIT/OFFSET) on a static-tables-only query`.

## 3. Type-system boundaries

The engine computes in exactly four types: `i64`, `f64`, UTF-8 string,
bool. Measured consequences:

- **f32 base tables reject** (`engine is f64-only`).
- **A static column is served at its declared arrow type or refused by
  name** — never widened into a neighbouring lane. Corrected 2026-08-15:
  the catalogue used to widen `float32` and the unsigned widths, and both
  diverged silently. `float32` in value AND type (`s.v * 3.0` over `0.1`
  is `0.30000001192092896`/FLOAT on DuckDB; f64 arithmetic here gave
  `0.30000000447034836`/DOUBLE), unsigned in type (`uint64` stays UINT64
  there, became int64 here). The row path had always refused both. The two
  exceptions that stay served are measured equivalent, not convenient:
  `large_string`/`utf8` (DuckDB normalises them to VARCHAR) and the
  decimal tiers (below).
- **DECIMAL static columns serve EXACTLY** (since TASK-91): the payload is
  the scaled integer in an i128 lane from ingest through the join, emitted
  as `decimal128(p,s)`. `2^53+1` comes back as itself, and so does
  `2^63+1` — an ordinary fit-time `sum(BIGINT)` produces that, which is
  why the lane is i128 and not i64. `decimal32`/`decimal64` inputs
  normalise to `decimal128(p,s)` output because DuckDB exports every tier
  as 128-bit arrow. What still refuses, by name: EXPRESSIONS over a
  decimal — arithmetic, `CAST` to anything but `DOUBLE`, and
  `COALESCE`/`CASE`/`greatest` unifying it with a non-identical type (m-8
  lattice phase 5; each of these used to serve a silently wrong double).
  Comparisons, joins, `CAST(d AS DOUBLE)` and `SELECT *` all serve.
  `decimal256` statics refuse (DuckDB itself refuses them at arrow
  register, at any precision), and decimal ROW columns stay opaque.
- **Structs of scalars SERVE** (since TASK-56): struct row columns are
  flattened to scalar lanes at build time — field access (`a.i`, deep
  `t.t.t.t` paths) and struct-star (`a.*` incl. EXCLUDE/REPLACE) are
  bit-identical to DuckDB. What still rejects, by name: the struct as a
  WHOLE value (`SELECT a` — non-scalar output), bracket field access
  (`a['i']` — DuckDB names such outputs by full expression text, not
  modeled), and struct fields whose own types are non-scalar.
- **Lists reject** (`row column 'x' has a non-scalar type`) — list types
  are out of the row-schema vocabulary and stay opaque: unreferenced
  (star expansion included) they cost nothing to declare; since TASK-56
  an unreferenced timestamp/list field no longer blocks a scalar-only
  query, and `EXCLUDE`/name filters/`REPLACE` can remove one from a
  star. Referenced, they refuse by name. Lists also still gate
  `regexp_extract_all` / `regexp_split_to_array` / the STRUCT form of
  `regexp_extract` (`list-valued — non-scalar in v0`).
- **DECIMAL literals are f64** — a documented divergence: DuckDB types
  `1.5` as `DECIMAL(2,1)` and does decimal arithmetic; we map to f64.
  Values agree on every corpus case; exact-decimal accumulation semantics
  are not reproduced. One visible consequence: `CAST(-2.5 AS BIGINT)` on
  the bare literal is a DECIMAL cast in DuckDB (half away from zero, `-3`)
  and a DOUBLE cast for us (half to even, `-2`). Casting a DOUBLE *column*
  agrees exactly — measure DOUBLE cast behaviour with a DOUBLE column or an
  explicit `::DOUBLE`, never with a literal, or you will pin the wrong
  rounding mode (TASK-70 did).
- Integer widths: the engine TYPES in DuckDB's lattice (TINYINT..BIGINT —
  literals are INTEGER by magnitude, `::SMALLINT` is real, `ascii` returns
  INTEGER, `infer_arrow` emits int8/int16/int32 from the type) but
  COMPUTES in two machine widths, i64 and f64. The width is observable
  exactly where DuckDB's is: the Arrow schema (shipped, TASK-79/m-8
  phase 2, catalogue pinned in `test_integer_widths.py`) and the overflow
  trap threshold — the trap half is m-8 phase 3, so until it lands a
  narrow lane that overflows serves the i64 value on the row path and
  refuses by name at the `infer_arrow` boundary; every input this refuses
  is one DuckDB itself errors on. HUGEINT and the unsigned family are not
  served at all — they refuse by name rather than collapse to i64 (see the
  static-column entry above); serving them is the i128 lane, whose
  cranelift dependency was verified GO on 2026-08-15 (TASK-100).

## 4. Semantics descoped after measurement

Each of these was measured against DuckDB 1.5.5 first (pins in
`docs/superpowers/specs/`), and rejected because serving it would risk a
wrong answer or require semantics we can't reproduce exactly:

| Construct | Why it's descoped |
|---|---|
| `^` operator | It IS pow in DuckDB, but sqlparser's precedence differs from DuckDB's (`2*x^y` would parse as `(2*x)^y`). Mapping it computes the wrong tree silently. Use `pow()`. |
| prefix `~`, `#`, `NOT GLOB` | Same class: precedence/parse divergences that would silently mis-associate. `xor()` covers bit-xor; `NOT (x GLOB p)` works. |
| Regex reject list: `\B`, `\Q…\E`, `(?<name>…)`, duplicate group names, bounds > 1000, stacked quantifiers (`a*+`), `\u` escapes, negated Perl classes inside `[...]` | The RE2↔rust-regex differential battery (98 entries) proved these are the constructs where the engines disagree or DuckDB itself is broken (`\B` crashes DuckDB at runtime on non-ASCII). Everything else is byte-identical. |
| Fuzzer-found regex rejects (TASK-54): `\1`–`\9` backrefs outside classes, the full stacked-quantifier grammar (`{2}*`, `?*`, `a???` — one lazy `?` is the only legal follower), nested repetition products > 1000, whitespace inside `{m, n}` bounds, class set-op lookalikes (`--`/`&&`/`~~`), non-POSIX `[` inside classes, Perl-class range endpoints (`[a-\d]`), capturing `(x){0}`, anchor-only multi-anchor patterns (DuckDB is SELF-inconsistent on these — its row path disagrees with its own constant fold), `$` anchors in non-final position (`'$hello'` — DuckDB's row path literal-optimizes the leading `$`+literal into a PREFIX match, matching "hello world", while its own constant fold matches normally; found by the standing fuzzer on seed 20260728), and counted repetitions over RE2's PROGRAM-SIZE budget (`(\p{L}){1,500}` is "pattern too large" in DuckDB while rust-regex serves it — rejected via a one-sided weight estimate that always fires before DuckDB's real budget; same seed, pins `pins-waveB/fuzzer-20260728.json`) | The standing differential fuzzer (`packages/confit/tests/test_duckdb_regexp_fuzz.py`, in the normal gate) found these 12 classes in its first 3k-case deep run — each one a silent-wrong-answer risk in rust-regex — then re-swept to ZERO divergences over 40k cases across 8 seeds. Pins: `pins-waveB/fuzzer-task54.json`. |
| `SIMILAR TO ... ESCAPE` | Not implemented in DuckDB itself. |
| `* EXCLUDE (t.key)` on a USING join | DuckDB UNMERGES the coalesced column (it reappears at the right table's position) — measured, not modeled. Unqualified EXCLUDE works. |
| `BETWEEN`/`IN` mixing non-numeric string literals with numbers | DuckDB converts at EXECUTION time (an empty input succeeds!); a bind-time conversion was measured to be over-eager. Numeric literals convert fine. |
| `COLUMNS(...)` inside expressions, lambda/list forms | Only bare `COLUMNS('re')` / `COLUMNS(*)` as select items are served. |
| Pad/repeat counts past the 1 GiB string-builder budget (TASK-88) | A LITERAL count that can exceed the budget refuses at build; a DATA-DRIVEN count (column, `CAST(k AS INTEGER)`) keeps the runtime cap — the engine traps at 1 GiB where DuckDB's own behaviour is spelling-dependent (a multi-GB string or its own builder error). No gigabyte allocations in a serving engine, by decision. |
| `CASE` where every branch is NULL, `COALESCE`/`least`/`greatest` of only NULLs | DuckDB binds the all-NULL FAMILY forms (SQLNULL → INTEGER); the engine still refuses them. A bare `NULL` select item, `nullif(NULL, x)`, and `NULL <op> NULL` all SERVE with DuckDB's types since m-8 phase 2 (bare NULL is int32, its INTEGER). |
| Bare `NULL` as `repeat`'s string (TASK-86, BLOB face) | DuckDB picks the **BLOB** overload — a type this engine doesn't have until the m-8 Blob phase — so adopting VARCHAR answered with a different schema. Spell it `CAST(NULL AS VARCHAR)`, which both engines type identically. Adopters that agree with DuckDB — `upper(NULL)`, `coalesce(NULL, x)`, `nullif(x, NULL)`, `nullif(NULL, x)` (int32, m-8 phase 2), a NULL `repeat` count — keep serving, pinned schema-equal. |

## 5. Deliberate contract choices (behavior differs from raw DuckDB surface)

These are served, but with a consciously chosen surface — know them:

- **Duplicate output column names are renamed**, using DuckDB's own
  boundary-rename algorithm (`id, id, id_1` → `id, id_1, id_1_1`;
  case-insensitive collision check). Raw DuckDB keeps duplicates at the
  top level, but a dict cannot — and this rename is
  bit-identical to what DuckDB itself does at every subquery/CTE/CTAS
  boundary and in `.df()`. Verified against `.df()` in tests.
- **Error TEXTS are approximate where noted.** Runtime traps
  (overflow, shifts, substring range) reproduce DuckDB's message bodies
  verbatim; some bind-time rejections (star-filter zero-match, regex
  compile errors) use our own wording with the same error class. The
  corpus only ever compares successful results, so texts never affect
  parity.
- **Two known oracle divergences** (excluded from the corpus by name in
  `packages/confit/tests/test_corpus_replay.py::_KNOWN_DIVERGENT_SOURCES`): DuckDB
  behaviors that depend on column STATISTICS (e.g. ILIKE's NUL handling
  selects a different kernel depending on *sibling rows*). A row-at-a-time
  engine cannot reproduce statistics-dependent semantics even in
  principle; the engine is NUL-transparent (the ASCII-kernel behavior).
- **A trapping subexpression DuckDB's OPTIMIZER deletes, we still
  evaluate** — the standing cost of running the oracle with the optimizer
  off (see "Which DuckDB" above). This is the one place where a query you
  can run in your own DuckDB session may raise here:

  ```sql
  SELECT (i + 1) > 5 FROM t              -- i INTEGER = 2147483647
  -- your DuckDB (optimizer on): true   -- it rewrites this to i > 4
  -- confit:                     Out of Range Error, overflow in INT32
  ```

  The shape is always the same: a subexpression that would trap, in a
  position where a plan rewrite removes it before it ever runs. The passes
  measured doing this are `expression_rewriter` (constant shifting, folding
  a trapping constant, dead-range elimination) and
  `statistics_propagation` (proving `IS NOT NULL` from a column's null
  statistic, and pruning a filter from a value range). A 4000-seed
  differential campaign puts it at 8 seeds in 28 findings; all eight are
  labelled `DIVERGE_OPT` by the campaign and enumerated in
  `docs/2026-08-17-fuzz-triage.md`.

  The trade is deliberate: matching the optimizer means matching an
  undocumented moving target that is not a function of the query, and in
  the other direction it would mean *serving* where your DuckDB raises.
  This way the divergence is always a loud trap or refusal, never a
  different served value. If it bites you, `PRAGMA disable_optimizer` in
  your DuckDB session reproduces exactly what confit does.
- **`%`-by-zero NaN bit pattern is platform-libm** — pinned as
  engine==oracle bit agreement per platform, not a constant.
- **Schema qualifiers are registry-noise** (TASK-55): the engine's table
  registry is schema-less, so `s1.t1` (and 3-part `s1.t1.col` refs)
  resolve when the table part matches a registered bare name. DuckDB's
  schema-existence errors (`schema "x" does not exist`) are not
  reproduced — a schema-less registry cannot know which schemas would
  exist. Ambiguous matches still error. With struct paths (TASK-56) the
  same rule extends to n-part references: resolution is
  longest-qualifier-first with backtracking (measured), and any first
  part is accepted as a schema when the second matches the table — so
  `w.w.w` on a table `w` with struct column `w` binds the LONGER
  schema-ish parse (a whole-struct rejection) where schema-aware DuckDB
  would fall through to `column.field`. The divergence is always a loud
  build-time rejection, never a different served value.

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

Four mechanisms, all in the normal test gate:

1. **The corpus replay** (678 statements mined from DuckDB's test suite):
   every statement must match bit-for-bit, reject cleanly, or be a named
   divergence — a wrong answer anywhere fails the gate.
2. **The executable twin** (`packages/confit/tests/test_known_limitations.py`): every
   limitation in this document is asserted; lifting one breaks a test.
3. **The standing differential fuzzer** (`packages/confit/tests/test_duckdb_regexp_fuzz.py`):
   randomized DuckDB-vs-engine sweeps of the regex surface on every run
   (seed/size overridable for deep runs) — new divergences fail with the
   reproducing seed and SQL, and their fix lands as a reject-list entry
   plus a row in this document.
4. **The campaign fuzzer reads DuckDB TWICE** (`packages/confit/fuzz/`),
   once with the optimizer off and once on, so a finding says which kind it
   is instead of needing a human to reason about it: a disagreement with the
   optimizer-off reading is a bug, a disagreement only with the optimizer-on
   one is the §5 cost above, and an agreement with optimizer-on *against*
   the oracle means the engine is reproducing a pass it should not — that
   last category is reported as a bug and is currently empty.
