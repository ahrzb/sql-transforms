# Known limitations — deliberate, named, and loud

This is the user-facing contract of Confit (`DuckDBInferFn`):
what it refuses to serve, **why**, and what you see when you hit a limit.
Its executable twin is [`packages/confit/tests/test_known_limitations.py`](../tests/test_known_limitations.py),
with further twins in the tests named per section. Coverage is demonstrated,
not total: a limitation with a twin breaks a test when it is lifted, and the
[ledger](oracle/07-the-divergence-ledger.md#executable-twin-coverage) records
the ones still without one.

**The contract.** For any SQL you hand it, the engine does exactly one of:

1. **Serve it bit-for-bit identical to DuckDB** (verified continuously
   against DuckDB's own test corpus: 543 of 678 statements as of 2026-10-05,
   recorded in [`reports/corpus-counts.json`](reports/corpus-counts.json)), or
2. **Refuse loudly at BUILD time** — `DuckDBInferFn(...)` raises a
   `ValueError` naming the construct. Nothing is ever silently wrong or
   silently dropped at inference time.

There is no third mode. Every limitation here is a *measured decision*
recorded in a pins spec (`packages/confit/docs/specs/`), not an accident.

**Which DuckDB** (a choice with a user-visible cost — see §5). "Identical to DuckDB" means DuckDB with its query optimizer off:

```sql
PRAGMA disable_optimizer;
```

That is not a smaller DuckDB. The binder is untouched, so types and
bind-time errors are the same; execution-level laziness — an untaken `CASE`
arm, `AND`/`OR` short-circuit in a filter — is the same. What it removes is
the 33 plan-rewrite passes, constant folding among them (`1 + 2` still
answers `3`, by execution), plus a dozen physical-plan choices that read the
same switch (window and DISTINCT ON operators, join flipping); see
[the oracle's scope](oracle/01-what-the-oracle-is.md).

The reason is that the optimizer-on reading is not a function of the query.
`statistics_propagation` decides from a column's stored null statistic, so
DuckDB answers the *same query over the same rows* differently depending on
the table's insert history — measured: a table built as `[-128, NULL]` and
then having the NULL deleted answers differently from one built as `[-128]`,
with identical contents. Confit compiles once against a *schema* and serves
many batches; it never sees a table, let alone its history. A target you
cannot compute from the query is not a target.

**What "identical" means for ROW ORDER.** Values are
bit-for-bit. A *sequence* is only promised where one is defined: on the row
path by the serving contract (output rows follow input rows -- `map` exactly,
`filter` as a subsequence, `many` as per-input-row blocks in input order,
join order within a block being the documented multiset). Anywhere else SQL
defines no order and neither do we -- DuckDB itself returns the same unordered
`GROUP BY` in twelve different row orders over twelve connections (measured).
The campaign fuzzer compares per this rule: sequence-strict self-legs on the
row path (a reversed batch must reverse), multiset otherwise.

**Boolean short-circuit** is decided per CONTEXT, as DuckDB decides it: selection context (the
WHERE root and every `CASE WHEN` condition, projections included) makes
`AND` lazy left-to-right and `OR` its exact dual, and exits to eager value
context at `NOT`/`IS NULL`/comparisons/function arguments and CASE arms. The
measured matrix runs against the live oracle in
`known_divergences/test_short_circuit.py`.

---

## 1. The specialization bargain (inherent to the engine model)

Confit's speed comes from doing ALL general work at build time:
parse once, bind once, compile once, freeze the static tables into the
code. Anything that would require re-doing general work per row is
rejected by design (class 3 in the
[restriction inventory](specs/serving-contract.md#restriction-inventory-by-class)):

| Limitation | You'll see | Why |
|---|---|---|
| Regex patterns must be constants (`regexp_matches(s, pattern_col)` rejects) | `unsupported: non-constant regex pattern (patterns compile at prepare)` | Regexes compile at prepare; DuckDB compiles per row. Per-row compilation is the opposite of specialization. |
| Replacement strings / regex options / extract group indexes must be constants | `non-constant regexp_replace replacement` etc. | Same. |
| Static (join) tables must be provided at build time; under the DEFAULT shapes their keys must be unique | `duplicate map key`; `has no equality key` for a static table of 2+ rows joined without one | Joins are frozen hash maps baked into the function. Duplicate keys mean 1:N join multiplicity (and so does a join whose `ON` only filters, over more than one static row), which SERVES under the opt-in `shape='many'` — along with cross joins (`a, b` or `a CROSS JOIN b`, the same join on DuckDB), inequality `ON` predicates, and constant `ON` clauses — with multiset parity vs DuckDB (its join output order is a measured hash-join accident; the engine emits probe order outer, build insertion order inner). Under `filter`/`map` the 1:1 contract holds. NULL *values* serve; NULL *keys* never match. |
| Self-joins require `shape='many'` | `joining the dynamic table to itself` | Under `'many'` the batch itself becomes the build side (assembled per call, the join condition as a per-pair residual) — comma/cross, `ON`, `USING` and `NATURAL` self-joins serve with multiset parity; a `USING`/`NATURAL` column is merged into the left occurrence exactly as DuckDB merges it. The default shapes raise the error. |
| Exactly one row table drives the query | `the specializer takes exactly one row table`, `must be the dynamic table` | The serving contract is rows-in → rows-out for one entity stream. |
| A SQL function expands at build time, up to 4,000,000 tokens | `sql function 'f' expands past 4000000 tokens`; `a sql function's value read outside the projection spells out past 4000000 nodes` | Confit replaces each call with its body before it binds the query. A token is one word, number or symbol of the SQL text. A value that the body reads more than once expands only once for each call. `SqlFunction.sql_lets` lists these values. The projection reads such a value where it stands. Two reads still take the whole value, as DuckDB does at each read. (1) If a value can trap, confit binds it again at each read. Then each read traps where DuckDB's read traps. (2) A read in WHERE or JOIN ON, or in the projection of a `shape='many'` query, takes the whole value. These two reads count toward a second cap of 4,000,000 for the whole query. This cap counts the tokens of a value that confit binds again, and the nodes of a value that a read takes whole. A node is one operation or one value in the bound query. A column without an alias gets its name from the query text, as DuckDB names it. So the name of `f(x)` is `f(x)`, and a name never contains the body. |
| A query compiles to one native function, and Cranelift caps it twice: 2^21 virtual registers, and 2^24 instructions, blocks or values (indexes it packs into 24 bits) | `unsupported: the compiled query is past the code generator's size limit (...)` | One function per query is what makes a row cheap. Three checks refuse, one message family: (1) once the query is lowered and verified, a floor on the registers Cranelift needs (`src/specializer/exec/size.rs`: what it provably assigns before lowering plus a temporary per helper call; `... at least N virtual registers ...`) refuses a query past 2^21 before anything else compiles it; (2) after the CLIF build, a function past 2^24 of anything refuses (`... Cranelift indexes each in 24 bits`) — past it a release Cranelift corrupts the function silently, and the floor, which bounds what survives Cranelift's optimizer, cannot see it; (3) Cranelift's own refusal (`Code for function is too large`) covers what neither proves: its count runs about 1.8-1.9x the floor on the native catalog's Normalizer, more on compare- and select-heavy queries, so a query just past the cap refuses only after the full compile. Fewer output columns or smaller expressions build. |

## 2. Out of scope for row-serving (by decision, not difficulty)

**The row-shape contract**: `DuckDBInferFn(..., shape=...)`
declares how many output rows each input row may produce, checked at
build time. `"filter"` (the default) is the engine's native 0..1;
`"map"` statically PROVES exactly-one (`out[i] ↔ in[i]`, the strict
serving guarantee) by rejecting anything that can drop a row — a WHERE
clause, an INNER join (key misses drop); `"many"` (0..N) is the multiplicity opt-in: duplicate-key
joins, cross joins, and inequality/constant `ON` joins build ONLY under
it (one join per query, a named rejection) — multiplicity can
never sneak into a serving path by default. Comma and `ON` self-joins
serve under `'many'` too, and so do `USING`/`NATURAL` self-joins.

The engine serves **row-at-a-time feature transforms**. Whole-relation
constructs are out of scope because their output shape is not
one-row-in/one-row-out. Syntax is not the scope test, though: the refusals
below are listed by the syntax that triggers them, and only the forms that
depend on sibling request rows are outside the model. A row-local CTE,
subquery or set operation is inside the model although this engine refuses it — the
[restriction inventory](specs/serving-contract.md#restriction-inventory-by-class)
classifies each family.

- Aggregation: `GROUP BY`, `HAVING`, `sum`/`count`/`avg`/... →
  `aggregate function ... (aggregation is not served)`
- `ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT` — row-independent transforms
  don't reorder or deduplicate.
- CTEs (`WITH`), `UNION`/`INTERSECT`/`EXCEPT`, subqueries, multiple
  statements.
- Table functions in FROM (`range(...)`) — there is no base table.
- `rowid` pseudo-column — rows have no stable identity in a stream.
- `FULL OUTER JOIN` — emits rows that no input row produced.

A **static-tables-only query** (nothing reads the request table) is outside
the model and refuses at build, naming the first construct the binder meets:
the driving relation (`table 's' as the driving relation (must be the dynamic
table …)`), `FROM-less SELECT`, or a clause such as `ORDER BY`. Run such a
query on DuckDB. The ruling is
[static-only queries](../../../loops/confit/decisions/closed/static-only-queries.md).

## 3. Type-system boundaries

The engine computes in exactly five types: `i64`, `i128` (DECIMAL's scaled
integer, HUGEINT and UBIGINT), `f64`, UTF-8 string, bool. Measured
consequences:

- **f32 base tables reject** (`engine is f64-only`).
- **A static column is served at its declared arrow type or refused by
  name** — never widened into a neighbouring lane, on the static and row
  paths alike. Widening diverges: `float32` in value AND type (`s.v * 3.0`
  over `0.1` is `0.30000001192092896`/FLOAT on DuckDB, `0.30000000447034836`/DOUBLE
  in f64). Every unsigned width has a lane of its own (`uint64` is UBIGINT
  on the i128 lane). The two exceptions that are served are measured
  equivalent, not convenient:
  `large_string`/`utf8` (DuckDB normalises them to VARCHAR) and the
  decimal tiers (below).
- **DECIMAL columns, row and static, serve EXACTLY**: the payload is
  the scaled integer in an i128 lane from ingest through the join, emitted
  as `decimal128(p,s)`. `2^53+1` comes back as itself, and so does
  `2^63+1` — an ordinary fit-time `sum(BIGINT)` produces that, which is
  why the lane is i128 and not i64. `decimal32`/`decimal64` inputs
  normalise to `decimal128(p,s)` output because DuckDB exports every tier
  as 128-bit arrow. Expressions over a decimal serve exactly, with
  DuckDB's types (docs/specs/decimal-expressions.md): `+ - * %`, the
  casts, comparisons, `CASE`/`COALESCE`/`greatest`/`IN` unification, and
  the DECIMAL overloads of `abs`/`ceil`/`floor`/`round`/`trunc`. What
  refuses, by name: a DECIMAL join key EXPRESSION against a VARCHAR build
  key or past the 38-digit common type (DECIMAL, integer and DOUBLE build
  keys serve, compared in DuckDB's common type), casts between a DECIMAL and a DOUBLE/VARCHAR/BOOLEAN
  in the INTO direction, and `IN`/`BETWEEN` families capped at 38 digits.
  `decimal256` columns refuse (DuckDB itself refuses them at arrow
  register, at any precision). A Python row value for a DECIMAL column is a
  `decimal.Decimal` or an `int`, exactly representable at the column's
  (p, s) — a float, or a value with more than `s` decimal places, refuses
  rather than rounding (`tests/test_decimal_rows.py`).
- **Struct fields SERVE.** Confit holds each scalar field of a struct
  column in a lane, in the request table and in a static table. Each
  spelling of a field read serves: `a.i`, deep paths such as `t.t.t.t`,
  `a['i']`, `(a).i`, `struct_extract(a, 'i')`, and chains that mix them. A
  struct star (`a.*`, with EXCLUDE or REPLACE) also serves. A field name
  can contain a dot (`a."x.y"`). The output names are the names that the
  oracle gives.
- **A whole struct is one output value** (`tests/test_struct_outputs.py`).
  These struct values serve:
  - a struct column or a nested struct field, in each spelling above;
  - a struct column of a static table. It is NULL where a LEFT join finds
    no row;
  - the row of a relation (`SELECT t FROM t`). It is never NULL. Where a
    LEFT join finds no row, it is a struct of NULLs;
  - a call of a UDF (`ExternFunction`) or a SQL function (`SqlFunction`)
    that returns a struct;
  - `struct_pack(k := v, ...)` and the struct literal `{'k': v, ...}`, over
    these values and over scalars;
  - a CASE whose results are struct values of one type, or NULL.

  `s IS NULL` tests the struct itself, not its fields. The Arrow entry
  point (`infer_arrow`) returns a nested Arrow `struct` column. The Python
  entry point (`infer_rows`) returns a nested dict. Each field keeps its
  type, for example DECIMAL, the unsigned widths, and HUGEINT as
  `decimal128(38, 0)`.
- **These struct forms refuse by name:**
  - a derived table or a CTE that carries a struct column;
  - a CASE whose results are structs of different types, or a struct and
    another value. The oracle converts struct types to a common type, and
    confit does not;
  - `struct_pack(...) IS NULL`, and `IS NULL` over a CASE of structs. The
    oracle builds each field first, so a field can trap;
  - a struct with a field of a type that confit does not compute in, for
    example a timestamp or a list;
  - a struct without fields;
  - `row(...)`, and `struct_pack` with an argument that has no name;
  - a whole struct where a scalar is necessary: an operand, a function
    argument, or a comparison (`s = s`);
  - the static side of a struct join key that USING or NATURAL merges
    (`d.w`, or `d.*`). The merged key `w` serves, with the values of the
    request table;
  - `a['i']` when `a` also names a relation in scope. There `a.i` reads the
    column of the relation, so the two spellings differ.
- **Lists reject** (`row column 'x' has a non-scalar type`) — list types
  are out of the row-schema vocabulary and are opaque: unreferenced
  (star expansion included) they cost nothing to declare, an
  unreferenced timestamp/list field does not block a scalar-only
  query, and `EXCLUDE`/name filters/`REPLACE` can remove one from a
  star. Referenced, they refuse by name. Lists also gate
  `regexp_extract_all` / `regexp_split_to_array` / the STRUCT form of
  `regexp_extract` (`list-valued, non-scalar`).
- **DECIMAL literals are DECIMAL**, as on DuckDB: `1.5` is
  `DECIMAL(2,1)`, and arithmetic over it is exact
  (docs/specs/decimal-expressions.md). A literal with an exponent
  (`1.5e0`) or more than 38 digits is DOUBLE. When pinning DOUBLE
  behaviour, spell the operand DOUBLE (`-2.5e0`, `::DOUBLE`): a bare
  `CAST(-2.5 AS BIGINT)` is a DECIMAL cast (half away from zero, `-3`),
  a DOUBLE one rounds half to even (`-2`).
- Integer widths: the engine TYPES in DuckDB's lattice (TINYINT..BIGINT —
  literals are INTEGER by magnitude, `::SMALLINT` is real, `ascii` returns
  INTEGER, `infer_arrow` emits int8/int16/int32 from the type) but
  COMPUTES in two machine widths, i64 and f64. The width is observable
  exactly where DuckDB's is: the Arrow schema (catalogue pinned in
  `test_integer_widths.py`) and the overflow trap threshold: a narrow value that
  overflows traps where it is PRODUCED, on both output paths and inside a
  wider expression (`a + a > 0`, `CAST(a + a AS BIGINT)`), exactly where
  DuckDB raises. The trap text names the width (`value out of range for
  INTEGER`) rather than DuckDB's operator wording, which the error-text
  rule permits. UTINYINT, USMALLINT and UINTEGER are narrow widths of the
  same lane (columns, statics and CAST targets, with DuckDB's own operator
  and unification lattices; `tests/test_unsigned.py`); unary minus over
  one (DuckDB wraps it), a shift, and a CAST from DOUBLE to one (DuckDB
  range-checks before rounding) refuse by name.
- **HUGEINT and UBIGINT compute on an i128 lane** (`tests/test_hugeint.py`),
  as UTINYINT..UINTEGER do on the i64 one: checked i128 arithmetic, UBIGINT
  range-checked to `[0, 2^64)`, DuckDB's operator and family lattices
  (`u64 + i64` is HUGEINT, `u64 + 1` UBIGINT), its literal typing
  (9223372036854775808 is HUGEINT), its string parser and its
  HUGEINT->DOUBLE conversion (two roundings). HUGEINT leaves as
  `decimal128(38, 0)`, as DuckDB exports it, past 38 digits too; UBIGINT as
  `uint64`. No arrow input type reads as HUGEINT. Refused by name: unary
  minus over UBIGINT (DuckDB wraps it), shifts over either width,
  `round`/`trunc` with digits over either, a string literal in an
  `IN`/`BETWEEN` beside either, and UHUGEINT (a literal past
  HUGEINT, a CAST target), which does not survive the Arrow trip. CAST
  targets: TINYINT through HUGEINT, every unsigned width but UHUGEINT,
  DOUBLE, DECIMAL, VARCHAR and BOOLEAN (and DuckDB's aliases for them) are
  served; every other target — UHUGEINT, FLOAT/REAL, INTERVAL, dates —
  refuses with `CAST target type <T>`.

## 4. Semantics descoped after measurement

Each of these was measured against DuckDB 1.5.5 first (pins in
`packages/confit/docs/specs/`), and rejected because serving it would risk a
wrong answer or require semantics we can't reproduce exactly:

| Construct | Why it's descoped |
|---|---|
| `^` operator | It IS pow in DuckDB, but sqlparser's precedence differs from DuckDB's (`2*x^y` would parse as `(2*x)^y`). Mapping it computes the wrong tree silently. Use `pow()`. |
| prefix `~`, `#`, `NOT GLOB` | Same class: precedence/parse divergences that would silently mis-associate. `xor()` covers bit-xor; `NOT (x GLOB p)` works. |
| Regex reject list: `\B`, `\Q…\E`, `(?<name>…)`, duplicate group names, bounds > 1000, stacked quantifiers (`a*+`), `\u` escapes, negated Perl classes inside `[...]` | The RE2↔rust-regex differential battery (98 entries) proved these are the constructs where the engines disagree or DuckDB itself is broken (`\B` crashes DuckDB at runtime on non-ASCII). Everything else is byte-identical. |
| Fuzzer-found regex rejects: `\1`–`\9` backrefs outside classes, the full stacked-quantifier grammar (`{2}*`, `?*`, `a???` — one lazy `?` is the only legal follower), nested repetition products > 1000, whitespace inside `{m, n}` bounds, class set-op lookalikes (`--`/`&&`/`~~`), non-POSIX `[` inside classes, Perl-class range endpoints (`[a-\d]`), capturing `(x){0}`, anchor-only multi-anchor patterns (DuckDB is SELF-inconsistent on these — its row path disagrees with its own constant fold), `$` anchors in non-final position (`'$hello'` — DuckDB's row path literal-optimizes the leading `$`+literal into a PREFIX match, matching "hello world", while its own constant fold matches normally; found by the standing fuzzer on seed 20260728), and counted repetitions over RE2's PROGRAM-SIZE budget (`(\p{L}){1,500}` is "pattern too large" in DuckDB while rust-regex serves it — rejected via a one-sided weight estimate that always fires before DuckDB's real budget; same seed, pins `pins-waveB/fuzzer-20260728.json`) | The standing differential fuzzer (`packages/confit/tests/test_duckdb_regexp_fuzz.py`, in the normal gate) found these 12 classes — each one a silent-wrong-answer risk in rust-regex. With them rejected, a 40k-case sweep across 8 seeds shows zero divergences. Pins: `pins-waveB/fuzzer-task54.json`. |
| `SIMILAR TO ... ESCAPE` | Not implemented in DuckDB itself. |
| `* EXCLUDE (t.key)` on a USING join, either qualifier | DuckDB UNMERGES the coalesced column (it reappears at the right table's position, with the right table's values) — measured, not modeled. Unqualified EXCLUDE works. |
| `BETWEEN`/`IN` mixing non-numeric string literals with numbers | DuckDB converts at EXECUTION time (an empty input succeeds!); a bind-time conversion was measured to be over-eager. Numeric literals convert fine. |
| `COLUMNS(...)` inside expressions, lambda/list forms | Only bare `COLUMNS('re')` / `COLUMNS(*)` as select items are served. |
| Pad/repeat counts past the 1 GiB string-builder budget | A LITERAL count that can exceed the budget refuses at build; a DATA-DRIVEN count (column, `CAST(k AS INTEGER)`) keeps the runtime cap and traps at 1 GiB. DuckDB is deterministic here (measured): `repeat` serves up to a 4294967295-byte string and raises Out of Range above it; `lpad`/`rpad` take an INTEGER count, so a count above 2147483647 is a Binder Error. Between 1 GiB and those bounds DuckDB serves and we refuse or trap — the ground is Confit's resource limit, not DuckDB instability: no gigabyte allocations per serving row, by decision. |
| `CASE` where every branch is NULL, over a condition that can trap | DuckDB types an all-NULL `CASE` as a bare NULL but still evaluates its conditions, so a trapping condition errors per row there; the engine would serve the constant NULL, so it refuses. All-NULL `CASE` over non-trapping conditions, and `COALESCE`/`IFNULL`/`least`/`greatest` of only NULLs, serve as a bare NULL (adoptable, INTEGER at the top level). |
| `error()` outside a CASE result, or with a message computed per row | DuckDB types `error()` as SQLNULL and never folds it. As a CASE result that type is adopted and the arm raises when taken, which is served (`tests/test_error_function.py`); elsewhere a SQLNULL operand is dropped by DuckDB's binder or carried as a type confit does not model, and a per-row message needs a runtime-built trap text. |
| `nullif(NULL, x)` where `x` can trap | DuckDB evaluates `x` (and traps on it) though the answer is NULL; the bare NULL has no node that evaluates an operand for its trap only, so it refuses. Over a trap-free `x` it serves (`tests/test_opt_emulated_triage.py`). |
| A JOIN ON comparison between the two sides with trapping operands (`t.k = s.k AND t.v < s.w * 2`) | DuckDB makes it a join condition and evaluates each operand over its whole table, matched or not, so the trap fires with no key match (measured, INNER and LEFT); the engine evaluates a residual per matched pair, so it refuses. An operand reading both sides, or the comparison under OR, is per pair on DuckDB too and serves. |
| Bare `NULL` as `repeat`'s string (BLOB face) | DuckDB picks the **BLOB** overload — a type this engine does not serve — so adopting VARCHAR would answer with a different schema. Spell it `CAST(NULL AS VARCHAR)`, which both engines type identically. Adopters that agree with DuckDB — `upper(NULL)`, `coalesce(NULL, x)`, `nullif(x, NULL)`, `nullif(NULL, x)` (int32), a NULL `repeat` count — serve, pinned schema-equal. |

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
- **One known oracle-divergent source, two statements** (excluded from the corpus by name in
  `packages/confit/fuzz/exclusions.py::SOURCE_EXCLUSIONS`): DuckDB
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
  statistic, and pruning a filter from a value range). The campaign fuzzer
  labels these findings `DIVERGE_OPT` (§7).

  The trade is deliberate: matching the optimizer means matching an
  undocumented moving target that is not a function of the query, and in
  the other direction it would mean *serving* where your DuckDB raises.
  This way the divergence is always a loud trap or refusal, never a
  different served value. If it bites you, `PRAGMA disable_optimizer` in
  your DuckDB session reproduces exactly what confit does.
- **`%`-by-zero NaN bit pattern is platform-libm** — pinned as
  engine==oracle bit agreement per platform, not a constant.
- **Schema qualifiers are registry-noise**: the engine's table
  registry is schema-less, so a relation qualifier (`JOIN s1.t1`) resolves
  when the table part matches a registered bare name. DuckDB's
  schema-existence errors (`schema "x" does not exist`) are not
  reproduced — a schema-less registry cannot know which schemas would
  exist. Ambiguous matches error. A relation is in scope under its bare
  name, and a column may be qualified through the schema it was named
  through (`main` when unqualified) and the `memory` catalog: `d.v`,
  `main.d.v` and `memory.main.d.v` all serve over `JOIN main.d`; another
  schema, another catalog, or a schema over an alias refuses, as DuckDB's
  binder does. Twins: the schema-qualifier tests in
  `test_known_limitations.py`.

## 6. How to read a rejection

Every rejection is a `ValueError` at `DuckDBInferFn(...)` construction
whose message starts with a classification:

- `unsupported: ...` — real SQL, deliberately not served (this document).
- `parse error: ...` — the dialect surface ends here.
- `bind error: ...` — the query is wrong against YOUR schema (typo,
  type mismatch) or your declarations disagree with each other or with the
  data (a UDF's `takes`/`returns`, a NULL in a column declared non-null),
  not a limitation.

If a message you hit isn't in this document or the tests, that's a bug in
our bookkeeping — file it.

## 7. How this document stays honest

Three mechanisms in the normal test gate, and one manual campaign:

1. **The corpus replay** (678 statements mined from DuckDB's test suite):
   every statement must match bit-for-bit, reject cleanly, or be a named
   divergence — a wrong answer anywhere fails the gate.
2. **The executable twin** (`packages/confit/tests/test_known_limitations.py`): the
   limitations it asserts break a test when lifted. Coverage is partial, not
   total; the ledger's `claim: doc-twin-totality` records the gaps.
3. **The standing differential fuzzer** (`packages/confit/tests/test_duckdb_regexp_fuzz.py`):
   randomized DuckDB-vs-engine sweeps of the regex surface on every run
   (seed/size overridable for deep runs) — new divergences fail with the
   reproducing seed and SQL, and their fix lands as a reject-list entry
   plus a row in this document.
4. **The campaign fuzzer reads DuckDB TWICE** (`packages/confit/fuzz/`),
   once with the optimizer off and once on, so a finding says which kind it
   is instead of needing a human to reason about it. It is NOT in the test
   gate: campaigns are a manual `python -m fuzz.runner` run, and the gate's
   `tests/test_fuzz_smoke.py` checks only its machinery (deterministic
   generation, reproducible verdicts, the verdict rules) — a green gate says
   nothing about zero campaign findings. Of its readings: a disagreement with the
   optimizer-off reading is a bug, a disagreement only with the optimizer-on
   one is the §5 cost above, and an agreement with optimizer-on *against*
   the oracle means the engine is reproducing a pass it should not — that
   last category is reported as a bug.
