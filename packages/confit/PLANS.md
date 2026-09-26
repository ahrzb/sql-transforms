# Confit plans

Open work only: gaps, defects, proposals and open questions. Facts about what
Confit does today live in `docs/`; this file lists only what it does not do yet.
Remove an item when it lands.

## Correctness defects (served wrong, not refused)

- **Dead `CASE` arms over-fold at bind time.** `(CASE WHEN false THEN x END)`
  with a column `x` folds to a bare NULL, which skips the `||` binder's
  `bind_foldable` gate. The result is typed INTEGER instead of VARCHAR, and
  `abs`/unary minus then serve where DuckDB's binder refuses. Pinned
  xfail-strict in `tests/test_open_divergences.py`.

## SQL surface gaps (refused, DuckDB serves)

- **Row-local CTEs, scalar/correlated/`IN` subqueries, derived tables
  (`FROM (SELECT ...)`) and set operations** are blanket syntax bans. The scope
  rule only excludes the batch-dependent forms of these, so each row-local
  form is a gap to close. `WITH` is the largest refusal class the campaign
  generator reaches.
- **Per-row aggregation over matched static rows.** This is inside the model
  but refused under both spellings: a correlated scalar subquery, and
  `JOIN` + `GROUP BY` under `shape='many'`. It needs an accumulator over the
  `many` fan-out walk, plus a ruling on whether `JOIN`+`GROUP BY` is per-row
  only when it yields one group per input row. `DOUBLE` `sum`/`avg` then needs
  the float-reduction bound. Its algorithm and edge domain (n=0/1,
  non-finite values, overflow, all-NULL input) must be settled before
  implementation (`docs/oracle/05-the-comparison-contract.md`).
- **`IS [NOT] DISTINCT FROM`** is refused through the expression catch-all,
  although the dialect plan already models it (`src/dialect/plan.rs`).
- **`BOOLEAN` compared with `VARCHAR`** (`b = 'true'`) is a bind error
  (`cannot compare BOOLEAN with VARCHAR`), but DuckDB casts and serves it.
- **Mixed-type join keys.** A `NATURAL`/`USING`/`ON` join of `BIGINT` against
  `VARCHAR` refuses (`cannot join i64 with str`), but DuckDB casts and serves it.
- **Struct join keys under `shape='many'`.** The fan-out loop implements plain
  equality only (no NOT-DISTINCT form, `src/specializer/lower.rs`). Struct
  keys whose field-name sets differ also refuse, where DuckDB serves a
  constant-empty join.
- **`BETWEEN`/`IN` mixing non-numeric string literals with numbers** refuses,
  because DuckDB converts these at execution time and a bind-time conversion
  is over-eager.
- **All-NULL `CASE`/`COALESCE`/`least`/`greatest`** refuses. DuckDB binds these
  as INTEGER.
- **`COLUMNS(...)` inside expressions, and lambda/list forms.** Only bare
  select-item `COLUMNS` serves.
- **`SELECT w.*` over a static struct and `SELECT s.*` over a struct-carrying
  table** refuse. A whole struct is not a servable value (see type lattice).
- **Struct fields whose own name contains a dot** are silently skipped on the
  static side and treated as opaque on the row side (`src/duckdb/mod.rs`).
  Lanes carry structured paths, so the skip can be lifted.
- **A column qualified through a schema-qualified relation** (`d.v` over
  `JOIN main.d`, 3-part `s1.d.v`) refuses where DuckDB serves.
- **Parse-divergence guards.** `^`, prefix `~`, `#` and `NOT GLOB` refuse
  because sqlparser's precedence differs from DuckDB's. Closing this needs the
  dialect frontend's own parser to replace sqlparser. The regex reject list
  needs an RE2-compatible engine to close.
- **Zero-row shapes with trapping constants** (`WHERE FALSE`, empty input)
  refuse at build, where DuckDB serves `[]`. This is kept as deliberate
  strictness. An optional cleanup is to make `fold` fallible and delete the
  duplicate `eval_i32_literal` walker, so that one evaluator knows constant
  semantics.

## Type lattice

- **Decimal expressions.** Arithmetic, `CAST` to anything but `DOUBLE`, and
  `COALESCE`/`CASE` unification over DECIMAL refuse. Bare decimal literals
  serve as f64, where DuckDB uses `DECIMAL(p,s)`, which makes
  `CAST(-2.5 AS BIGINT)` round differently. Decimal row columns are opaque.
  Closing this needs DuckDB's per-operator result-scale rules measured, and
  `Dec(p,s)` threaded through the expression tree. It empties the `UNSHIPPED`
  bucket.
- **i128 lane and the unsigned family.** `HUGEINT`, `UHUGEINT` and unsigned row
  and static columns refuse. This needs i128 arithmetic plus overflow traps on
  both backends, and exact `sum`/`product` accumulation at decimal128(38,0).
- **`float32`.** Row columns and statics refuse (`engine is f64-only`). A
  served FLOAT needs f32 arithmetic, not widening.
- **Temporal types.** `DATE`/`TIMESTAMP`/`TIME` row columns, casts, literals and
  functions (`year`, date arithmetic) all refuse. Opaque scalar join keys
  (TIMESTAMP, DATE, FLOAT32, UINT64, LIST, row-side DECIMAL) refuse by name.
  They could serve through a key-only lane kind with per-type equality proofs.
- **Non-scalar values.** Whole-struct output, struct literals (`{'a': v}`),
  bracket field access (`a['i']`), lists (`list_value`, `regexp_extract_all`,
  `regexp_split_to_array`, the STRUCT form of `regexp_extract`), and struct
  fields of non-scalar type all refuse. These need nested output support at
  the Arrow boundary.
- **BLOB.** There is no lane, so a bare `NULL` as `repeat`'s string (the BLOB
  overload) refuses. `decimal256` statics are upstream-blocked, because DuckDB
  refuses them at Arrow registration.
- **`RowTy` newtype.** This would delete the `Ty::Dec(..) => unreachable!` arms
  and belongs with the change that makes DECIMAL a row lane.

## Joins and multiplicity

- **More than one join under `shape='many'`.** One join per query is enforced
  (`src/specializer/lower.rs`). Composing multiplicity across joins is the
  hardest open join item.
- **Remove the static-only fold.** Queries that read no request table are
  outside the model, but `eval_static_only`/`Engine::Constant`
  (`src/duckdb/mod.rs`) still serve them with `backend == "constant"`. Removal
  touches `backend`/`boundary`'s public `"constant"` value (documented in
  `sql_transform/_projection.py`), the fuzzer's `static_agg` arm and
  `constant-*` compare modes, the static-only row-limit refusal and its tests,
  and the corpus floor. The floor drops by the table-function matches this
  path serves.

## Refusal quality

- **Echoing refusals.** The expression catch-all
  (`unsupported: expression: {other}`) echoes SQL text instead of naming the
  construct. Examples: `IS DISTINCT FROM`, scalar and correlated subqueries,
  `IN (SELECT ...)`, struct literals, and `DATE '...'`. Derived tables echo the
  subquery (`unsupported: FROM (SELECT ...) AS x`). C5 requires construct
  naming, and `fuzz/runner.py::refusal_quality` measures it. Each echo needs a
  named refusal site.
- **No mapping from refusal sites to inventory rows.** About 164
  `PrepareError` sites exist, and nothing ties each one to a class in the
  restriction inventory (`docs/specs/serving-contract.md`). A new refusal can
  therefore appear in no document.
- **Leafless-struct presence lanes.** The `Present` arm in
  `src/duckdb/arrow.rs::ingest` does not check that the batch column is a
  struct, so a leafless struct lane reads any column's validity.

## Oracle, evidence and gates

- **The generator never emits most CAST targets.** `fuzz/gen.py` renders
  only BIGINT, INTEGER, DOUBLE, VARCHAR and BOOLEAN, so the campaign never
  checks the refusal of the others (pinned in `test_known_limitations.py`)
  nor TINYINT/SMALLINT. Adding targets changes every seed's query: do it
  together with a fresh dated reading.
- **CI runs neither `cargo test` nor a debug-build pytest pass.** The
  random-IR interpreter-vs-cranelift differential and the Rust unit suites
  (`src/**/tests.rs`) never run in CI. Lowering invariants are
  `debug_assert!`s that release builds compile out.
- **816 pin queries cannot be replayed mechanically.** 460 have an untyped
  `input_repr` (a bare value that names no column or type), and 356 describe
  their tables only in prose (`docs/specs/pins-drift.json`,
  `scripts/pin_corpus.py`). Giving each a typed setup would bring them under
  the drift report.
- **IR generator coverage.** `ir::gen::gen_program` never emits 10 of the IR
  instructions (`Dtof`, `Itod`, `StoiOpt`, `StofOpt`, `ReMatch`, `ReExtract`,
  `ReReplace`, `ExternCall`, `ProbeRange`, `ProbeRead`), so the backend
  differential cannot see them. A totality test belongs beside the
  differential in `src/specializer/exec/tests.rs`.
- **C1 depth.** The training round-trip seeded differential defaults to 25
  cases (`MARGINALIZE_FUZZ_N`), against a specified 1,500–2,000. Either run
  the depth somewhere standing, or amend the control by reviewed decision.
- **Campaign cadence.** The generated-grammar campaign is a manual CLI with no
  standing schedule. The retired phase-2 width-residual count needs a replay
  of stored SQL or a fresh labelled run, followed by classification.
- **Unruled ledger entries.** Most rows of
  `docs/oracle/07-the-divergence-ledger.md` carry a proposed disposition
  without a ruling.
- **Order-sensitive families.** Each new one (beyond the join multiset rule)
  needs a justified ordering contract before it serves
  (`docs/oracle/03-nondeterminism.md`).
- **Executable twins.** `HAVING`, `OFFSET`/`FETCH`/`TOP`,
  `INTERSECT`/`EXCEPT`, subqueries, multiple statements, table functions and
  `QUALIFY` refuse without a twin in `tests/test_known_limitations.py`. No
  executable check covers the 2 GiB Arrow batch ceiling.
- **Optional, not adopted.** Corpus-wide pin governance metadata, generic
  re-record tooling, and evidence mutability classes.

## Performance

- **Native transform families.** About 93% of a fitted transformer's per-row
  cost is sklearn's own `transform()` (≈60 µs, against ≈1.5 µs for the same
  arithmetic in numpy). Native typed entries (`PCA`, `StandardScaler`, and so
  on, behind the existing extern slots) are the ~100x lever. They are blocked
  on adopting per-family parity bounds by review: bit-exact for scaler/tree
  tiers, a declared ulp bound for matvec tiers, gated by swap-the-entry.
- **Vectorized `apply_batch` for `infer_arrow`.** UDFs are called once per row
  through the scalar protocol even on the Arrow path.
- **Serving bench against the Python twin.** The engine reads 1.20–1.80x
  slower than the `python_dict` twin at n=64 on a fresh release wheel
  (2026-09-26 reading), so a stale wheel is not the cause. The cause is
  unsettled between a baseline that changed identity and a real regression;
  bisect against `a6fa318`. The n=1 twin cell swings up to 2x between runs,
  so a recorded ratio should be the n=64 one. Settle this before adopting any
  bench refresh cadence.
- **`benchmarks/bench_transforms.py` does not run.** It declares UDF types as
  a tuple of strings; the declaration takes an Arrow schema. Port the
  declarations, then take the transformer-path reading.
- **Tree scoring.** Not built: `HistGradientBoosting*` (binned thresholds),
  MLP (`mat_stack`), a QuickScorer or vectorized multi-tree walk that keeps
  the accumulation sequential in `tree_span` order, and kNN/kernel SVM
  (training set as state).

## Authoring boundary (`sql_transform`)

- **Admission-ladder headroom**, roughly by value:
  - step semantics for order-keyed windows off the training support
  - static-table joins with frozen composition
  - IN-subqueries as fitted sets
  - star bundles into transformers
  - typed takes (string features)
- **Composition directions not yet law** (`docs/properties.md`):
  FROM-position templates, frozen-artifact inlining, `as_udf()`, and the
  leakage/cross-fitting question.
- **Arrow-hostile partition keys.** BIT, TIMETZ and UNION keys cannot round-trip
  through the params table (xfail in `sql_transform/model/_marginal_test.py`).
- **Dialect reverse frontends.** `parse(sql, dialect=SPARK|BIGQUERY)` is
  demand-driven and unbuilt. The BigQuery L3 gate runs only when
  `CONFIT_BIGQUERY_PROJECT` is set, and its execution leg is not written:
  wire the Spark leg's seam through the BigQuery client (ship tables with
  `load_table_from_dataframe`, run the printed SQL, compare `rows_of()`).

## Code health

- `src/specializer/frontend.rs` is about 8,500 lines and wants splitting.
- `DuckDBInferFn::new` represents the shape three ways (`many`, `shape_kind`,
  `strict_map`). Unifying them into one `Shape` enum gets easier once the
  static-only fold is gone.

## Owner choice: which query classes next

Candidates for the next query class, each with its cost or blocker. None is
chosen yet.

| candidate | unlocks | cost / blocker |
|---|---|---|
| decimal arithmetic | exact DECIMAL expressions and literals; empties `UNSHIPPED` | per-operator scale rules through the whole expression tree |
| HUGEINT / unsigned (i128 lane) | the remaining integer widths, and exact wide aggregates | i128 arithmetic and traps on both backends |
| struct-valued outputs | whole structs, struct literals, bracket access | nested output schema at the Arrow boundary |
| multiple joins under `shape='many'` | multi-join serving | multiplicity composition across joins |
| native transform families | ~100x on transformer queries | per-family parity bounds adopted by review first |
| row-local CTEs / subqueries | the largest refusal class the generator reaches | binder support for nested scopes, and a named row-locality test |
| per-row aggregation over statics | aggregates over matched static rows | a `many`-walk accumulator, and resolving the float-reduction bound's domain |
