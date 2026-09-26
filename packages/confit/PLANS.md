# Confit plans

My working list: open work only, highest value first within each section.
Facts about what confit does live in `docs/`; questions waiting on a ruling are
records in `docs/decisions/open/`. Remove an item when it lands.

## Next

1. **Small surface wins DuckDB serves and we refuse.**
   - Struct fields whose own name contains a dot: skipped on the static side,
     opaque on the row side (`src/duckdb/mod.rs`).
2. **Generator reach.** `fuzz/gen.py` renders only BIGINT, INTEGER, DOUBLE,
   VARCHAR and BOOLEAN CAST targets, and never the forms in item 1. Widen it
   together with a fresh dated reading (a generator change re-deals every
   seed).

## Waiting on the owner

- Removing the static-only fold (`eval_static_only`/`Engine::Constant`,
  `backend == "constant"`): ruled out of the model
  (`docs/decisions/closed/static-only-queries.md`), removal changes
  user-visible behavior, drops the corpus floor by the table-function matches
  it serves, and touches `sql_transform/_projection.py`'s documented `backend`
  values and the fuzzer's `static_agg` arm and `constant-*` modes.
- `docs/decisions/open/`: next query classes, C1 depth, native-transform
  parity bounds.

## Query classes (large; order is the owner's call)

- **Row-local CTEs, scalar/correlated/`IN` subqueries, derived tables, set
  operations.** Blanket bans today; only batch-dependent forms are out of
  scope. `WITH` is the largest refusal class the generator reaches (86 of 526
  refusals of queries DuckDB answers, 2026-09-26).
- **Per-row aggregation over matched static rows** (correlated scalar
  subquery; `JOIN` + `GROUP BY` under `shape='many'`). Needs an accumulator
  over the `many` walk, a ruling on when `JOIN`+`GROUP BY` is per-row, and the
  float-reduction bound's algorithm and edge domain
  (`docs/oracle/05-the-comparison-contract.md`).
- **Decimal expressions.** Arithmetic, casts to anything but DOUBLE and
  unification over DECIMAL refuse; bare decimal literals serve as f64 (the
  `UNSHIPPED` bucket). Needs DuckDB's per-operator scale rules and `Dec(p,s)`
  through the expression tree. A `RowTy` newtype belongs with making DECIMAL
  a row lane.
- **i128 lane.** HUGEINT/UHUGEINT and the unsigned family (columns, statics,
  CAST targets) refuse. Needs i128 arithmetic and traps on both backends and
  exact `sum`/`product` at decimal128(38,0). The literal 9223372036854775808
  alone is 34 of the 526.
- **float32** (needs f32 arithmetic, not widening), **temporal types**
  (columns, casts, literals, functions; opaque temporal join keys could use a
  key-only lane), **BLOB**.
- **Non-scalar values.** Whole structs, struct literals, bracket access, lists
  and list-valued regex forms; `SELECT s.*` over struct-carrying statics.
  Needs nested output at the Arrow boundary. `decimal256` statics are blocked
  upstream (DuckDB refuses them at Arrow registration).
- **More than one join under `shape='many'`** (`src/specializer/lower.rs`);
  struct keys under `many` (plain equality only in the fan-out loop); struct
  keys whose field-name sets differ refuse where DuckDB serves a constant-empty
  join.
- **Parse-divergence guards.** `^`, prefix `~`, `#`, `NOT GLOB` wait on the
  dialect frontend's parser replacing sqlparser; the regex reject list on an
  RE2-compatible engine.

## Refusals

- **Echo fallback.** `expr_refusal` names the expression forms seen so far; an
  unlisted form still prints itself. Add names as `refusal_quality`'s echo
  list shows them.
- **No site-to-inventory mapping.** About 164 `PrepareError` sites, none tied
  to a class of the restriction inventory (`docs/specs/serving-contract.md`).
- **Executable twins missing** for `HAVING`, `OFFSET`/`FETCH`/`TOP`,
  `INTERSECT`/`EXCEPT`, subqueries, multiple statements, table functions,
  `QUALIFY`, and the 2 GiB Arrow batch ceiling (`tests/test_known_limitations.py`).
- Zero-row shapes with trapping constants (`WHERE FALSE`, empty input) refuse
  at build where DuckDB serves `[]` (deliberate). Optional: make `fold`
  fallible and delete the duplicate `eval_i32_literal` walker.
- `BETWEEN`/`IN` mixing non-numeric string literals with numbers refuses
  (DuckDB converts at execution; a bind-time conversion is over-eager).
- `COLUMNS(...)` inside expressions and lambda/list forms refuse.

## Evidence and gates

- **No debug-build pytest pass.** Lowering invariants are `debug_assert!`s the
  release extension compiles out.
- **IR generator coverage.** `ir::gen::gen_program` never emits `Dtof`, `Itod`,
  `StoiOpt`, `StofOpt`, `ReMatch`, `ReExtract`, `ReReplace`, `ExternCall`,
  `ProbeRange`, `ProbeRead`; add them and a totality test beside the
  differential in `src/specializer/exec/tests.rs`.
- **816 pin queries are not mechanically replayable** (460 untyped
  `input_repr`, 356 prose-only tables; `docs/specs/pins-drift.json`).
- **Campaign cadence.** Manual CLI, no schedule.
- **Unruled ledger rows** in `docs/oracle/07-the-divergence-ledger.md`; each
  new order-sensitive family needs an ordering contract first
  (`docs/oracle/03-nondeterminism.md`).

## Performance

- **Native transform families.** A fitted transformer costs ~118 µs per row
  against 1.4 µs without it (`bench_transforms.py`, 2026-09-26); nearly all is
  sklearn's `transform()`. Waits on the parity-bound ruling.
- **Serving vs the Python twin.** 1.20–1.80x slower at n=64 on a fresh
  release wheel. Bisect against `a6fa318` (regression vs the twin's change of
  identity). Record the n=64 ratio; the n=1 twin cell swings 2x.
- **Vectorized `apply_batch`** for `infer_arrow` (UDFs are called per row).
- **Tree scoring** not built: `HistGradientBoosting*`, MLP, a vectorized
  multi-tree walk keeping `tree_span` accumulation order, kNN/kernel SVM.

## Code health

- `src/specializer/frontend.rs` is ~8,500 lines. Split by concern (scope and
  binding, expressions, star expansion, joins, casts/literals) before the next
  large query class lands in it.
- `DuckDBInferFn::new` carries the shape three ways (`many`, `shape_kind`,
  `strict_map`); one `Shape` enum, easier once the static-only fold is gone.

## Authoring boundary (`sql_transform`)

- Admission-ladder headroom: step semantics for order-keyed windows off the
  training support; static-table joins with frozen composition; IN-subqueries
  as fitted sets; star bundles into transformers; typed takes.
- Composition directions not yet law (`docs/properties.md`): FROM-position
  templates, frozen-artifact inlining, `as_udf()`, leakage/cross-fitting.
- BIT, TIMETZ and UNION partition keys cannot round-trip through the params
  table (xfail in `sql_transform/model/_marginal_test.py`).
- Reverse dialect frontends (`parse(sql, dialect=SPARK|BIGQUERY)`) are unbuilt;
  the BigQuery L3 execution leg is not written (wire the Spark leg's seam
  through the BigQuery client: `load_table_from_dataframe`, run the printed
  SQL, compare `rows_of()`).
