# Confit plans

My working list: open work only, highest value first within each section.
Facts about what confit does live in `docs/`; questions waiting on a ruling are
records in `docs/decisions/open/`. Remove an item when it lands.

## Next

1. **Nightly campaign follow-through.** `.github/workflows/nightly-campaign.yml`
   runs `fuzz.nightly` (400k fresh seeds in four parallel shards, plus the
   metamorphic suite) and files red runs as a "Nightly campaign findings"
   issue. Triage each filed class to a fix, a named exclusion, or an
   open-divergence pin. The first six sharded nights are triaged
   (`docs/reports/2026-10-05-sharded-nightly-triage.md`); still open from them:
   - an INNER/CROSS join on an empty static table: whether DuckDB evaluates
     the row side depends on its pipeline shape
     (`docs/decisions/open/empty-static-join-trap-timing.md`, owner ruling);
   - `nullif(NULL, x)` drops `x`, which DuckDB evaluates and can trap on
     (seeds 3058298, 3722953). The adoptable-NULL channel has no node that
     evaluates an operand for its trap only;
   - narrow-width shifts: `1 << 31` traps at INTEGER on DuckDB, while confit's
     shift kernel is 64-bit;
   - a narrow-width overflow traps on both engines, but confit's text names the
     arrow range instead of DuckDB's `Overflow in multiplication of INT8`.

   Still waiting from #305: seed 1159605 (OPT_EMULATED, owner ruling) and the
   TIMEOUT class where confit traps first while DuckDB builds a 2 GiB string
   (seed 1102717, EXCLUDED ratification).
2. **Parity migration.** On the campaign verdict: `duck_check` (about 550
   calls across the `test_duckdb_*` files, the shrinker's pin template),
   `test_null_operands`, `test_arm_widening`, `test_derived_tables`,
   `test_metamorphic`. Still on their own helpers: `test_integer_widths`,
   `test_join_keys`, `test_infer_arrow`, `test_struct_column_access`,
   `test_params_joins`, `test_udfs`, `duck_check_ulp` (ulp tolerance) and the
   UDF helpers (`fuzz.parity` has no UDF input yet).
3. **Subquery design, PR 4** (static-only subqueries computed at
   construction) waits on the owner: its 48 measured candidates turned out to
   be unread CTEs, which now serve, so the class has no generated case yet
   (`docs/specs/2026-09-26-row-local-subqueries-design.md`, "Measured
   recovery").
4. **Unaliased expression names** (`tests/test_open_divergences.py`):
   DuckDB names `a + 1` as `(a + 1)`, printing the bound expression; confit
   echoes the SQL text. Needs DuckDB's expression printer for the output
   name, at the top level and at every subquery boundary.

## Waiting on the owner

- `docs/decisions/open/`: the order of the query classes after derived tables
  and CTEs, native-transform parity bounds, trap timing of a join on an empty
  static table.

## Query classes (large; order is the owner's call)

- **Scalar/correlated/`IN` subqueries, a CTE referenced twice, set
  operations.** Blanket bans today; only batch-dependent forms are out of
  scope. Row-local derived tables and CTEs are in Next.
- **Per-row aggregation over matched static rows** (correlated scalar
  subquery; `JOIN` + `GROUP BY` under `shape='many'`). Needs an accumulator
  over the `many` walk, a ruling on when `JOIN`+`GROUP BY` is per-row, and the
  float-reduction bound's algorithm and edge domain
  (`docs/oracle/05-the-comparison-contract.md`).
- **Decimal remainders.** Decimal expressions serve
  (docs/specs/decimal-expressions.md). Still refused by name: casts from
  DOUBLE/VARCHAR/BOOLEAN into a DECIMAL (DuckDB's double->decimal rounding
  and its string parser), a DECIMAL join key expression against a
  non-DOUBLE build key, `IN`/`BETWEEN` families capped at 38 digits, and
  DECIMAL ROW columns (a `RowTy` newtype belongs with making DECIMAL a row
  lane).
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
- A constant `CAST` that fails refuses at build even where a NULL-folded
  sibling makes the whole expression NULL on DuckDB
  (`CAST('' AS DOUBLE) / sqrt(NULL)` serves NULL there).
- `BETWEEN`/`IN` mixing non-numeric string literals with numbers refuses
  (DuckDB converts at execution; a bind-time conversion is over-eager).
- `COLUMNS(...)` inside expressions and lambda/list forms refuse.
- **Relation read as a struct** (`d['k']`, `(d).k`, `__THIS__['a']`): DuckDB
  serves the relation's row struct; confit refuses in every spelling, but as
  a bind error ("column 'd' does not exist"), which claims DuckDB rejects it.
  Either serve it (it is `d.k`) or refuse it as unsupported by name.
- **Wrapped-query refusals** the metamorphic suite allowlists
  (`fuzz/metamorphic.py` `KNOWN`): a struct- or list-valued column inside a
  derived table (about 13% of generated queries when wrapped) and a subquery
  under a `shape='many'` join. Serving struct slots removes the first.

## Evidence and gates

- **No debug-build pytest pass.** Lowering invariants are `debug_assert!`s the
  release extension compiles out.
- **IR generator coverage.** `ir::gen::gen_program` never emits `Dtof`, `Itod`,
  the decimal-expression opcodes (`Dop`, `Dcast`, `DcastOk`, `Dunary`, `Dtos`),
  `StoiOpt`, `StofOpt`, `ReMatch`, `ReExtract`, `ReReplace`, `ExternCall`,
  `ProbeRange`, `ProbeRead`; add them and a totality test beside the
  differential in `src/specializer/exec/tests.rs`.
- **816 pin queries are not mechanically replayable** (460 untyped
  `input_repr`, 356 prose-only tables; `docs/specs/pins-drift.json`).
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
- **Decimal kernels.** Checked decimal casts, capped-width arithmetic and
  the rounding builtins are one helper call per row on the JIT; a DECIMAL
  query costs ~110 ns/row where its DOUBLE spelling costs ~55
  (`docs/reports/2026-09-28-decimal-expressions.md`). Inlining the checks
  is the next step if decimals show up in serving profiles.
- **`scripts/bench_specializer.py` is stale**: it passes a pydantic model
  where `DuckDBInferFn` now takes an Arrow schema, so it errors at build.
- **Vectorized `apply_batch`** for `infer_arrow` (UDFs are called per row).
- **Tree scoring** not built: `HistGradientBoosting*`, MLP, a vectorized
  multi-tree walk keeping `tree_span` accumulation order, kNN/kernel SVM.

## Code health

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
