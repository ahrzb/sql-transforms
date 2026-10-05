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
   - `nullif(NULL, x)` drops `x`, which DuckDB evaluates and can trap on
     (seeds 3058298, 3722953). The adoptable-NULL channel has no node that
     evaluates an operand for its trap only;
   - a narrow-width overflow traps on both engines, but confit's text names the
     arrow range instead of DuckDB's `Overflow in multiplication of INT8`
     (narrow left shifts likewise: `Overflow in left shift` / `Left-shift value
     8 is out of range`).

   The 2026-10-05 night (seeds 4200000..4599999) is triaged: its three
   live classes are fixed and pinned (`test_fuzz_smoke.py`); its TIMEOUT and
   OPT_EMULATED cases are the two classes below.

   Still waiting from #305: seed 1159605 (OPT_EMULATED, owner ruling) and the
   TIMEOUT class where confit traps first while DuckDB builds a 2 GiB string
   (seed 1102717, EXCLUDED ratification).
2. **Subquery design, PR 4** (static-only subqueries computed at
   construction) waits on the owner: its 48 measured candidates turned out to
   be unread CTEs, which now serve, so the class has no generated case yet
   (`docs/specs/2026-09-26-row-local-subqueries-design.md`, "Measured
   recovery").

## For the native catalog

`sql_transform.native` (its own loop, `packages/sql-transform/docs/native/`)
lists what it needs from confit under its PLANS "Needs from confit"; this
loop builds those, ahead of the query classes, since each one unblocks
catalog entries. Today:

Nothing the catalog has asked for is open; its PLANS "Needs from confit"
is where new needs land. Follow-ups from the shared-subexpression work:

- A function past Cranelift's limits can panic instead of refusing by name
  (master: `index out of bounds` building 64 l2 Normalizer lanes; now: `Value
  alias loop detected` building `greatest` over 128 arguments, n lanes deep).
  Both should be the named size refusal.
- `greatest`/`least` over n shared arguments keeps all n live across its n²
  blocks (64 arguments: 24 s to build); a tournament (the catalog's
  `row_max`) is linear to lower, but its text is cubic in the row and passes
  the 4M-token macro cap at 64 lanes.
- Sharing covers a stage's projection only: not WHERE, join keys, or the
  `shape="many"` loop; and only trap-free subexpressions (a repeated `sqrt`
  is evaluated where it stands, its trap-free operand once).
- A CASE lowers to blocks even when every arm is a trap-free leaf, so the
  catalog's `coalesce(x, NaN)` per term splits three blocks, and its
  right-nested l2 sum carries its pending partial sums through each:
  catalog l2 serves 64 rows in 4 ms at 64 features and 15 ms at 128 (l1,
  whose sum is left to right, 2.9 ms at 128). Lowering such a CASE to
  `select` would make it linear.

Delivered:

- A subexpression repeated across a stage's projection that cannot trap
  (`can_trap`, now with DOUBLE `abs` on its allowlist) and has at least six
  nodes is computed once per row, before the first item, and read where it
  occurred (`specializer/share.rs`); its value leaves the live stack after
  its last read. A sibling's trap skeleton looks through operations that
  cannot trap themselves (`x_j / CASE .. END` keeps the CASE every lane
  shares), the sibling cache no longer misses once per projection item, a
  read over a guarded body (`null_when`: `CASE .. THEN NULL ELSE
  struct_pack(..) END`) uses that cache too, and a field read reuses its
  call's expansion without substituting the body again. A Normalizer-shaped
  call read lane by lane builds l2 at 128 lanes in about 4 s and serves 64
  rows in 2 ms; on master 32 lanes took 15 s and 169 ms, and 64 failed.
  Through the catalog (`to_native`, bit-exact): l2 at 64 features builds in
  1.9 s, l1 at 128 in 2.5 s.
- A constant CASE result (`CAST('0.0' AS DOUBLE)`) is not a sibling trap
  (`can_trap`).
- A sibling kept for its traps is reduced to its trap skeleton (CASE
  conditions, NULL for trap-free results), each distinct one kept once: n
  reads of n lanes sharing `ELSE error(..)` compile n lanes plus one check.
- A SQL function call read by field expands and parses once per distinct
  call (`frontend/calls.rs`); its siblings' skeletons bind once per call and
  scope. 128 lanes read 128 times build in 0.4 s (13.6 s before both), 128
  lanes over 3 groups (past the 4M-token cap before) in 0.7 s, 256 lanes in
  1.4 s. An unaliased read is named as DuckDB names it, `(f(x)).p`.
- `greatest`/`least` build as one flat CASE (n² in the argument count, was
  4^n).
- A Cranelift size limit refuses by name (`unsupported:`).
- An expression past DuckDB's depth limit (1000) refuses by name; an
  AND/OR chain past 64 terms binds as a balanced tree, and bind, fold and
  lower grow their stack on demand, so a 20000-term chain serves (about
  23 s to build: still superlinear).

## Query classes (in the ruled order: docs/decisions/closed/next-query-classes.md)

The first five are ruled, in this order; the rest follow.

- **Decimal remainders.** Decimal expressions serve
  (docs/specs/decimal-expressions.md). Still refused by name: casts from
  DOUBLE/VARCHAR/BOOLEAN into a DECIMAL (DuckDB's double->decimal rounding
  and its string parser), a DECIMAL join key expression against a VARCHAR
  build key, a decimal key pair whose common type passes 38 digits (DuckDB
  caps it and the per-row cast can fail), and `IN`/`BETWEEN` families capped
  at 38 digits. DECIMAL row columns serve (`tests/test_decimal_rows.py`), as
  join keys too, against DECIMAL, integer and DOUBLE build keys; the
  generator carries one on every seed 3 (mod 7), observable through star
  expansion and the row boundary, and keys that seed's first ON join on it
  (tag `decimal-key`), since decimals are not in the expression grammar yet.
- **i128 lane.** UTINYINT/USMALLINT/UINTEGER serve on the i64 lane
  (`tests/test_unsigned.py`; the generator makes some narrow columns and
  casts unsigned on every seed 6 (mod 11), tag `unsigned`), refusing by name
  unary minus, shifts, and DOUBLE-to-unsigned casts. HUGEINT, UBIGINT and
  UHUGEINT (columns, statics, CAST targets) refuse. Needs i128 arithmetic
  and traps on both backends and exact `sum`/`product` at decimal128(38,0). `-9223372036854775808` serves
  (it is BIGINT on DuckDB); the bare 9223372036854775808, and the minimum
  negated twice, are HUGEINT and refuse.
- **Non-scalar values.** Whole structs, struct literals, bracket access, lists
  and list-valued regex forms (served so far: a list literal read by a
  constant index or projected whole, with elements of one type,
  `tests/test_list_literals.py`; a field read over struct_pack or a CASE); `SELECT s.*` over struct-carrying statics.
  Needs nested output at the Arrow boundary. `decimal256` statics are blocked
  upstream (DuckDB refuses them at Arrow registration).
- **More than one join under `shape='many'`** (`src/specializer/lower.rs`);
  struct keys under `many` (plain equality only in the fan-out loop); struct
  keys whose field-name sets differ refuse where DuckDB serves a constant-empty
  join.
- **Per-row aggregation over matched static rows** (correlated scalar
  subquery; `JOIN` + `GROUP BY` under `shape='many'`). Needs an accumulator
  over the `many` walk, a ruling on when `JOIN`+`GROUP BY` is per-row, and the
  float-reduction bound's algorithm and edge domain
  (`docs/oracle/05-the-comparison-contract.md`).
- **Scalar/correlated/`IN` subqueries, a CTE referenced twice, set
  operations.** Blanket bans today; only batch-dependent forms are out of
  scope. Row-local derived tables and CTEs are in Next.
- **float32** (needs f32 arithmetic, not widening), **temporal types**
  (columns, casts, literals, functions; opaque temporal join keys could use a
  key-only lane), **BLOB**.
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
- **Wrapped-query refusals** the metamorphic suite allowlists
  (rewrite tolerances in `fuzz/exclusions.py`): a struct- or list-valued column inside a
  derived table (about 13% of generated queries when wrapped) and a subquery
  under a `shape='many'` join. Serving struct slots removes the first.

## Evidence and gates

- **No debug-assertion pytest pass.** Lowering invariants are `debug_assert!`s
  the release extension compiles out. The nightly campaign builds with them on
  (`CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS`); the PR gate does not.
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
  sklearn's `transform()`. The parity bound is ruled
  (`docs/decisions/closed/native-transform-parity-bounds.md`); the route is
  the owner's SQL-defined transforms proposal, with Rust kernels for what SQL
  cannot express (see "Waiting on the owner").
- **Serving vs the Python twin.** 1.20–1.80x slower at n=64 on a fresh
  release wheel. Bisect against `a6fa318` (regression vs the twin's change of
  identity). Record the n=64 ratio; the n=1 twin cell swings 2x.
- **Decimal kernels.** Checked decimal casts, capped-width arithmetic and
  the rounding builtins are one helper call per row on the JIT; a DECIMAL
  query costs ~110 ns/row where its DOUBLE spelling costs ~55
  (`docs/reports/2026-09-28-decimal-expressions.md`). Inlining the checks
  is the next step if decimals show up in serving profiles.
- **Vectorized `apply_batch`** for `infer_arrow` (UDFs are called per row).
- **Tree scoring** not built: `HistGradientBoosting*`, MLP, a vectorized
  multi-tree walk keeping `tree_span` accumulation order, kNN/kernel SVM.

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
