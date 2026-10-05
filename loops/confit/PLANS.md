# Confit plans

My working list: open work only, highest value first within each section.
Facts about what confit does live in `packages/confit/docs/`; questions waiting
on a ruling are records in [decisions/open/](decisions/). Remove an item when it
lands. How the loop runs: [README.md](README.md); the live tickets:
[tickets.md](tickets.md).

## Next

1. **Nightly campaign follow-through.** `.github/workflows/nightly-campaign.yml`
   runs `fuzz.nightly` (400k fresh seeds in four parallel shards, plus the
   metamorphic suite) and files red runs as a "Nightly campaign findings"
   issue. Triage each filed class to a fix, a named exclusion, or an
   open-divergence pin. The first six sharded nights are triaged
   ([reports/2026-10-05-sharded-nightly-triage.md](reports/2026-10-05-sharded-nightly-triage.md)); still open from them:
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

`sql_transform.native` (its own loop, [loops/native/](../native/README.md))
lists what it needs from confit under its PLANS "Needs from confit"; this
loop builds those, ahead of the query classes, since each one unblocks
catalog entries. Today:

Open, low priority (the catalog caps it meanwhile):

- **Two CASE trees in one expression build superlinearly** (asked
  2026-10-05). QuantileTransformer's entry, `0.5*(tree(c) - tree(-c))` with
  `c = coalesce(x, NaN)` and q bisection leaves per tree, built in 4.48 s at
  q=2,000 on 49acad5; after #363 it builds in 0.28 / 0.66 / 1.76 s at q =
  500 / 1,000 / 2,000, against 0.10 / 0.21 / 0.46 s for one tree of the
  column: about 3.8 times one tree, growing a little faster than q. Find
  where the remaining superlinear term is (the per-leaf `c` is shared now)
  before the catalog lifts its 4,000-quantile cap.

Follow-ups from the shared-subexpression work:

- Sharing covers a stage's projection only: not WHERE, join keys, or the
  `shape="many"` loop; and only trap-free subexpressions (a repeated `sqrt`
  is evaluated where it stands, its trap-free operand once).
- A CASE lowers to blocks even when every arm is a trap-free leaf, so the
  catalog's `coalesce(x, NaN)` per term splits three blocks, and its
  right-nested l2 sum carries its pending partial sums through each:
  catalog l2 serves 64 rows in 3.1 ms at 64 features and 14.5 ms at 128
  (l1, whose sum is left to right, 2.5 ms at 128). Lowering such a CASE to
  `select` would make it linear.

Delivered:

- Sharing (#363) no longer hoists what only CASE arms hold: a subexpression
  is shared when evaluated unconditionally and more than once, or twice
  behind one gate (the CASE conditions on its path, hash-consed, so a
  `null_when` function's field reads gate alike). QuantileTransformer with
  three instances serves a 64-row call in 1,388 µs again (21,480 after
  #363, 1,153 before); Normalizer l1/l2/max at 32 features still build in
  0.1-0.3 s. A value shared behind a gate is computed for every row,
  including rows that do not open the gate (one evaluation, where those
  rows paid none); hoisting into the gated arm would avoid it.
- `round(DOUBLE)` (`SKind::Round`) is trap-free, and `sin`/`cos`/`tan`
  under a CASE guard that rules out the infinities (`abs(x) = inf` passed,
  `abs(x) < k` or `abs(x) <> inf` taken) cannot raise, so a struct read does
  not evaluate them beside the read field: the catalog's guarded-`sin`
  FunctionTransformer at 64 features serves a 64-row call in 0.62 ms
  (1,370 µs per row before).
- `greatest`/`least` over I64, F64 or VARCHAR arguments lower as a running
  extreme (`SKind::Extreme`): linear in the arguments, each evaluated once in
  order. The catalog's max norm, `greatest(abs(x1), ..., abs(xn))`, builds in
  0.011 s at 128 arguments (6.1 s before) and 0.1 s at 1,024 (refused past
  the size limit at 256 before). BOOLEAN, DECIMAL and the i128 lane keep the
  flat CASE.
- The early size refusal (#358) verifies the oversized program only under
  debug assertions, as the tests and the nightly build it; a release build
  refuses the 520-lane test program in 2.5 s (6.0 s with the verify). With
  #363, Normalizer l2 at 32 features is no longer refused: it builds in
  0.5 s.
- A subexpression repeated across a stage's projection that cannot trap
  (`can_trap`, now with DOUBLE `abs` on its allowlist) and has at least six
  nodes is computed once per row, before the first item, and read where it
  occurred (`specializer/share.rs`); its value leaves the live stack after
  its last read. A sibling's trap skeleton looks through operations that
  cannot trap themselves (`x_j / CASE .. END` keeps the CASE every lane
  shares; without it l2 at 128 lanes serves in 69 ms). A Normalizer-shaped
  call read lane by lane builds l2 at 128 lanes in 2.5 s and serves 64 rows
  in 2.6 ms; on 477ca2f 32 lanes took 15 s and 169 ms, and 64 failed. Through
  the catalog (`to_native`, bit-exact): l2 at 64 features builds in 1.7 s,
  l1 at 128 in 1.9 s.
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
- A sibling that cannot raise is not evaluated by a field read: DOUBLE
  negation, `exp`, `abs`, `round`, `cbrt`, `floor`/`ceil`/`trunc` are total,
  and `ln`/`log2`/`log10`/`sqrt` under a CASE guard that excludes their
  domain error (`WHEN x <= 0 THEN .. ELSE ln(x)`, `WHEN x > 0 THEN ln(x)`)
  cannot raise. 32 lanes of a `-x` quantile shape serve a 64-row call in
  0.61 ms (9.43 ms before); 64 guarded-`ln` lanes in 0.62 ms (22.6 ms).
- A dropped function returns its JIT memory (`OwnedJit`): every build kept
  2 mappings before, so a process stalled at `vm.max_map_count` after about
  32k builds. 3,000 build/call/drop cycles now hold 479 mappings and 121 MB
  flat (`tests/test_jit_memory.py`).
- A struct-returning SQL function read field by field builds in time
  linear in the lanes read, with or without `null_when` (whose CASE arm
  reads as a call of its own) and with output aliases equal to the field
  names (the scope key counts only aliases the call's arguments name). The
  catalog's degree-2 shape with `null_when` and an `error()` arm: 1,035
  lanes in 0.8 s (16.4 s before), 2,016 in 2.1 s, 3,240 in 4.0 s; no
  expansion cap reached.
- `greatest`/`least` build as one flat CASE (n² in the argument count, was
  4^n).
- A Cranelift size limit refuses by name (`unsupported:`), and as soon as
  the program is lowered and verified when a floor on the virtual registers
  Cranelift needs passes its 2^21 (`exec/size.rs`; checked against Cranelift
  on every debug-assertion compile): l2 Normalizer at 32 features in 4.0 s
  (23.8 s before), a 520-lane struct read in one sum in 6.3 s (88.7 s;
  2026-10-05). Near the cap the floor (about half of Cranelift's count
  there) does not reach it: l1 at 32 features (9.5 s) and l2 at 30 (18 s)
  still refuse from Cranelift, until T1's shared subexpressions shrink them.
- An expression past DuckDB's depth limit (1000) refuses by name; an
  AND/OR chain past 64 terms binds as a balanced tree, and bind, fold and
  lower grow their stack on demand, so a 20000-term chain serves (about
  23 s to build: still superlinear).

## Query classes (in the ruled order: [decisions/closed/next-query-classes.md](decisions/closed/next-query-classes.md))

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
- **i128 lane.** Every integer width but UHUGEINT serves: UTINYINT..UINTEGER
  on the i64 lane (`tests/test_unsigned.py`, seed gate 6 mod 11, tag
  `unsigned`), HUGEINT on an i128 lane with UBIGINT as its narrow width
  (`tests/test_hugeint.py`, seed gate 5 mod 13, tag `wide`). Still refused
  by name: unary minus over an unsigned width (DuckDB wraps it), shifts over
  an unsigned width or HUGEINT, a CAST from DOUBLE to UTINYINT..UINTEGER
  (range-checked before rounding, 255.5 wraps), `round`/`trunc` with digits
  over UBIGINT/HUGEINT, UHUGEINT (literals past HUGEINT, CAST targets), and a
  HUGEINT join key against a DECIMAL build key. Left: exact `sum`/`product`
  at decimal128(38,0) belong to per-row aggregation, not to this lane; the
  i128 ops are one helper call each on the JIT (inline `iadd.i128` with an
  overflow check is the next step if HUGEINT shows up in serving profiles).
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
  A numeric string member converts through f64 and rounds to BIGINT, so past
  2^53 it is the wrong integer (`b IN ('9007199254740993')` over a BIGINT
  `b`); parse it with the integer kernels at the family's type instead.
  Beside UBIGINT/HUGEINT such members refuse by name for this reason.
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
  ([decisions/closed/native-transform-parity-bounds.md](decisions/closed/native-transform-parity-bounds.md)); the route is
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
- **Builds near Cranelift's register cap are slow.** A Normalizer (l1) at
  30 features builds in 22-24 s; at 24 features Cranelift's register
  allocation is 4.3 s, its IR verifier 0.8 s (1.7 s at 32), its egraph 0.8 s
  (2026-10-05). A 100x100 checked-integer matvec took 443 s in Cranelift (not
  investigated). Candidates: the verifier off in release builds, one
  function per group of output columns.
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
