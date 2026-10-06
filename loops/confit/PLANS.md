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
   - a narrow-width overflow traps on both engines, but confit's text names the
     arrow range instead of DuckDB's `Overflow in multiplication of INT8`
     (narrow left shifts likewise: `Overflow in left shift` / `Left-shift value
     8 is out of range`).

   The 2026-10-05 night (seeds 4200000..4599999) is triaged: its three
   live classes are fixed and pinned (`test_fuzz_smoke.py`). Its OPT_EMULATED
   cases, with those of the 2026-10-05 campaigns, are confit findings, not
   owner questions (docs/oracle/04: OPT_EMULATED is a finding), triaged
   2026-10-06 (`tests/test_opt_emulated_triage.py`): `nullif(NULL, x)` over a
   trapping `x` refuses by name (seeds 992226, 3058298, 3722953); a trapping
   join condition between the two sides refuses by name (4291817); the
   empty-static witness pads DECIMAL columns, so 1159605 is excused by its
   ruling. Two questions do need the owner, each with a record in
   [decisions/open/](decisions/): the empty-static witness against a
   multi-trap row side (seed 4313391), and the oracle TIMEOUT where confit
   traps first (seeds 4226438, 946454).

   The 2026-10-06 night (seeds 4600000..4999999, replayed on master
   f72214b with the generator that declares SQL functions) is triaged, its
   fixed classes pinned in `test_fuzz_smoke.py`: an expression over a bare
   NULL subquery column types as over a NULL literal, and where DuckDB keeps
   the NULL a node of its own (a CAST, a comparison, CASE, COALESCE, a list,
   a UDF argument), that node does not fold either (4824388, 4946335; with
   it, any NULL that reads a column, such as an all-NULL CASE over a column
   condition, stays a node there: master folded it, answering NULL where
   DuckDB traps, and ran a pure UDF over it at bind, where DuckDB runs it
   per row); NOT over a comparison binds as the negated
   comparison, as DuckDB's transformer reads it, so in a JOIN ON it is a join
   condition (metamorphic 4873273); a LEFT JOIN over a static table with no
   rows never evaluates its ON clause (4712724); a pure UDF in the condition
   of a struct's CASE folds at bind instead of panicking the fold (4704064);
   a field read over a struct stays a call unless DuckDB's binder makes each
   NULL a constant (`outputs::DuckNulls`; 4848122). Its TIMEOUT cases are the
   open oracle-TIMEOUT record's mechanism, added to it as evidence with a
   witness that covers a trap inside the long string's own expression; its
   OPT_EMULATED case (4889739), an INNER join whose static keys are all NULL,
   asks the owner to extend the empty-static ruling
   ([decisions/open/all-null-key-static.md](decisions/open/all-null-key-static.md)).
   Still open from it:
   - A field read over struct_pack keeps only the siblings that can trap, so
     a NULL field folds where DuckDB's struct_pack, which reads every field,
     does not: `struct_pack(f := a, g := CAST(NULL AS BIGINT)).g + (a +
     9223372036854775807)` returns rows where DuckDB traps (master too).
   - A regexp call over a NULL literal pattern is a NULL constant in confit,
     but DuckDB calls it per row (the regexp functions handle a NULL
     themselves), so it reads the subject and does not fold: `CAST((s ~
     NULL) AS BIGINT) + (a + 9223372036854775807)` and the same over
     `regexp_extract(s, NULL)` return rows where DuckDB traps, and over
     `regexp_matches(s, NULL)` they are OPT_EMULATED (master too).
   - TIMEOUT 4667128 is no trap: the request table has no rows and both
     readings agree, but the optimizer-on reading folds a constant
     `repeat('0', 2147483647)` inside a SQL function at plan time (about
     0.5 s per 10^8 characters). The bracket reading could run under its own
     time limit, so that a slow bracket costs only the DIVERGE_OPT label.
   - DuckDB's hash join also skips the ON clause of a LEFT JOIN whose static
     keys are all NULL, while confit still evaluates it (not yet seen in a
     campaign; an AnyJoin reads such a static row by row, so the plan shape
     decides).
2. **Subquery design, PR 4** (static-only subqueries computed at
   construction) waits on the owner: its 48 measured candidates turned out to
   be unread CTEs, which now serve, so the class has no generated case yet
   (`docs/specs/2026-09-26-row-local-subqueries-design.md`, "Measured
   recovery").

3. **Owner docs in simple English.** The docs the owner reviews or maintains
   follow the `simple-english` skill. Next: `packages/confit/README.md`, then the oracle
   and spec docs, one file at a time. Closed and postponed decision records stay as
   ruled. `loops/native/` is the native loop's to rewrite.

## For the native catalog

`sql_transform.native` (its own loop, [loops/native/](../native/README.md))
lists what it needs from confit under its PLANS "Needs from confit"; this
loop builds those, ahead of the query classes, since each one unblocks
catalog entries. Today:

Open, low priority (the catalog caps it meanwhile):

- **A trap in one field of a SQL function's struct costs each field read**
  (asked by the native loop 2026-10-06; repro
  `/mnt/project-files/transforms-loop/confit-trap-field-reads.py`). At 800
  fields, reading every field builds in 0.21 s without a trap and 0.53 s
  with the catalog's id trap and a trap on the parameters (the whole struct:
  0.05 s either way; release build, f72214b plus this triage). Message the
  Transforms Loop thread when it changes, so it can measure the catalog
  again.

- **Closed 2026-10-05: two CASE trees in one expression.** The catalog now
  spells each QuantileTransformer feature as one search tree (native #391;
  0.54 / 1.3 / 3.2 s at 1,000 / 2,000 / 4,000 quantiles), so it no longer
  needs this. What remains general is Cranelift's superlinear compile of a
  deep CASE tree (`define_function`: 1.1 s at q=1,000, 4.6 s at q=2,000 on a
  program linear in q; IsotonicRegression at 20,000 thresholds: 22 s), on
  the board as its own item.

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

- An unaliased item is named after its text in the query as written, as
  DuckDB names it (`sc(x)`, `(pair(x)).lo`, `sc(x) + 1`), not after the
  expanded body (`frontend/mod.rs` parses the query as written beside the
  expanded one and pairs them level by level). A name never spells a let
  out, so the name cap is gone. The campaign generator now declares SQL
  functions (`gen._sql_functions`, seeds with `seed % 19 == 6`, tagged
  `sql-function`): a scalar or struct body over cast parameters, with one
  optional let, each called by its query. On its 13,361 seeds of
  5,200,000..5,479,999 it found five classes, all fixed: a field read of a
  struct that DuckDB folds to NULL is an untyped NULL there
  (`outputs::struct_folds_to_null`; foldable means no column anywhere,
  dead arms included, and a NULL argument of a strict call drops its
  columns; it checks the conditions on the path taken first, walks each
  let's text once, and is cached per call and scope, so 250 field reads of
  a constant call over a 20-step recurrence build in 3.5 s, as on master);
  DuckDB qualifies each SELECT item before it expands a macro, so
  an ambiguous column in an unused argument refuses
  (`resolve::qualify_written`); the oracle registers a macro after the
  trees it reads, and a CREATE MACRO bind error is a build refusal; a let
  read is its text for the checks of DuckDB's literal typing (`tinyint +
  48` is TINYINT arithmetic, `lets::through`, memoized per let because a
  recurrence reads its let twice); a let that folds to a string constant
  reads as a VARCHAR, not a string literal (`lets::fold_kept`).
- A value a SQL function body reads more than once binds once
  (`SqlFunction.sql_lets` / `sql_let_body`, `frontend/lets.rs`): each call
  expands it once, the binder binds it once per scope, and `share.rs`
  interns it where it is read, so it is computed once per row where it
  cannot trap (the same DAG as the text spelled out). A value that can trap
  binds again at each read, so each read traps where DuckDB's does; a read
  in WHERE, JOIN ON or a `shape='many'` projection takes the value whole.
  What those spell out counts toward
  one more 4M budget per query (`lets::spend`), so many reads under the cap
  each still refuse together (known-limitations §1). Release build, one container: SplineTransformer
  at degree 5, 7 knots, 32 features builds in 2.2 s (refused at the token
  cap before: 11 MB spelled out), 64 features in 4.5 s; degree 3, 8 knots,
  64 features in 2.3 s (4.0 s with lets off); `periodic` at degree 5, 32
  features in 1.2 s (7.0 s). The native repro `h = h * (2 - h)` builds at
  depth 100 in 0.01 s (depth 18 refused after 2.6 s, 21 after 53 s). A
  CASE per tree read by each of its lanes (RandomTreesEmbedding shape, 100
  trees of depth 5, 2,286 lanes) builds in 0.85 s (4.1 s). Serving time is
  unchanged within noise. Asked by the native loop for SplineTransformer
  and RandomTreesEmbedding.
- `cbrt` is DuckDB's bit for bit on Linux: the kernel calls glibc's, looked
  up in `libm.so.6` (the toolchain's own `cbrt`, which both `f64::cbrt` and
  an `extern "C"` declaration reached, differed on about half of 100,001
  draws by up to 3 ulps; reported by the native loop, #388). A sweep of
  exp, ln, log2, log10, sqrt, sin, cos, tan, pow, log(b, x), fmod, `//` and
  `%` over 100,000 draws each found no other difference.
- A shared value is computed just before the first item that reads it and
  dropped after its last reader, so it lives only across the items that use
  it (before, every one was computed before the first item and rode every
  later item's blocks). The SplineTransformer-shaped repro (320 struct
  lanes, 9-arm CASE polynomials) builds in 1.4-1.5 s at 4, 8, 16 or 32
  parameters (2.9 s at 4 and 4.3 s at 32 before); QuantileTransformer
  serves its 3-instance step in 1,135 µs.
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
- **Struct values (T6).** A whole struct is one output value: struct
  columns and nested struct fields in every spelling, static struct columns
  (NULL on a LEFT miss), a relation's row struct, struct-valued UDF and SQL
  function calls, struct_pack and `{'k': v}` over them, and a CASE over
  structs of one type (`tests/test_struct_outputs.py`; generator gate 2 mod
  17, tag `struct-value`). Still refused by name: a struct in a derived table
  or CTE (phase 1 carries scalar slots), CASE arms of different struct types
  (DuckDB unifies them), `struct_pack(...) IS NULL` and IS NULL over a CASE
  of structs (DuckDB builds the fields first, so a field can trap), a whole
  struct as an operand (`s = s`, `coalesce(s, t)`), `row(...)` and an
  unnamed struct_pack argument, opaque or list leaves, and the static side of
  a merged struct USING key.
- **Lists.** List columns, lists inside structs, and list-valued regex forms
  (served so far: a list literal read by a constant index or projected whole,
  with elements of one type, `tests/test_list_literals.py`). `decimal256`
  statics are blocked upstream (DuckDB refuses them at Arrow registration).
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
- **Branches that need not be branches.** Each CASE lowers to a branch, and
  Cranelift's build time grows with the branches. A forest of 100 depth-5
  trees (`case_forest.py` in #422) builds in 1.0 s as one list of
  `CAST(leaf = k AS DOUBLE)`, in 3.0 s with `CASE WHEN leaf = k THEN 1.0
  ELSE 0.0 END` fields, and in 4.2 s read field by field: a field read under
  a `null_when` binds as `CASE WHEN g IS NULL THEN NULL ELSE <field> END`,
  one branch per read. Where the value cannot trap (`plan::can_trap`), the
  read could take the guard as its NULL flag, as a list or struct output
  does since #422, and a CASE over constants could lower to a select.

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
