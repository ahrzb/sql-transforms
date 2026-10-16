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
   ruling. Two questions needed the owner. The empty-static witness now
   accepts any trap, so the multi-trap row side of seed 4313391 is excused
   ([decisions/closed/empty-static-witness-trap-order.md](decisions/closed/empty-static-witness-trap-order.md)).
   The oracle TIMEOUT where confit traps first (seeds 4226438, 946454) is
   excused by its ruling: once the oracle runs past the time limit, a new
   process reads it again with each string length capped at 1,000
   ([decisions/closed/oracle-timeout-before-a-trap.md](decisions/closed/oracle-timeout-before-a-trap.md)).

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
   oracle-TIMEOUT record's mechanism, which its ruling now excuses (the
   witness covers a trap inside the long string's own expression too); its
   OPT_EMULATED case (4889739), an INNER join whose static keys are all NULL,
   is excused: the owner extended the empty-static ruling to such a join
   ([decisions/closed/all-null-key-static.md](decisions/closed/all-null-key-static.md)).
   Still open from it:
   - DuckDB's hash join also skips the ON clause of a LEFT JOIN whose static
     keys are all NULL, while confit still evaluates it (not yet seen in a
     campaign; an AnyJoin reads such a static row by row, so the plan shape
     decides).

   The eight nights of the pause (2026-10-07..14, seeds 5000000..8199999,
   all on master b4b8d03) are triaged in
   [reports/2026-10-14-milestone.md](reports/2026-10-14-milestone.md). 46 of
   their 65 gated cases have seeds (issue #435 and the shard logs; the rest
   are only in the run artifacts, which the proxy does not serve). Each
   DIVERGE and OPT_EMULATED case replays the same on b4b8d03 and on 24674cd
   (before #430). Its five live classes are fixed and pinned in
   `test_fuzz_smoke.py`: a HUGEINT product of exactly -2^127 traps unless an
   operand is -2^127, as `Hugeint::TryMultiply` does (6065493, 6112137,
   6263288, 6349920, 6917565, 7511938); BETWEEN in a join's ON clause is
   checked as its two comparisons, which DuckDB's planner splits like any
   two conjuncts (6738617); a field read over a closed value that evaluates
   to NULL, such as a SQL function whose body is NULL under a constant
   condition, types SQLNULL (`binds_to_null`; 5397716, 6212474, 7430849,
   7843947); a pure UDF whose arguments name a column runs per row, since
   DuckDB's binder folds by the text (`args_closed`; 5144477, 6949338); and
   a DECIMAL builtin over a constant NULL DECIMAL, or an all-NULL CASE or
   COALESCE whose parts are operators that bind SQLNULL, binds SQLNULL
   (`null_part`; 5248694, 5923305, 6186379, 6686677).

   Not confit defects: TIMEOUT 6154334 is `levenshtein` over two strings of
   131,068 characters (DuckDB 73 s, confit 55 s; confit runs first, so the
   20 s limit stops confit); in 5068704 and 7731793, as in 4667128, only the
   optimizer-on reading folds a constant string of 2^31 - 1 characters, and
   it now runs under a time limit of its own (`--bracket-timeout`), so they
   keep their optimizer-off verdict;
   24 of the 26 DuckDB TIMEOUTs with seeds are the oracle-TIMEOUT record's
   mechanism (now excused by its ruling), and the two OPT_EMULATED cases (7242549, 7829571)
   are the all-NULL-key record's, now canaries of the empty-static rule.
   Their seeds are in those two records.

   The 2026-10-15 night (seeds 8200000..8599999, master df653be) filed no
   findings: each shard died after its campaign and left no summary. Under
   the nightly's debug assertions, the slot-arity check in
   `materialize_statics` failed for every grouped table whose item is an
   expression over its aggregates (#471): 137 PANIC verdicts in shard 0.
   The metamorphic suite then met such a panic, which `outcome` did not
   catch and the pool could not send back (#478). The night's other gated
   cases, read from the four shard logs and replayed on master 905ee80 with
   the night's generator (df653be): a computed key of a comma join that
   reads a bare name the static table has too now refuses as ambiguous, as
   on DuckDB (8244787; the key bound before the table was in scope); the
   empty-static witness pads a timestamp column with the epoch, so 8522528
   is excused by its ruling. Not confit defects: in 8249704 confit traps on
   `- c2` over -2147483648 while the oracle first builds an `lpad` of
   2^31 - 1 characters in another column (the oracle-TIMEOUT record's
   mechanism, now a canary of its rule; 55 s alone, AGREE_TRAP); in 8517819
   both engines serve the same `repeat(lpad('0', c1, 'a'), c1)` over
   c1 = 32767, a string of 1,073,676,289 characters just under the 1 GiB
   ceiling of a string builder, and the case takes more than the 20 s
   limit (one confit run 5 to 12 s on each of its two backends, DuckDB
   3.6 s, the rows as Python values about 10 s).
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

Open, low priority (the catalog builds within its caps):

- **A cheaper pass over a SQL function's body in Python** (asked by the
  native loop 2026-10-14). `SqlFunction` finds the values that a body
  reads twice or more and writes them as lets (`sql.shared_texts`, #412),
  building every node again. For a balanced CASE tree of 8,000 leaves over
  `coalesce(p, NaN)`, the constructor takes 0.68 s (the body 0.15 s),
  beside 4.34 s for the build and one call; at 4,000 leaves 0.29 s beside
  1.70 s (master). QuantileTransformer at 8,000 quantiles over one feature
  spends 1.7 to 1.9 s there beside a build of 6.9 to 7.3 s, and the
  catalog's build estimates do not count it. Reproduction:
  `/mnt/project-files/transforms-loop/confit-function-body-pass.py`.

- **A struct output read field by field still costs more than the struct
  read whole** (the native loop's need "A struct output read field by
  field, built once for the call"; the part a trap added is closed below).
  Release build of that change, 2026-10-06:
  - 2,000 fields over 64 inputs build in 1.1 to 1.4 s read field by field
    and 0.09 s read whole. RandomTreesEmbedding of 100 trees of depth 5
    over 8 features builds in 1.7 s, and in 0.8 s as a list. Since
    2026-10-14 a field read under `null_when` whose field cannot trap is
    no branch (`lower.rs` `null_guard`); one whose field can trap still
    is. From #456 to 2026-10-15 such reads built in time about
    quadratic in the fields (closed below). Since then the shape of
    `confit-field-reads-456.py` builds 2,000 fields in 0.40 to 0.52 s,
    4,000 in 0.97 to 1.16 s and 8,000 in 3.15 to 3.34 s (three builds
    each): still faster than linear above 4,000 fields, not examined.
  - OneHotEncoder over 8 string features of 50 categories (400 fields)
    builds in 28 s, and in 2.4 s without its input guard (master refused
    it at Cranelift's size limit). The guard's `IN` lists share prefixes
    with the `IN` lists of the fields, so `share.rs` computes each prefix
    at field 0 and carries it to the last field that reads it: about 190
    values ride every block (650,000 block params at 100 fields, 24,000
    without the guard). The native loop plans to spell the guard as one
    substring search a feature (its PLANS "Next" 1), which avoids it. In
    confit, a shared value that would ride many blocks could be computed
    again where it is read.
  - A guard of 64 tests in the `null_when` condition still builds in 3.2 s
    at 2,000 fields, against 1.2 s without a guard
    (`confit-trap-where.py`). In field 0 or in the unknown-id arm it costs
    about nothing, and the catalog keeps its guard in field 0.
  - OrdinalEncoder over 32 string features of 125 categories builds in
    2.8 s. A first version of `compute_once`, which stored every value a
    block passes on, built it in 1.7 to 1.9 s from Python
    (`DuckDBInferFn`). The same rule in a Rust test (the program dumped
    from Python) built it in 2.7 s, like the final rule, so the gap is not
    the store rule; not explained. Its `to_native` takes 25 to 29 s,
    nearly all outside the trial build.

- **A function past 2^24 instructions after Cranelift's optimizer stops
  with a verifier error, not a refusal** (asked by the native loop
  2026-10-06, its T27, #429; the catalog leaves the step Python either
  way). OneHotEncoder over 16 string features of 125 categories (2,000
  fields read one by one; the input guard is one `IN` list of 125 strings a
  feature, in field 0) stops in `to_native` after 33 s with `cranelift
  define: verifier errors (1 in all): inst16780117 (v16829075 = iconst.i64
  1797): instruction result v16829075 is not defined by the instruction`.
  `check_size` (`exec/cranelift.rs`) counts the function as built, so the
  instructions that Cranelift adds after it (here a constant it rebuilds
  in a block) can pass the 24-bit index. Smaller wide encoders get the
  named refusal. Reproduction:
  `/mnt/project-files/transforms-loop/confit-wide-encoder-verifier.py`
  (master e1f15b9 with #429's `encode.py`). Since #430 dropped the
  repeated guard copies, the reproduction builds (1.2 s, the native loop on
  master b4b8d03, 2026-10-14), and no catalog step that the native loop
  measured meets the verifier error. The gap in `check_size` stays: a
  function that Cranelift's optimizer grows past the index still stops
  with a verifier error.

- **Closed 2026-10-15: a query that reads a struct output field by
  field built in time about quadratic in the fields read, since #456**
  (asked first by the native loop 2026-10-14, for its reading 6). The
  catalog's query reads each output field as one SELECT item. The check
  of each item for a subquery (`subquery::has_subquery`, which marks an
  alias that DuckDB refuses to read in the SELECT list) walked the whole
  SQL function call at every field read, since each read holds the
  call's marker. Now each level answers a marker once
  (`Binder::call_subquery`): one marker id always stands for the same
  expansion and the same arguments. Release builds, one container:
  - The native loop's reproduction
    (`/mnt/project-files/transforms-loop/confit-field-reads-456.py`, three
    builds each): 4,000 fields in 1.00 to 1.12 s, against 25.9 to 28.1 s
    on master a2ba98d and 23.6 to 27.8 s with #459; 2,000 fields in 0.41
    to 0.48 s (2.7 to 3.5 s on a2ba98d); 500 in 0.09 to 0.11 s (0.21 to
    0.26 s). The native loop measured 0.92 to 1.01 s at 4,000 fields on
    4bb88f5, before #456.
  - The forest of 100 depth-5 trees (`case_forest.py` in #422), 3,200
    fields read field by field, one run each: 2.2 s with `CAST(leaf = k AS
    DOUBLE)` fields and 2.3 s with CASE fields, against 34 s and 39 s
    with #459 (6.8 s on master 5c613c8, before #422).
  - The catalog's RandomTreesEmbedding of 250 trees of depth 5 over 8
    features (5,982 fields on this loop's data, master 8fbdd11's
    catalog): built in 5.8 to 6.2 s, with `to_native` in 7.6 to 7.8 s,
    against 158 s and 160 s with #459 (one run).
  - `test_a_struct_read_field_by_field_builds_in_about_linear_time`
    (`tests/test_sql_functions.py`) builds 500 and 4,000 fields: 8 times
    the fields must build in less than 25 times the time (10 to 14 now,
    40 to 150 before).

- **Closed 2026-10-14: a value that a SQL function builds once and reads
  many times served slowly, about 60 ns a read** (asked by the native loop,
  its T37). The catalog reads each numeric feature as `coalesce(x, NaN)`, a
  CASE of one arm that `share.rs` leaves unshared (five nodes), so each
  read lowered to a chain of three blocks that carried the whole live
  stack. Each field read of a catalog step is a CASE too, `CASE WHEN c
  THEN NULL ELSE v END` from the step's `null_when`, and each block it
  starts loads again every column that `v` reads (two helper calls a
  load in Cranelift). Now a CASE whose parts cannot trap lowers to
  selects (`lower.rs` `select_case`) when it holds at most 8 operations
  (the IS NULL spelling and `ifnull` with `coalesce`) or is such a guard
  (`null_guard`). Release builds of master a2ba98d and the change, one
  container, 64 rows a call, two runs each unless noted:
  - The native loop's reproduction
    (`/mnt/project-files/transforms-loop/confit-shared-value-reads.py`,
    plus a 256x64 shape): 256 parameters read by 16 fields serve in 2.2
    to 2.3 ms (15 to 22 ms on master; plain reads 2.2 to 2.3 ms) and
    build in 0.43 to 0.47 s (1.4 s; plain 0.4 s); 256x64 in 3.6 to
    4.1 ms (56 to 61 ms; plain 4.0 to 5.1 ms), built in 3.4 to 3.6 s
    (7.9 to 8.1 s; plain 2.5 to 2.7 s); 16x256 in 1.7 ms (15 to 16 ms;
    plain 1.6 ms), built in 1.3 to 1.7 s (2.3 to 2.6 s; plain 0.7 to
    1.3 s). The rest of the build gap is not the selects: a cache of
    them per block changed nothing measurable.
  - The 256x64 shape with an id parameter and a `null_when`, one run:
    4.0 ms, against 28 ms with the small CASE alone and 4.0 ms without
    the `null_when`.
  - The catalog's PCA over 256 features of 64 components (native #458,
    `allow_bound=True`): 4.9 to 5.2 ms (59 ms on master; its twin 59 to
    66 ms on this container), built in 3.2 s (7.3 to 7.5 s), `to_native`
    in 4.5 s (8.6 to 9.0 s).
  - The catalog's Normalizer at 128 features (a follow-up of the
    shared-subexpression work): l1 1.8 to 1.9 ms (2.7 to 3.0 ms on
    master), l2 2.3 to 2.5 ms (3.8 to 4.2 ms), max 1.8 to 2.0 ms
    (2.5 ms).

- **Closed 2026-10-06: a trap in one field of a SQL function's struct
  costs each field read** (repros in `/mnt/project-files/transforms-loop/`).
  Every field read carried its own copy of field 0's input guard, and
  Cranelift computed a value the IR passes on again in each CASE arm that
  reads it, so W reads of a guard over T inputs built in W * T. Now a read
  drops a sibling's check that an earlier item ran on every row that
  reaches it (`share.rs` `repeated_checks`), Cranelift computes once, where
  the IR does, a value read in places none of which dominates the others
  (`exec/cranelift.rs` `compute_once`), and a sibling's trap-free parts
  are lets of the projection (`lets.rs` `let_parts`). Builds, master
  24674cd against the change (release, one machine): 2,000 fields with a
  guard over 64 inputs 5.4 s to 1.1 s (1.1 to 1.4 s for any guard of 0 to
  64 inputs), the guard in the unknown-id arm 6.4 s to 1.4 s, in every
  field 6.6 s to 1.7 s; RandomTreesEmbedding, 100 trees of depth 5, 13 s
  to 2.4 s at 64 features and 7 s to 1.7 s at 8; SplineTransformer at 64
  features 5.3 s to 2.1 s, and at 128 features it builds (4.6 to 12 s
  for degree 3 to 5), where master stops at Cranelift's size limit after
  11 to 20 s; OrdinalEncoder over 32 string features of 125 categories
  30 s to 2.8 s; StandardScaler at 256 features 1.3 s to 0.4 s;
  PolynomialFeatures of degree 2 over 64 features 5.0 s to 1.7 s. A
  64-row call of RandomTreesEmbedding and OneHotEncoder serves about 30%
  faster, seven other classes within noise. The Transforms Loop was told,
  to refit its build estimates (native T26).

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
- **More than one join under `shape='many'`.** No new work: the owner's
  amendment of 2026-10-15 to the ruled order leaves the shapes `filter` and
  `many` as they serve today
  ([decisions/closed/next-query-classes.md](decisions/closed/next-query-classes.md)).
  The joins lower as loops, one for each join, nested in FROM order (`lower_many_loops` in
  `src/specializer/lower.rs`; `tests/test_duckdb_stageb_many.py`; generator
  gate 29 mod 31, tag `many-joins`). Still refused by name: a self-join after
  another join (DuckDB evaluates its build side for the whole batch, a loop
  only for the rows that reach it), a join in a query with a subquery, and
  IS NOT DISTINCT FROM or struct keys under `many` (plain equality only in
  the fan-out loop); struct keys whose field-name sets differ refuse where
  DuckDB serves a constant-empty join.
- **Per-row aggregation over matched static rows** (correlated scalar
  subquery; a `JOIN` with `GROUP BY` stays refused under each shape, and its
  refusal names the scalar subquery:
  [decisions/closed/join-group-by-per-request.md](decisions/closed/join-group-by-per-request.md)). Served: a scalar
  subquery with one `count`, `min`, `max`, integer, BOOLEAN, DECIMAL or
  DOUBLE `sum`, or integer, DECIMAL or DOUBLE `avg`, over all the values of a column or its
  DISTINCT ones (also `bool_and`/`bool_or` and `count_if` over BOOLEAN,
  `bit_and`/`bit_or`/`bit_xor` over integers, and an ORDER BY of columns of
  the table inside the call, 2026-10-16), over one static table, keyed by `=` or IS NOT
  DISTINCT FROM between a bare column of the table and an outer expression
  that cannot trap (the first key's may trap, see below), and filtered by
  WHERE terms that read only the table and cannot trap (the scan below
  serves the other terms). It binds as a hidden
  LEFT join to a table that the build
  filters and groups (`src/specializer/frontend/subquery.rs`, `AggTable` and
  `AggFilter` in `plan.rs`; claim `scalar-subquery-as-join`;
  `tests/test_scalar_subquery.py`; generator gate 33 mod 43, tag
  `scalar-subquery`; 2,400 gated seeds, 0 findings besides 10 DIVERGE_OPT of
  the known trap-elision class). `avg`
  follows DuckDB's finalization exactly (`src/duckdb/avg.rs`): `double` for
  SMALLINT and DECIMAL(<=4, s), x87 `long double` for the rest (sum, divisor
  and quotient rounded to 64 bits, then to 53), so it serves on x86_64 Linux
  only; refused past i128 in Σ|v| (HUGEINT addition traps). The tests hold
  groups per storage type that tell the models apart; 24,000 gated seeds
  after avg gave 0 gated verdicts, with 2,013 cases of a subquery avg (778
  AGREE, 31 AGREE_TRAP). The generator draws the static terms after every
  other draw; on 30,000,000..30,049,999, 672 subquery cases gained terms (309
  AGREE, 10 AGREE_TRAP), and the campaign gave 0 gated verdicts.
  A DOUBLE `sum`/`avg` (ruled 2026-10-15,
  [decisions/closed/double-sum-bound.md](decisions/closed/double-sum-bound.md))
  adds in table order from +0.0 (`Acc::Float` in `duckdb/mod.rs`) and serves
  only as a whole output column (`float_bounds` in `specializer/mod.rs`); the
  campaign builds the case again with `float_bounds=True` and compares within
  B (`confit.compare.within_float_bound`, AGREE klass `float-reduction-bound`).
  Still refused by name: a DOUBLE `sum`/`avg` that is not a whole output
  column, reads a NaN or an infinity, or whose group has sum|v| + B past the
  range, an ORDER BY term inside a call that is not a column of the table,
  `count(DISTINCT *)` (DuckDB's binder error)
  and DISTINCT over a constant, an item that traps over the aggregates of a group or of no rows, holds a
  subquery or calls a SQL function, a UDF or a model, a static term that can
  trap in a WHERE that reads no outer column (DuckDB runs that subquery once
  per batch, an empty batch too), a static term that holds a subquery or
  calls a SQL function, an outer side that reads a joined table, a term that
  can trap in a WHERE subquery beside a comma join keyed on a WHERE term
  (DuckDB runs that subquery over every pair of the cross product) or under
  `shape='many'` (the loops run every join before the WHERE), a VARCHAR key
  against another type, a subquery in FROM, in JOIN ON or inside the
  subquery, the request table, and a scalar subquery under `shape='many'` in
  a two-level query. Of the 59 static terms that campaign refused as
  trapping, 42 held LIKE and 12 a string function; `can_trap` now names
  those that cannot trap (see "Comma-join term order over-refuses"), which
  served 46 more subquery cases on 30,050,000..30,099,999. The first key's
  outer side may trap: DuckDB runs it only if a static row passes the terms
  written before it (the lower half of a BETWEEN; its upper half moves to
  the end), so the build counts those rows into a keyless guard table read
  through one more hidden join, and the key becomes `CASE WHEN count > 0
  THEN key END` (`guarded_key` in `subquery.rs`). A hidden join that the
  WHERE does not read now probes after the WHERE (`JoinSpec::after_where`),
  where DuckDB runs a SELECT-list subquery. On 30,050,000..30,099,999,
  against the verdicts of #469, 15 cases moved from a refusal (10 AGREE, 4
  AGREE_TRAP, 1 DIVERGE_OPT of the known trap-elision class in the outer
  WHERE), no other verdict moved, and the campaign gave 0 gated verdicts;
  7 refusals of a later key that can trap and 2 under `shape='many'` remain.
  The item may be an expression over aggregates (`max(v) + 1`,
  `coalesce(sum(v), 0) + count(*)`): DuckDB evaluates it once per distinct
  outer value, over the aggregates of the matched rows or of no rows
  (measured 2026-10-15), so the build runs it over each group
  (`plan::AggPost`, `post_values` in `duckdb/mod.rs`) and the bind runs it
  over no rows, read as `CASE WHEN JoinHit THEN value ELSE miss END`; a trap
  in either refuses by name. On 30,050,000..30,099,999, against the verdicts
  of #470, the 157 cases whose SQL changed gave 97 AGREE, 58 REFUSED (8 were
  AGREE before), 1 AGREE_TRAP and 1 DIVERGE_OPT, no other verdict moved, and
  0 verdicts gated; the 6,780 cases of 30,100,000..32,099,999 that select an
  expression gave 3,887 AGREE, 141 AGREE_TRAP, 0 gated. A trap over no rows
  could run per request instead (where DuckDB runs it, like the guarded
  key); 19 of those cases refused it, 7 where DuckDB served.
  An aggregate may read DISTINCT values (`count(DISTINCT v)`): the build
  reads one value of each class of DuckDB's equality (-0.0 = 0.0, one NaN;
  measured 2026-10-15; `AggIn::class` in `duckdb/mod.rs`), `min` and `max`
  ignore it, and an output name prints DISTINCT and drops ALL
  (`naming.rs`). On 30,050,000..30,099,999, against the verdicts of #471,
  the 214 cases whose SQL changed kept their verdicts (112 AGREE, 99
  REFUSED, 2 AGREE_TRAP, 1 DIVERGE_OPT), no other verdict moved, and 0
  verdicts gated (a first run found 11 output names without DISTINCT); the
  4,428 cases of 32,100,000..33,099,999 that read DISTINCT values gave 2,527
  AGREE, 94 AGREE_TRAP, 0 gated, and no unexplained metamorphic finding.
  EXISTS and NOT EXISTS serve over the same FROM and WHERE: DuckDB gives
  `EXISTS (SELECT 1 FROM s WHERE w)` the value and the traps of
  `(SELECT count(*) FROM s WHERE w) > 0` (measured 2026-10-15), so the bind
  checks the SELECT items, replaces them with `count(*)` and compares the
  count with 0 (`exists_subquery` in `subquery.rs`). DuckDB evaluates the
  items over the matched rows, so each item must be `*`, a constant or a
  column of the table; an item that can trap, an aggregate item (EXISTS over
  it is always TRUE), LIMIT, ORDER BY, DISTINCT and GROUP BY refuse by name.
  DuckDB's NOT takes the operators above its level, so `NOT EXISTS (q) IS
  NULL` parses as `NOT (EXISTS (q) IS NULL)` (`precedence.rs`; 19 new forms
  in `operator-precedence.json`). The generator draws `[NOT] EXISTS` in place
  of about one subquery in five that is an item or the operand of IS NULL,
  after every other draw. On 30,050,000..30,099,999, against the verdicts of
  #472, the 227 cases whose SQL changed gave 130 AGREE, 87 REFUSED and 10
  AGREE_TRAP (8 were REFUSED before, for the item that EXISTS replaced), no
  other verdict moved, and 0 verdicts gated; the 4,527 cases of
  33,100,000..34,099,999 that read EXISTS gave 2,759 AGREE, 108 AGREE_TRAP,
  0 gated, and no unexplained metamorphic finding.
  `x [NOT] IN (SELECT s.c FROM s WHERE ...)` serves over the same FROM and
  WHERE, from three counts (`in_subquery` in `subquery.rs`): `hit`, with the
  IN test `s.c = x` as one more key after the WHERE terms; `rows`,
  `count(*)`; and `values`, `count(s.c)`. IN is TRUE if `hit > 0`, else FALSE
  if `rows = 0`, else NULL if `values < rows`, else NULL if `x` is NULL, else
  FALSE, and NOT IN swaps TRUE and FALSE (measured 2026-10-15 and pinned).
  DuckDB evaluates `x` once the subquery has an item that is not NULL (its
  hash table holds no NULL): after the whole WHERE, both bounds of BETWEEN
  included, so over a WHERE without a key `x` is a guarded key over every
  term whose guard counts the items that are not NULL (`guarded_key` with
  `Guard::InItem`), and the CASE tests `x IS NULL` only after
  `values < rows`, so only if `values > 0`. #474 guarded `x` by `count(*)`
  and tested it in the same arm as `values < rows`, so confit trapped where
  DuckDB answers NULL, when every item that passes the WHERE is NULL; the
  fix in #475 adds two parity cases that fail on #474 (IN and NOT IN with
  the WHERE `s.f IS NULL`). Over a WHERE with a
  key, DuckDB evaluates `x` for every request of the batch once one request
  matches a static row, with a NULL item too, so an `x` that can trap
  refuses there (batch-dependent). The item must
  be a column of the table, since DuckDB evaluates it over the matched rows;
  a VARCHAR against another type is DuckDB's binder error and refuses, as do
  an `x` that reads a joined table or holds a subquery. The 27 IN forms now
  in `operator-precedence.json` (24 trees, 3 refusals) agree with the parser
  as it was. The generator draws `x [NOT] IN` in place of about one other
  stand-alone subquery in five, after every other draw, with `x` a key's
  outer column or, one time in ten, that column + 1. On
  30,050,000..30,099,999, against the verdicts of #473, the 149 cases whose
  SQL changed gave 83 AGREE, 63 REFUSED, 2 AGREE_TRAP and 1 DIVERGE_OPT (8
  moved from AGREE or AGREE_TRAP to the refusal of an `x` that can trap, 2
  from REFUSED to AGREE), no other verdict moved, and 0 verdicts gated; the
  5,513 cases of 34,100,000..36,099,999 that read IN gave 3,158 AGREE, 146
  AGREE_TRAP, 0 gated, and no unexplained metamorphic finding. 149 of their
  2,175 refusals name an `x` that can trap over a WHERE with a key.
  `x op ANY (SELECT s.c FROM s WHERE ...)`, SOME alike, and `x op ALL (...)`
  serve over the same FROM and WHERE (`quantified_subquery` in
  `subquery.rs`). DuckDB answers ALL as NOT over ANY with the opposite
  comparison, and `= ANY` as IN (so `<> ALL` as NOT IN), which bind through
  `in_subquery`. For another comparison DuckDB runs a nested loop join: TRUE
  if `x op c` holds for the item of a row that passes the WHERE, else FALSE
  over no rows, else NULL if `x` or such an item is NULL, else FALSE
  (measured 2026-10-15 and pinned against a formula over min, max and the
  counts, BIGINT and DOUBLE with NaN and -0.0). It casts both sides to their
  max type (`dec_common`: DECIMAL(9,4) against DECIMAL(38,0) is
  DECIMAL(38,0), EXPLAIN shows `CAST(x AS DECIMAL(38,0))`), not to the
  comparison operator's DECIMAL(38, max scale), so `quantified_residual`
  casts to `dec_common` before `cmp`. Confit keeps `rows` and `values` from
  IN and replaces `hit` by one more hidden join whose table holds
  `count(*)` and the extremes the comparison reads (max under < and <=, min
  under > and >=, both under <>), read through the residual
  `CASE WHEN count > 0 THEN x op extreme END` (hit-guarded, NULL is no
  match). DuckDB evaluates `x` once a row passes the whole WHERE, NULL items
  too (unlike IN), and over a WHERE with a key for the whole batch once one
  request matches: so the CASE evaluates `x` only if `count > 0`, an `x`
  that can trap refuses over a WHERE with a key, and the join counts as a
  trapping key under shape='many' and beside a comma join (`trap_keys`). The
  no-key table holds a row over no input, so the guard matters there.
  `Agg::ties` lets min and max answer a tie of -0.0 and 0.0 or of two NaNs,
  which only a comparison reads. An item whose cast to the compare type can
  trap refuses (DuckDB casts every item: `CAST(t.j AS DECIMAL(38,20)) < ANY
  (SELECT s.v ...)` traps on a BIGINT item past 10^18). `can_trap` now counts
  DECIMAL to DOUBLE and a DECIMAL cast that holds every value of its source
  (`dec_cast_is_total`) as unable to trap; without it every DECIMAL item
  refused. The parser ends `a op ANY (SELECT ...)` at the closing
  parenthesis (`quantified` in `precedence.rs`; 42 new forms in
  `operator-precedence.json`: 39 trees, 3 syntax errors), and unaliased
  items get DuckDB's names (`naming.rs`). The generator draws ANY, SOME or
  ALL in place of about one IN in two, after every other draw; with these
  draws cleared it renders 30,050,000..30,099,999 as master does. On
  30,050,000..30,099,999, against the verdicts of #475 (same build, same
  range), no verdict moved (4 refusal messages changed wording), the 72 ANY
  cases gave 35 AGREE, 2 AGREE_TRAP and 35 REFUSED, and 0 verdicts gated;
  the 5,655 cases of 36,100,000..40,099,999 that read ANY, SOME or ALL gave
  3,235 AGREE, 136 AGREE_TRAP, 2,255 REFUSED (152 name an `x` that can trap
  over a WHERE with a key), 23 DIVERGE_OPT, 0 gated, and no unexplained
  metamorphic finding. Each part fails its tests when removed alone: the
  ties flag (10 tests), the guard `count > 0` (12), a guard that counts only
  items that are not NULL (12), the DECIMAL arms of `can_trap` (38), the
  `trap_keys` entry (5), the refusal of a keyed `x` that can trap (4), and the
  refusal of an item cast that can trap (1).
  Found on master while measuring the compare type: `cmp` compares two
  DECIMALs at a capped width at `dec_common` (DECIMAL(38,0) for
  DECIMAL(9,4) against DECIMAL(38,0), which rounds x), where DuckDB's
  comparison keeps the scale (DECIMAL(38,4)) and traps when the integer part
  does not fit: DIVERGE_VALUE for `CAST(1 AS DECIMAL(9,4)) + 0.5 <
  CAST(2 AS DECIMAL(38,0))`, DIVERGE_TRAP for an operand 10^35 (scalar
  subquery, comma join, literal). Fixed next: `cmp` compares at
  `cmp_common` (`frontend/decimal.rs`), DuckDB's `BindComparison`:
  `dec_common`, except that two DECIMALs past 38 digits compare at
  DECIMAL(38, max scale), where the cast of the side with more than
  38 - scale integer digits traps per row. EXPLAIN (2026-10-15) shows these
  casts under `=`, IS DISTINCT FROM, BETWEEN, an IN list, a simple CASE and
  nullif, which all bind through `cmp` or refuse past 38 digits
  (`unify_family`, `promote_key`); greatest, CASE results and ANY keep
  `dec_common`. Two probes of 1,162 cases (comparisons, guards, filters,
  subqueries, joins; scratch n1019/dec/probe.py, probe2.py) gave 522
  DIVERGE_VALUE and 62 DIVERGE_TRAP on master d8b8dab, 0 gated with the
  fix. 13 cases moved to REFUSED, as the comparison can now trap: a static
  WHERE term of a subquery (`s.sb > 1.5`), a JOIN ON comparison between the
  two sides, and a comma-join WHERE term before the key. #476's
  `dec_cast_is_total` had made master serve them (wrongly). 1 case moved to
  DIVERGE_OPT. 31 new tests in `test_decimal_expressions.py`, 29 fail on
  master; on 30,050,000..30,099,999 no verdict moved (the generator never
  compares two DECIMALs past 38 digits).
  The item may read the columns of the one row that a request matches, with
  no aggregate (`(SELECT s.v * 2 FROM s WHERE s.k = t.k)`). DuckDB
  decorrelates it into a SINGLE join: a request that matches more than one
  row traps (under a CASE that is not taken too), one that matches none reads
  NULL without evaluating the item, and with no key more than one passing row
  traps on every call, an empty batch included (measured 2026-10-15, three
  pins). The bind reads each column the item names through `min(col)` and the
  table counts each group's rows last (`AggPost::row`); the build refuses a
  group of more than one row by name, before the ties check (`more_rows` in
  `duckdb/mod.rs`), and runs the item over the groups of one row only, so the
  keyless group of no rows and a miss read NULL. Refused by name: a column of
  the outer query (a name of no table answers the binder's error), the whole
  row of the table (`SELECT s FROM s`, unless `s` has a column `s`), a field
  or an element of one of its columns, a bare NULL item (DuckDB's SQLNULL),
  and the types that `min` does not serve. The generator draws such an item in
  place of the aggregate of about one other subquery in five that stands
  alone, a column two times in three, and two times in three keeps only the
  first row of each key in the table. On 8,600,000..8,624,999, against the
  verdicts of master 661aedb, no other seed moved; the 85 cases that draw the
  item gave 45 AGREE, 38 REFUSED and 2 AGREE_TRAP (48, 35 and 2 with their
  aggregates), and 132,906 metamorphic pairs gave no finding. The 12,747
  cases of 40,100,000..44,099,999 that draw it gave 6,843 AGREE, 245
  AGREE_TRAP, 5,595 REFUSED (952 name a key of more than one row: DuckDB
  traps in 839, serves in 112, where no request matches the key), 45
  DIVERGE_OPT of the known trap-elision class, 19 EXCLUDED, 0 gated, and
  72,066 metamorphic pairs gave no finding. All 5,798 confit tests pass on an
  unoptimized debug build, under `ulimit -s 262144`: with 8 MB of stack the
  two 6,000-term AND/OR chains of `test_expression_depth.py` overflow it there.
  ORDER BY, LIMIT and OFFSET serve after the WHERE of such an item
  (`(SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o DESC NULLS FIRST LIMIT 1
  OFFSET 1)`). Each ORDER BY term is a column of the table or the item (its
  position 1, or its alias, which DuckDB binds before a column of that
  name); LIMIT and OFFSET take a whole number or NULL, and `LIMIT ALL` turns
  into `LIMIT NULL` in a token rewrite (`rewrite_limit_all`), which also gives
  an unaliased subquery DuckDB's output name; a top-level `LIMIT ALL` or
  `LIMIT NULL` now serves as no LIMIT. The bind keeps them in
  `AggPost::order` (`RowOrder` in `plan.rs`); the build (`materialize_rows`
  in `duckdb/mod.rs`, now #480's path too) runs the item over every row of
  every group first, as DuckDB does before LIMIT (it traps over a row that
  LIMIT drops), sorts each group in DuckDB's order (NULLS LAST by default
  under ASC and DESC, NaN greatest, -0.0 = 0.0, VARCHAR by bytes, FALSE <
  TRUE), and refuses by name a group that keeps more than one row (DuckDB
  traps for a request that matches it) and a tie of the kept row with a row
  whose item has other bits (DuckDB leaves the order of ties unspecified).
  Under a key, DuckDB keeps the rows numbered offset < r <= offset + limit
  with a BIGINT sum that wraps, so `LIMIT NULL OFFSET 1` keeps no row:
  reproduced, and listed in `docs/oracle/02-inherited-quirks.md`. Refused by
  name: ORDER BY or LIMIT over an aggregate item and in EXISTS, IN and ANY,
  ORDER BY ALL, a term that is an expression or reads the outer query, a
  position other than 1, `LIMIT m, n`, and a LIMIT or OFFSET that is not a
  whole number or NULL. With the optimizer on, DuckDB plans one ORDER BY
  term without NULLS FIRST and no OFFSET as arg_min/arg_max with n = the
  LIMIT, which raises for n >= 1,000,000 (LIMIT NULL and ALL included); the
  oracle serves, so these cases are DIVERGE_OPT (ledger:
  subquery-limit-ceiling). The generator adds the clauses to about one item
  over one row in two (the table then keeps all its rows). On
  8,600,000..8,624,999 no other seed moved against #480's campaign; the 52
  cases that draw the clauses gave 25 AGREE, 25 REFUSED and 2 AGREE_TRAP, and
  132,906 metamorphic pairs gave no finding. The 6,453 cases of
  40,100,000..44,099,999 that draw them gave 3,371 AGREE, 117 AGREE_TRAP,
  2,928 REFUSED, 26 DIVERGE_OPT (4 of them the arg_min class above), 11
  EXCLUDED and 0 gated; of the refusals, 402 keep more than one row of a key
  (DuckDB traps in 369) and 200 name a tie; 36,690 metamorphic pairs gave
  no finding.
  Measured 2026-10-14 (optimizer off): DuckDB evaluates the subquery's WHERE in
  written order over (distinct correlated values) x (static rows), so a static
  predicate that can trap before the key traps on unmatched rows; the
  aggregate argument runs on matched rows only; an empty static table or an
  empty batch evaluates nothing; the subquery runs for every row that reaches
  its operator, under a CASE guard too (after WHERE in the SELECT list, before
  WHERE inside it); an uncorrelated one traps even on an empty batch.
  Measured 2026-10-15: a WHERE subquery under a comma join runs over every
  pair of the cross product, before the WHERE matches the comma key.
  The scan (owner idea, decisions thread 2026-10-15 20:17Z, "say 100 or so
  items"): a WHERE term that compares the table with the request other than
  by a key serves by a scan of at most 100 rows for each request
  (`Binder::scan_value` in `subquery.rs`; `emit_fold` and `many_fold` in
  `lower.rs`; the build check `scan_limit` in `src/duckdb/mod.rs`; tests
  `test_a_scan*`; the generator adds such a term to about one subquery in
  four that selects an aggregate or is EXISTS). It serves count, integer,
  BOOLEAN and DECIMAL(<=36) sum, min and max over integer, DECIMAL and
  VARCHAR, and EXISTS, where no term tells -0.0 from
  0.0 (DuckDB evaluates the subquery once per distinct outer value, with the
  first request's value; pin `test_pin_the_subquery_runs_once_for_minus_zero_and_zero`).
  The same holds for the outer side of a key under every form, so a key
  whose outer side tells them apart refuses too (`double_taint` in the key
  loop of `subquery_value`, 2026-10-16; the left side of IN, ANY and ALL
  runs for each request and needs no check).
  A program with a scan runs on the interpreter only (MultiMap statics).
  Terms that can trap serve in the scan (2026-10-16). DuckDB evaluates the
  WHERE over the pairs of a distinct outer value and a static row, each term
  over the pairs on which the terms before it are TRUE: in written order,
  but the upper half of a BETWEEN after every other term, OR's right side
  only if its left is not TRUE, both sides under NOT, CASE lazily (measured,
  scratch n1041/trap_order.py and trap_or.py). Any trapping term but the
  first key's outer side sends the subquery to the scan; `scan_value` puts
  the terms in that order (`Step`), finds the first one that can trap, and
  from there evaluates every term in the fold's `pred`. The terms before it
  filter at build (a BETWEEN whose upper half falls after keeps its lower
  half, `build_part`) or key the scan, and a first key after only filter
  terms keeps `guarded_key`. A fold that can trap (`Fold::traps`) is forced:
  it runs where its join runs, for each request that reaches the subquery
  (under a CASE too, as DuckDB), and counts in `trap_keys` (refused under
  `shape='many'` and beside a comma join). A WHERE that reads no outer
  column refuses a trapping term: DuckDB runs that subquery once per batch,
  an empty batch included. The probes also found that DuckDB evaluates every
  item of an IN list, where confit stopped at the first TRUE equality in a
  WHERE or a CASE condition (master: OPT_EMULATED for `WHERE b IN (0, a +
  1)`; the generator draws only literal items, so the campaign never saw
  it): the binder wraps a list with an item that can trap in NOT over NOT
  (`expr.rs`), which `emit_truth` lowers as a value. CAMPAIGN_NUMBERS
  Open for the scan, in value order (refusals in the 2026-10-16 prepare-only
  sweep of the 6,000 subquery seeds of 50,000,000.., scratch n1040/sweep): an
  expression over the aggregate (66); DISTINCT (34); `bool_and`, `bool_or`
  and the bitwise aggregates (20); avg and DOUBLE sum (26, float-reduction
  bound); min and max over DOUBLE (about 52; refuse at build a column with
  equal values of other bits); a trapping static term in a WHERE that reads
  no outer column (evaluate it at build in DuckDB's order, refuse only if it
  traps); IN, ANY and ALL; ORDER BY with LIMIT; Cranelift code for
  ProbeRange and ProbeRead. The generator could draw IN-list items that are
  not literals.
- **Other scalar, correlated and `IN` subqueries, a CTE referenced twice,
  set operations.** Refused today, except the scalar subquery, EXISTS and
  IN above; only batch-dependent forms are out of scope. Row-local derived
  tables and CTEs are in Next.
- **float32** (needs f32 arithmetic, not widening), **temporal types**
  (columns, casts, literals, functions; opaque temporal join keys could use a
  key-only lane), **BLOB**.
- **Parse-divergence guards.** Binary operators parse with DuckDB's
  precedence (`src/specializer/precedence.rs`, held to
  `docs/specs/pins-wave5/operator-precedence.json`), so `^` can now serve as
  `pow` (parity tests for `2*x^y`, `-x^2`, `x^y^z` and its name `(x ^ y)`).
  Prefix `~` (DuckDB: looser than `+`, tighter than `<<`, so `~1+1` is
  `~(1+1)`) and `NOT GLOB` could each be one more hook there; `#` does not
  parse in DuckDB either. The regex reject list waits on an RE2-compatible
  engine.

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
- A negated integer literal is a BIGINT to confit, so `lpad(x, -2147483648,
  'p')` refuses as `lpad(VARCHAR, BIGINT, VARCHAR)`. DuckDB types it BIGINT
  too (`typeof`), but casts it to INTEGER as it casts a plain literal
  (measured 2026-10-15; 2 cases of the class 5 part 5 dense campaign).
- A JOIN ON residual that reads the static side only refuses when
  `may_trap` says it can trap, and `may_trap` counts all arithmetic, DOUBLE
  included, though DuckDB answers a DOUBLE division by zero with ±inf
  (measured). So `(0.5e0 / s0.f) >= 0.75` refuses, and since BETWEEN in a
  join's ON clause is checked as its two comparisons, so does
  `(0.5e0 / s0.f) BETWEEN 0.75 AND x` (seed 9047919, served before). Using
  `can_trap`, which knows that DOUBLE arithmetic cannot trap, is a widening
  with its own question of evaluation order (`may_trap`'s comment):
  measure it first.
- **Comma-join term order over-refuses.** `check_comma_order`
  (`src/specializer/frontend/from.rs`) refuses a WHERE term written before a
  key term when `can_trap` says it can trap. On the 2,000 seeds in
  9600008..9684714 that the generator picks (22 mod 41, not 29 mod 31; 1,644
  get the tag `comma-keys`) it refuses 297 cases that agreed before (AGREE or
  AGREE_TRAP): 206 name the generator's BIGINT-max addition, 25 other integer
  arithmetic, a UDF or ln/sin/cos/pow, and 66 only LIKE and string functions
  (`replace`, `ltrim`, `upper`, `reverse`, `length`, `strpos`, `concat`,
  `repeat`, `lpad`), many of them constant. `can_trap` now names LIKE
  without ESCAPE, GLOB, concat, trim, the searches, replace, translate, a
  VARCHAR slice, upper, lower, the lengths, unicode, ord, ascii,
  strip_accents and reverse (each measured: `tests/test_string_ops_cannot_trap.py`);
  substr, lpad/rpad, repeat, a VARCHAR subscript and the edit distances can
  trap. On 30,050,000..30,099,999 that served 25 more comma-join cases, and
  one more DIVERGE_OPT of the trapping-key-first class. To serve more, each
  measured on DuckDB 1.5.5 first: (1) `fold` folds constant string calls
  (`lpad('a', 3, 'x')`), so a constant term whose call can trap, but not on
  its constants, serves; (2) a prefix of WHERE terms that read only the
  request row runs before the probe (DuckDB evaluates such a term over every
  pair, with one answer for all the
  static rows; an empty static table is the `empty-static-trap-timing`
  exclusion); (3) under `shape='many'`, a refused order runs as DuckDB runs
  it, a loop over every static row with the WHERE terms in written order.
- **Wrapped-query refusals** the metamorphic suite allowlists
  (rewrite tolerances in `fuzz/exclusions.py`): a struct- or list-valued column inside a
  derived table (about 13% of generated queries when wrapped) and a subquery
  under a `shape='many'` join. Serving struct slots removes the first.

## Evidence and gates

- **No debug-assertion pytest pass.** Lowering invariants are `debug_assert!`s
  the release extension compiles out. The nightly campaign builds with them on
  (`CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS`); the PR gate does not. That cost
  the 2026-10-15 night: the check that #471 broke fails in
  `test_scalar_subquery.py` under debug assertions. The confit tests take
  3 min under them on 4 workers, after a build of about 6 min in a target
  directory of its own. Until a gate runs them, run `tests/` under such a
  build before merging a change to lowering or to the materializer.
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
- **Branches that need not be branches.** A CASE lowers to a branch unless
  it cannot trap and is small or a NULL guard (`lower.rs` `select_case`,
  2026-10-14), and Cranelift's build time grows with the branches. A
  forest of 100 depth-5 trees (`case_forest.py` in #422) builds in 0.9 to
  1.1 s as one list, of `CAST(leaf = k AS DOUBLE)` or of `CASE WHEN leaf =
  k THEN 1.0 ELSE 0.0 END` (2.5 s before the select form). What still
  branches, besides a CASE that can trap: a CASE that is not a NULL guard
  and holds more than 8 operations, such as a deep CASE tree, or holds a
  cast (a cast to VARCHAR would cost every row).

## Authoring boundary (`sql_transform`)

The current authoring boundary is in
[packages/sql-transform/spec/](../../packages/sql-transform/spec/README.md).
Use that contract for fit, composition, supported decorrelation and bounded
window marginalization; the engine contract remains separate.

- Admission-ladder headroom: step semantics for order-keyed windows off the
  training support; static-table joins with frozen composition; IN-subqueries
  as fitted sets; star bundles into transformers; typed takes.
- Composition directions not yet law (packages/sql-transform/spec/composition.md): FROM-position
  templates, frozen-artifact inlining, `as_udf()`, leakage/cross-fitting.
- BIT, TIMETZ and UNION partition keys cannot round-trip through the params
  table (xfail in `sql_transform/_marginal_test.py`).
- Reverse dialect frontends (`parse(sql, dialect=SPARK|BIGQUERY)`) are unbuilt;
  the BigQuery L3 execution leg is not written (wire the Spark leg's seam
  through the BigQuery client: `load_table_from_dataframe`, run the printed
  SQL, compare `rows_of()`).
